"""Run and validate the original CASMI CPU submission pipeline."""

import argparse
import csv
import hashlib
import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import pyarrow.parquet as pq

from casmi_baseline import TOP_K, connectivity, predict


def validate_submission(path, test_path):
    expected = set(pq.read_table(test_path, columns=["molecule_id"])[0].to_pylist())
    seen = set()
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames != ["molecule_id", "smiles"]:
            raise ValueError("Expected molecule_id,smiles CSV columns")
        for row in rows:
            molecule_id = row["molecule_id"]
            if molecule_id not in expected or molecule_id in seen:
                raise ValueError(f"Unexpected or repeated molecule ID: {molecule_id!r}")
            structures = (row["smiles"] or "").split(";")
            keys = [connectivity(smiles, None) if smiles.strip() else None for smiles in structures]
            if len(keys) > TOP_K or None in keys or len(keys) != len(set(keys)):
                raise ValueError(f"Invalid, duplicate or too many structures for {molecule_id!r}")
            seen.add(molecule_id)
    if seen != expected:
        raise ValueError(f"Missing predictions for {len(expected - seen)} molecules")
    return len(seen)


def run(data, output, catalogs):
    rdkit_version = version("rdkit")
    if rdkit_version != "2026.3.3":
        raise ValueError(f"Competition identity requires RDKit 2026.3.3, found {rdkit_version}")

    test = data / "test.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = output.with_name(f"{output.stem}.receipt.json")
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        start = perf_counter()
        candidate = Path(tmp) / output.name
        counts = predict(data / "train.parquet", test, candidate, catalogs=catalogs)
        rows = validate_submission(candidate, test)
        receipt = {
            "molecules": rows,
            "catalogs": [path.name for path in catalogs],
            "packages": {name: version(name) for name in ("numpy", "pyarrow", "rdkit")},
            "elapsed_seconds": round(perf_counter() - start, 3),
            "submission_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "counts": counts,
        }
        temp_receipt = Path(tmp) / receipt_path.name
        temp_receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(candidate, output)
        os.replace(temp_receipt, receipt_path)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Directory containing train.parquet and test.parquet")
    parser.add_argument("--catalog", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("runs/submission.csv"))
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.output, args.catalog), indent=2))


if __name__ == "__main__":
    main()
