"""Prepare mass-scoped PubChem shards from the official CID extras.

Only the unlabeled precursor m/z and adduct columns determine the scope.
The same input archives can be used to prepare shards for a new query batch.
"""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv
import pyarrow.parquet as pq

from casmi_catalog import META_RDKIT_KEY, META_SORTED_MASS_KEY, neutral_mass
from casmi_eval import METRIC_RDKIT, _canonicalize_catalog_chunk, check_official_rdkit, file_record


# PubChem and RDKit can use different electron-mass conventions for ions.
SOURCE_MASS_PADDING = 0.002
WORKER_BATCH_SIZE = 512
SHARD_SIZE = 250_000


def mass_windows(paths, ppm):
    windows = set()
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["precursor_mz", "adduct"]):
            for mz, adduct in zip(batch["precursor_mz"].to_pylist(), batch["adduct"].to_pylist()):
                mass = neutral_mass(mz, adduct)
                if mass is not None:
                    width = mz * ppm * 1e-6 + SOURCE_MASS_PADDING
                    windows.add((mass - width, mass + width))
    if not windows:
        raise ValueError("The query files contain no supported neutral masses")

    return np.asarray(sorted(windows), dtype=np.float64).T


def merge_windows(windows):
    merged = []
    for lo, hi in windows.T:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return np.asarray(merged, dtype=np.float64).T


def cid_batches(path, columns, types):
    previous_cid = 0
    with pa.input_stream(str(path), compression="gzip") as stream:
        reader = csv.open_csv(
            stream,
            read_options=csv.ReadOptions(column_names=columns, block_size=8 * 1024 * 1024),
            parse_options=csv.ParseOptions(delimiter="\t", quote_char=False),
            convert_options=csv.ConvertOptions(include_columns=list(types), column_types=types),
        )
        for batch in reader:
            if not batch.num_rows:
                continue
            cid_column = batch["cid"]
            if cid_column.null_count:
                raise ValueError(f"{path}: found a missing PubChem CID")
            cids = cid_column.to_numpy()
            if cids[0] <= previous_cid or np.any(cids[1:] <= cids[:-1]):
                raise ValueError(f"{path}: expected strictly increasing PubChem CIDs")
            previous_cid = cids[-1]
            yield batch, cids


def closest_cids(cids, masses, windows, limit):
    order = np.argsort(masses)
    sorted_masses = masses[order]
    sorted_cids = cids[order]
    lower, upper = windows
    centers = (lower + upper) / 2
    starts = np.searchsorted(sorted_masses, lower, side="left")
    stops = np.searchsorted(sorted_masses, upper, side="right")
    selected = []
    for center, start, stop in zip(centers, starts, stops):
        nearby_cids = sorted_cids[start:stop]
        if len(nearby_cids) > limit:
            distances = np.abs(sorted_masses[start:stop] - center)
            nearby_cids = nearby_cids[np.lexsort((nearby_cids, distances))[:limit]]
        selected.append(nearby_cids)
    return np.unique(np.concatenate(selected))


def select_cids(path, windows, limit):
    lower, upper = merge_windows(windows)
    matches = []
    matched_masses = []
    rows = 0
    next_report = 20_000_000
    for batch, cids in cid_batches(
        path, ["cid", "formula", "monoisotopic_mass", "exact_mass"],
        {"cid": pa.uint32(), "exact_mass": pa.float64()},
    ):
        masses = batch["exact_mass"].to_numpy(zero_copy_only=False)
        indices = np.searchsorted(lower, masses, side="right") - 1
        in_window = (indices >= 0) & (masses <= upper[indices])
        matches.append(cids[in_window])
        if limit is not None:
            matched_masses.append(masses[in_window])
        rows += len(cids)
        if rows >= next_report:
            print(f"[pubchem] Scanned {rows:,} mass records", flush=True)
            next_report += 20_000_000
    cids = np.concatenate(matches) if matches else np.empty(0, dtype=np.uint32)
    window_cids = len(cids)
    if limit is not None and window_cids:
        cids = closest_cids(cids, np.concatenate(matched_masses), windows, limit)
    return cids, rows, window_cids


def selected_smiles(path, selected_cids):
    start = 0
    for batch, cids in cid_batches(path, ["cid", "smiles"], {"cid": pa.uint32(), "smiles": pa.string()}):
        stop = np.searchsorted(selected_cids, cids[-1], side="right")
        if stop > start:
            cid_slice = pa.array(selected_cids[start:stop])
            matches = batch.filter(pc.is_in(batch["cid"], value_set=cid_slice))
            yield from matches["smiles"].to_pylist()
        start = stop
        if start == len(selected_cids):
            break


def canonical_batches(smiles, workers, counters):
    pending = deque()
    chunk = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for structure in smiles:
            counters["matched_smiles"] += 1
            # The mass index describes the complete PubChem record. Dropping
            # fragments after lookup would introduce an unrelated parent mass.
            if not structure or "." in structure:
                continue
            counters["single_fragment_smiles"] += 1
            chunk.append(structure)
            if len(chunk) < WORKER_BATCH_SIZE:
                continue
            pending.append(pool.submit(_canonicalize_catalog_chunk, (chunk, None)))
            chunk = []
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        if chunk:
            pending.append(pool.submit(_canonicalize_catalog_chunk, (chunk, None)))
        while pending:
            yield pending.popleft().result()


def write_shard(directory, index, records):
    rows = sorted(
        (mass, key, smiles, formula)
        for key, (mass, smiles, formula) in records.items()
    )
    table = pa.table({
        "exact_mass": pa.array([row[0] for row in rows], type=pa.float64()),
        "inchikey14": pa.array([row[1] for row in rows], type=pa.string()),
        "smiles": pa.array([row[2] for row in rows], type=pa.string()),
        "formula": pa.array([row[3] for row in rows], type=pa.string()),
    }).replace_schema_metadata({META_RDKIT_KEY: METRIC_RDKIT.encode(), META_SORTED_MASS_KEY: b"true"})
    name = f"pubchem_{index:04}.parquet"
    pq.write_table(table, directory / name, compression="zstd")
    print(f"[pubchem] Wrote {name} with {len(rows):,} structures", flush=True)
    return {"name": name, "structures": len(rows)}


def prepare(args):
    check_official_rdkit()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not math.isfinite(args.ppm) or args.ppm <= 0 or args.workers < 1:
        raise ValueError("Use a positive finite ppm tolerance and at least one worker")
    windows = mass_windows(args.queries, args.ppm)
    if args.max_cids_per_precursor is not None and args.max_cids_per_precursor < 1:
        raise ValueError("The per-precursor CID cap must be positive")
    cids, mass_rows, window_cids = select_cids(args.mass, windows, args.max_cids_per_precursor)
    if not len(cids):
        raise ValueError("No PubChem CIDs matched the query windows")
    print(f"[pubchem] Selected {len(cids):,} of {window_cids:,} CIDs across {windows.shape[1]:,} precursor windows", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent, prefix=".pubchem-") as temp:
        stage = Path(temp) / "catalog"
        stage.mkdir()
        counters = {"matched_smiles": 0, "single_fragment_smiles": 0}
        shards = []
        records = {}
        for batch in canonical_batches(selected_smiles(args.smiles, cids), args.workers, counters):
            for key, mass, smiles, formula in batch:
                record = (mass, smiles, formula)
                if key not in records or record < records[key]:
                    records[key] = record
            if len(records) >= SHARD_SIZE:
                shards.append(write_shard(stage, len(shards) + 1, records))
                records.clear()
        if records:
            shards.append(write_shard(stage, len(shards) + 1, records))
        if not shards:
            raise ValueError("No usable PubChem structures matched the query windows")

        manifest = {
            "source": "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/",
            "inputs": {"mass": file_record(args.mass), "smiles": file_record(args.smiles)},
            "queries": [file_record(path) for path in args.queries],
            "rdkit_version": METRIC_RDKIT,
            "scope": "single-fragment structures in the supplied precursor/adduct mass windows",
            "ppm": args.ppm,
            "source_mass_padding_da": SOURCE_MASS_PADDING,
            "mass_records": mass_rows,
            "cids_in_mass_windows": window_cids,
            "max_cids_per_precursor": args.max_cids_per_precursor,
            "selected_cids": len(cids),
            **counters,
            "shards": shards,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(args.output)
    print(f"[pubchem] Ready: {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass", required=True, type=Path, help="official CID-Mass.gz")
    parser.add_argument("--smiles", required=True, type=Path, help="official CID-SMILES.gz")
    parser.add_argument("--queries", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path, help="new directory for prepared catalog shards")
    parser.add_argument("--ppm", default=10, type=float)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--max-cids-per-precursor", type=int, help="keep this many nearest PubChem CIDs per precursor before canonicalization")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
