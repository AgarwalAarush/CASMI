"""CPU spectral-library baseline for CASMI 2026."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
except ImportError:
    Chem = None


PRECURSOR_PPM = 10
FALLBACK_PPM = 1000
FRAGMENT_DA = 0.02
TOP_K = 25
VALIDATION_SOURCE = "enveda-np-examples"
PEAK_COLUMNS = ["ms2_mzs", "ms2_normalized_intensities"]
QUERY_COLUMNS = ["adduct", "precursor_mz", *PEAK_COLUMNS]
LABEL_COLUMNS = ["inchikey14", "normalized_smiles", "ingest_lib"]
TAUTOMERS = rdMolStandardize.TautomerEnumerator() if Chem else None


@lru_cache(maxsize=400_000)
def connectivity(smiles, provided_key):
    if Chem is None:
        return provided_key
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToInchiKey(TAUTOMERS.Canonicalize(molecule)).split("-")[0] or None


def clean_peaks(mzs, intensities):
    mzs = np.asarray([] if mzs is None else mzs, dtype=np.float64)
    intensities = np.asarray([] if intensities is None else intensities, dtype=np.float64)
    if mzs.ndim != 1 or mzs.shape != intensities.shape:
        return mzs.ravel()[:0], intensities.ravel()[:0]
    valid = np.isfinite(mzs) & np.isfinite(intensities) & (mzs > 0) & (intensities > 0)
    mzs, intensities = mzs[valid], intensities[valid]
    order = np.argsort(mzs)
    norm = np.linalg.norm(intensities)
    if norm:
        intensities /= norm
    return mzs[order], intensities[order]


def spectral_signature(peaks):
    mzs, intensities = peaks
    digest = hashlib.blake2b(digest_size=16)
    digest.update(mzs.round(5).tobytes())
    digest.update(intensities.round(5).tobytes())
    return digest.digest()


def cosine(query, reference):
    """Greedily pair peaks by intensity product, using each peak at most once."""
    query_mz, query_intensity = query
    ref_mz, ref_intensity = reference
    lower = np.searchsorted(ref_mz, query_mz - FRAGMENT_DA)
    upper = np.searchsorted(ref_mz, query_mz + FRAGMENT_DA, side="right")
    pairs = [
        (query_intensity[i] * ref_intensity[j], i, j)
        for i in np.flatnonzero(upper > lower)
        for j in range(lower[i], upper[i])
    ]
    query_used, ref_used = set(), set()
    score = 0.0
    for product, i, j in sorted(pairs, reverse=True):
        if i not in query_used and j not in ref_used:
            score += product
            query_used.add(i)
            ref_used.add(j)
    return min(1.0, float(score))


@dataclass
class Molecule:
    truth: str | None = None
    spectra: list = field(default_factory=list)
    candidates: dict = field(default_factory=dict)
    fallback: tuple | None = None

    def add_spectrum(self, row):
        mz = row["precursor_mz"]
        if row["adduct"] and mz and np.isfinite(mz) and mz > 0:
            self.spectra.append((
                row["adduct"], mz,
                clean_peaks(row["ms2_mzs"], row["ms2_normalized_intensities"]),
            ))

    def ranked(self, excluded=()):
        return sorted(
            (hit for key, hit in self.candidates.items() if key not in excluded),
            key=lambda hit: (-hit[0], hit[1], hit[2]),
        )[:TOP_K]


def validation_queries(train_path, limit):
    # Use natural-product timsTOF queries and reserve their source for evaluation.
    table = pq.read_table(
        train_path, columns=QUERY_COLUMNS + LABEL_COLUMNS,
        filters=[("ingest_lib", "=", VALIDATION_SOURCE)],
    )
    grouped = {}
    for row in table.to_pylist():
        if row["normalized_smiles"]:
            key = connectivity(row["normalized_smiles"], row["inchikey14"])
            if key:
                grouped.setdefault(key, Molecule(truth=key)).add_spectrum(row)
    keys = sorted(key for key, group in grouped.items() if group.spectra)
    if not keys:
        raise ValueError(f"No usable validation spectra in {VALIDATION_SOURCE}")
    rng = np.random.default_rng(2026)
    return [grouped[key] for key in rng.permutation(keys)[:limit]]


def search_library(
    train_path, molecules, *, excluded_sources=None, excluded_keys=(),
    excluded_spectra=(), excluded_key_sources=(), excluded_row_ids=None,
    strict_only=False, identity=connectivity,
):
    by_adduct = defaultdict(list)
    for molecule in molecules:
        for adduct, mz, peaks in molecule.spectra:
            by_adduct[adduct].append((mz, peaks, molecule))
    index = {}
    for adduct, queries in by_adduct.items():
        queries.sort(key=lambda query: query[0])
        index[adduct] = (np.array([q[0] for q in queries]), queries)

    heldout = {
        spectral_signature(peaks)
        for molecule in molecules if molecule.truth is not None
        for _, _, peaks in molecule.spectra
    }
    columns = QUERY_COLUMNS + LABEL_COLUMNS
    window_ppm = PRECURSOR_PPM
    if not strict_only and any(m.truth is None for m in molecules):
        window_ppm = FALLBACK_PPM
    window_frac = window_ppm * 1e-6
    parquet = pq.ParquetFile(train_path)
    blocked_sources = pa.array(sorted(excluded_sources), type=pa.string()) if excluded_sources else None
    excluded_rows = np.array(sorted(excluded_row_ids), dtype=np.int64) if excluded_row_ids else None
    row_offset = 0
    for batch in parquet.iter_batches(batch_size=32768, columns=columns):
        batch_len = batch.num_rows
        if excluded_rows is not None and len(excluded_rows):
            lo = int(np.searchsorted(excluded_rows, row_offset))
            hi = int(np.searchsorted(excluded_rows, row_offset + batch_len))
            if lo < hi:
                keep_mask = np.ones(batch_len, dtype=bool)
                keep_mask[excluded_rows[lo:hi] - row_offset] = False
                batch = batch.filter(pa.array(keep_mask))
        row_offset += batch_len
        if blocked_sources is not None and batch.num_rows:
            source = pc.cast(batch["ingest_lib"], pa.string())
            visible = pc.invert(pc.is_in(source, value_set=blocked_sources))
            batch = batch.filter(visible)
        if not batch.num_rows:
            continue
        masses = batch["precursor_mz"].to_numpy(zero_copy_only=False)
        for adduct, (query_masses, queries) in index.items():
            rows = pc.indices_nonzero(pc.equal(batch["adduct"], adduct)).to_numpy()
            ref_masses = masses[rows]
            starts = np.searchsorted(query_masses, ref_masses / (1 + window_frac))
            ends = np.searchsorted(query_masses, ref_masses / (1 - window_frac), side="right")
            keep = np.flatnonzero((starts < ends) & np.isfinite(ref_masses))
            if not len(keep):
                continue
            near = batch.take(rows[keep]).to_pylist()
            for pos, row in zip(keep, near):
                smiles = row["normalized_smiles"]
                if not smiles:
                    continue
                key = None
                reference_peaks = None
                signature = None
                for mz, peaks, molecule in queries[starts[pos]:ends[pos]]:
                    # Keep validation spectra and their duplicates out of the reference side.
                    if excluded_sources is None and molecule.truth is not None and row["ingest_lib"] == VALIDATION_SOURCE:
                        continue
                    ppm = abs(row["precursor_mz"] - mz) / mz * 1e6
                    strict = ppm <= PRECURSOR_PPM
                    improve_fallback = not strict_only and molecule.truth is None and (
                        molecule.fallback is None or ppm <= molecule.fallback[0]
                    )
                    if not strict and not improve_fallback:
                        continue
                    if key is None:
                        key = identity(smiles, row["inchikey14"])
                        if not key or key in excluded_keys or (key, row["ingest_lib"]) in excluded_key_sources:
                            break
                    if reference_peaks is None and (strict or excluded_spectra):
                        reference_peaks = clean_peaks(row[PEAK_COLUMNS[0]], row[PEAK_COLUMNS[1]])
                        if strict_only and not len(reference_peaks[0]):
                            break
                        if heldout or excluded_spectra:
                            signature = spectral_signature(reference_peaks)
                    if signature in excluded_spectra:
                        break
                    if molecule.truth is not None and signature in heldout:
                        continue
                    fallback_hit = (ppm, key, smiles)
                    if improve_fallback and (molecule.fallback is None or fallback_hit < molecule.fallback):
                        molecule.fallback = fallback_hit
                    if strict:
                        score = cosine(peaks, reference_peaks)
                        hit = (score, ppm, key, smiles)
                        previous = molecule.candidates.get(key)
                        if previous is None or (-score, ppm, smiles) < (-previous[0], previous[1], previous[3]):
                            molecule.candidates[key] = hit


def validate(train_path, limit):
    molecules = validation_queries(train_path, limit)
    search_library(train_path, molecules)
    heldout_keys = {molecule.truth for molecule in molecules}
    ranks = [
        next((i for i, hit in enumerate(molecule.ranked(), 1) if hit[2] == molecule.truth), 0)
        for molecule in molecules
    ]
    unseen_ranks = [
        next((i for i, hit in enumerate(molecule.ranked(heldout_keys), 1) if hit[2] == molecule.truth), 0)
        for molecule in molecules
    ]
    result = {
        "molecules": len(molecules),
        "spectra": sum(len(molecule.spectra) for molecule in molecules),
        "retrieval_recall": sum(m.truth in m.candidates for m in molecules) / len(molecules),
        "top_1": ranks.count(1) / len(molecules),
        "recall_at_25": sum(bool(rank) for rank in ranks) / len(molecules),
        "mrr_at_25": sum(1 / rank for rank in ranks if rank) / len(molecules),
        "unseen_structure_mrr_at_25": sum(1 / rank for rank in unseen_ranks if rank) / len(molecules),
        "connectivity": "RDKit canonical tautomer" if Chem else "provided inchikey14",
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def predict(train_path, test_path, output, *, catalogs=(), ppm=PRECURSOR_PPM,
            formulas_path=None, candidate_output=None):
    if formulas_path is not None and not catalogs:
        raise ValueError("Formula evidence requires a structure catalog")
    molecules = {}
    for row in pq.read_table(test_path, columns=["molecule_id", *QUERY_COLUMNS]).to_pylist():
        molecule_id = row["molecule_id"]
        if not molecule_id:
            raise ValueError("Test contains a missing molecule_id")
        molecules.setdefault(molecule_id, Molecule()).add_spectrum(row)
    if not molecules:
        raise ValueError("Test contains no molecules")
    if catalogs and Chem is None:
        raise ValueError("RDKit is required to deduplicate catalog and library predictions")
    search_library(train_path, list(molecules.values()))

    counts = dict(spectral_hits=0, mass_only=0, placeholder=0)
    catalog_hits = {}
    if catalogs or candidate_output is not None:
        from casmi_catalog import export_candidates, load_formulas, mass_order, merge_candidates, search_catalogs

        if catalogs:
            query_rows = (
                {"molecule_id": molecule_id, "adduct": adduct, "precursor_mz": mz}
                for molecule_id, molecule in molecules.items() for adduct, mz, _ in molecule.spectra
            )
            formulas = load_formulas(formulas_path, molecules)
            _, catalog_hits = search_catalogs(
                catalogs, query_rows, identity=connectivity, ppm_limit=ppm, formulas=formulas,
            )
            counts["molecules_with_catalog_hits"] = len(catalog_hits)
            counts["molecules_with_catalog_guesses"] = 0
        if candidate_output is not None:
            export_candidates(candidate_output, molecules, catalog_hits)

    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["molecule_id", "smiles"])
        for molecule_id, molecule in sorted(molecules.items()):
            ranked_library = {hit[2]: hit[3] for hit in molecule.ranked()}
            hits = catalog_hits.get(molecule_id, {})
            guesses = list(ranked_library.values())
            if hits:
                keys = merge_candidates(ranked_library, mass_order(hits)[:TOP_K], TOP_K)
                guesses = [ranked_library[key] if key in ranked_library else hits[key].smiles for key in keys]
                counts["molecules_with_catalog_guesses"] += any(key not in ranked_library for key in keys)
            if ranked_library:
                counts["spectral_hits"] += 1
            elif guesses:
                counts["mass_only"] += 1
            elif molecule.fallback is not None:
                guesses = [molecule.fallback[2]]
                counts["mass_only"] += 1
            else:
                guesses = ["CCO"]
                counts["placeholder"] += 1
            writer.writerow([molecule_id, ";".join(guesses)])
    print(json.dumps({"submission_rows": len(molecules), **counts}, indent=2), flush=True)
    return counts


def kaggle_run(*, catalogs=(), ppm=PRECURSOR_PPM, formulas_path=None, candidate_output=None):
    train_path, = sorted(
        path for path in Path("/kaggle/input").rglob("train.parquet")
        if path.with_name("test.parquet").exists()
    )
    print(f"Competition data: {train_path.parent}", flush=True)
    validate(train_path, limit=80)
    predict(train_path, train_path.with_name("test.parquet"), Path("/kaggle/working/submission.csv"),
            catalogs=catalogs, ppm=ppm, formulas_path=formulas_path, candidate_output=candidate_output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="directory with train.parquet and test.parquet")
    parser.add_argument("--validate", type=int, metavar="MOLECULES", help="run local validation")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--catalog", type=Path, action="append", default=[])
    parser.add_argument("--ppm", type=float, default=PRECURSOR_PPM, help="catalog precursor tolerance")
    parser.add_argument("--formulas", type=Path, help="CSV with independent molecule_id,formula predictions")
    parser.add_argument("--candidate-output", type=Path, help="write all retrieved candidates to Parquet")
    args = parser.parse_args()
    train = args.data / "train.parquet"
    if args.validate is not None:
        if args.validate < 1:
            parser.error("--validate must be positive")
        if args.catalog or args.formulas or args.candidate_output is not None:
            parser.error("Use casmi_eval.py evaluate to validate with a catalog or export candidates")
        validate(train, args.validate)
    else:
        predict(train, args.data / "test.parquet", args.output, catalogs=args.catalog, ppm=args.ppm,
                formulas_path=args.formulas, candidate_output=args.candidate_output)


if __name__ == "__main__":
    main()
