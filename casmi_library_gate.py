"""Calibrate and evaluate an original library-lead rule on fixed candidate pools."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import groupby, product
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from casmi_baseline import LABEL_COLUMNS, QUERY_COLUMNS, TOP_K, Molecule, search_library
from casmi_catalog import mass_order, merge_candidates, search_catalogs
from casmi_eval import (
    batch_keys, catalog_membership, file_record, fold_labels, identity_for,
    load_key_cache, load_manifest, score_groups, stable_hash, suitable,
)


FIT_SOURCES = ("enveda-np-examples", "riken", "gnps", "massbank", "mona")
SCORE_GRID = (0.45, 0.55, 0.65, 0.75, 0.85, 0.90, 0.95, 0.98)
MARGIN_GRID = (0.0, 0.025, 0.05, 0.10, 0.15, 0.20)


@dataclass(frozen=True)
class Evidence:
    library: list
    catalog: list

    @property
    def library_keys(self):
        return [hit[2] for hit in self.library]

    def gated_order(self, minimum_cosine, minimum_margin):
        library_keys = self.library_keys
        if self.library and self.catalog:
            best = self.library[0][0]
            runner_up = self.library[1][0] if len(self.library) > 1 else 0.0
            if best >= minimum_cosine and best - runner_up >= minimum_margin:
                catalog_first = [library_keys[0], *self.catalog]
                return merge_candidates(catalog_first, library_keys, TOP_K)
        return merge_candidates(self.catalog, library_keys, TOP_K)


def calibration_cases(fit, identity, allowed, limit, seed):
    # Keep labels from fit only and select positives with another reference source.
    sources = defaultdict(set)
    parquet = pq.ParquetFile(fit)
    for batch in parquet.iter_batches(batch_size=32768, columns=LABEL_COLUMNS):
        for key, source in zip(batch_keys(batch, identity), batch["ingest_lib"].to_pylist()):
            if key in allowed and source:
                sources[key].add(source)
    groups = defaultdict(list)
    for key, available in sources.items():
        if len(available) > 1:
            source = next((source for source in FIT_SOURCES if source in available), None)
            if source:
                groups[source].append(key)
    sampled = {}
    for source, keys in groups.items():
        keys.sort(key=lambda key: stable_hash("library-calibration", seed, key))
        sampled.update((key, source) for key in keys[:limit])
    return sampled


def calibration_queries(fit, identity, sampled, seed):
    selected = defaultdict(dict)
    parquet = pq.ParquetFile(fit)
    for batch in parquet.iter_batches(batch_size=32768, columns=QUERY_COLUMNS + LABEL_COLUMNS):
        keys = batch_keys(batch, identity)
        indices = [
            i for i, (key, source) in enumerate(zip(keys, batch["ingest_lib"].to_pylist()))
            if key in sampled and sampled[key] == source
        ]
        if not indices:
            continue
        for i, row in zip(indices, batch.take(indices).to_pylist()):
            signature = suitable(row, query=True)
            if signature is not None:
                key = keys[i]
                selected[key].setdefault(signature, {**row, "molecule_id": key})
    return [
        row
        for key, rows in sorted(selected.items())
        for _, row in sorted(
            rows.items(), key=lambda item: stable_hash(seed, key, item[0].hex()),
        )[:3]
    ]


def reciprocal_rank(keys, truth):
    return next((1 / rank for rank, key in enumerate(keys[:TOP_K], 1) if key == truth), 0.0)


def calibrate(args):
    manifest = load_manifest(args.split)
    if [file_record(path) for path in args.catalog] != manifest["inputs"]["catalogs"]:
        raise ValueError("Pass the original catalog files in the order in the split manifest")
    if args.limit_per_source < 1:
        raise ValueError("--limit-per-source must be positive")
    cache, _ = load_key_cache(args.key_cache) if args.key_cache else (None, None)
    identity = identity_for(manifest["identity"], cache)
    fit = args.split / "fit.parquet"
    assignments_path = args.split / "assignments.parquet"
    assignments = {
        row["inchikey14"]: row["fold"]
        for row in pq.read_table(assignments_path, columns=["inchikey14", "fold"]).to_pylist()
    }
    allowed = {key for key, fold in assignments.items() if fold == "train"}
    sampled = calibration_cases(fit, identity, allowed, args.limit_per_source, args.seed)
    in_catalog = catalog_membership(args.catalog, identity, set(sampled))
    sampled = {key: source for key, source in sampled.items() if key in in_catalog}
    rows = calibration_queries(fit, identity, sampled, args.seed)
    molecules = defaultdict(Molecule)
    for row in rows:
        key = row["molecule_id"]
        molecules[key].truth = key
        molecules[key].add_spectrum(row)
    search_library(
        fit, list(molecules.values()), strict_only=True, identity=identity,
        excluded_sources=(), excluded_key_sources=set(sampled.items()),
        excluded_keys=assignments.keys() - allowed,
    )
    catalogs, _ = search_catalogs(args.catalog, rows, identity=identity, ppm_limit=10)
    pairs = []
    for key, molecule in molecules.items():
        catalog = mass_order(catalogs[key])
        if key not in catalog:
            continue
        library = molecule.ranked()
        # Both cases use the same spectra and catalog. Hiding the correct
        # spectral hit simulates catalog-only identification.
        masked = molecule.ranked(excluded={key})
        pairs.append((key, Evidence(library, catalog), Evidence(masked, catalog)))
    if not pairs:
        raise ValueError("No usable fit calibration cases")

    def objective(cosine, margin):
        scores = [
            reciprocal_rank(evidence.gated_order(cosine, margin), key)
            for key, positive, masked in pairs
            for evidence in (positive, masked)
        ]
        return float(np.mean(scores))

    thresholds = [*product(SCORE_GRID, MARGIN_GRID), (1.01, 0.0)]
    score, cosine, margin = max(
        (objective(cosine, margin), cosine, margin) for cosine, margin in thresholds
    )
    return {
        "minimum_cosine": cosine,
        "minimum_margin": margin,
        "seed": args.seed,
        "limit_per_source": args.limit_per_source,
        "fit_molecules": len(pairs),
        "fit_molecules_by_source": dict(Counter(sampled[key] for key, _, _ in pairs)),
        "fit_mrr_at_25_with_equal_masked_weight": score,
        "fit_input": file_record(fit),
        "assignments": file_record(assignments_path),
        "catalogs": manifest["inputs"]["catalogs"],
        "note": "Fit-only source holdout with exact-duplicate exclusion. Masked cases simulate catalog-only search. "
                "Cross-source near-duplicates can remain. The two thresholds define a ranking heuristic.",
    }


def read_evidence(path, ids):
    columns = ["molecule_id", "inchikey14", "spectral_cosine", "spectral_ppm", "catalog_ppm"]
    rows = (
        row for batch in pq.ParquetFile(path).iter_batches(columns=columns)
        for row in batch.to_pylist() if row["molecule_id"] in ids
    )
    for molecule_id, group in groupby(rows, key=lambda row: row["molecule_id"]):
        library, catalog = [], []
        for row in group:
            key = row["inchikey14"]
            if row["spectral_cosine"] is not None:
                library.append((row["spectral_cosine"], row["spectral_ppm"], key))
            if row["catalog_ppm"] is not None:
                catalog.append((row["catalog_ppm"], key))
        library.sort(key=lambda hit: (-hit[0], hit[1], hit[2]))
        catalog.sort()
        yield molecule_id, Evidence(library, [key for _, key in catalog])


def paired_changes(rows, candidate, baseline):
    candidate_rr, baseline_rr = [], []
    for row in rows:
        molecule_id, truth = row["molecule_id"], row["inchikey14"]
        candidate_rr.append(reciprocal_rank(candidate.get(molecule_id, ()), truth))
        baseline_rr.append(reciprocal_rank(baseline.get(molecule_id, ()), truth))
    delta = np.array(candidate_rr) - np.array(baseline_rr)
    rng = np.random.default_rng(2026)
    samples = [float(np.mean(rng.choice(delta, size=len(delta)))) for _ in range(2000)]
    return {
        "mrr_at_25_delta": float(np.mean(delta)),
        "mrr_at_25_delta_95pct_interval": np.quantile(samples, (0.025, 0.975)).tolist(),
        "correct_top_1_gained": sum(a == 1 and b < 1 for a, b in zip(candidate_rr, baseline_rr)),
        "correct_top_1_lost": sum(a < 1 and b == 1 for a, b in zip(candidate_rr, baseline_rr)),
    }


def evaluate(args):
    labels = fold_labels(args.split, "dev", args.panel)
    threshold = json.loads(args.thresholds.read_text())
    minimum_cosine = threshold["minimum_cosine"]
    minimum_margin = threshold["minimum_margin"]
    if threshold["catalogs"] != load_manifest(args.split)["inputs"]["catalogs"]:
        raise ValueError("Threshold artifact was calibrated against a different catalog snapshot")
    if threshold["assignments"] != file_record(args.split / "assignments.parquet"):
        raise ValueError("Threshold artifact was calibrated on a different split")
    ids = {row["molecule_id"] for row in labels}
    orders = defaultdict(dict)
    union_pool, catalog_pool = {}, {}
    for molecule_id, evidence in read_evidence(args.candidates, ids):
        if molecule_id in union_pool:
            raise ValueError("Candidate export rows must be grouped by molecule")
        library = evidence.library_keys
        catalog = evidence.catalog
        union_pool[molecule_id] = set(library).union(catalog)
        catalog_pool[molecule_id] = set(catalog)
        orders["catalog_mass_only"][molecule_id] = catalog[:TOP_K]
        orders["catalog_first"][molecule_id] = merge_candidates(catalog, library, TOP_K)
        orders["library_first"][molecule_id] = merge_candidates(library, catalog, TOP_K)
        orders["confidence_gated"][molecule_id] = evidence.gated_order(minimum_cosine, minimum_margin)
    if not union_pool:
        raise ValueError("No candidates matched the selected development panel")
    groups = {
        "class_1": [row for row in labels if row["novelty"] == "class_1"],
        "class_2": [row for row in labels if row["novelty"] == "class_2"],
        "timsTOF_np": [row for row in labels if row["domain"] == "timsTOF_np"],
        "focused_class_2": [
            row for row in labels if row.get("focused_panel") and row["novelty"] == "class_2"
        ],
    }
    return {
        "fold": "dev",
        "panel": args.panel,
        "thresholds": threshold,
        "scores": {
            name: score_groups(labels, order, catalog_pool if name == "catalog_mass_only" else union_pool)
            for name, order in orders.items()
        },
        "paired_confidence_gated_deltas": {
            name: {
                baseline: paired_changes(rows, orders["confidence_gated"], orders[baseline])
                for baseline in ("library_first", "catalog_first")
            }
            for name, rows in groups.items() if rows
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    calibration = commands.add_parser("calibrate", help="Choose thresholds on the frozen fit partition")
    calibration.add_argument("--catalog", type=Path, action="append", required=True)
    calibration.add_argument("--key-cache", type=Path)
    calibration.add_argument("--limit-per-source", type=int, default=160)
    calibration.add_argument("--seed", type=int, default=2026)
    calibration.set_defaults(run=calibrate)
    scoring = commands.add_parser("evaluate", help="Compare rankings on an existing development candidate export")
    scoring.add_argument("--candidates", type=Path, required=True)
    scoring.add_argument("--thresholds", type=Path, required=True)
    scoring.add_argument("--panel", choices=("all", "focused"), default="all")
    scoring.set_defaults(run=evaluate)
    for command in (calibration, scoring):
        command.add_argument("--split", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(args.run(args), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
