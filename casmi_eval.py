"""Build, prepare, and evaluate frozen, molecule-disjoint CASMI 2026 splits."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
from tempfile import TemporaryDirectory
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from casmi_baseline import (
    LABEL_COLUMNS, QUERY_COLUMNS, TOP_K, Chem, Molecule,
    clean_peaks, connectivity, cosine, search_library, spectral_signature,
)
from casmi_catalog import (
    ADDUCT_SHIFT, META_RDKIT_KEY, META_SORTED_MASS_KEY,
    catalog_rows, export_candidates, is_prepared_catalog, load_formulas,
    mass_order, merge_candidates, neutral_mass, search_catalogs,
)


ANCHOR_SOURCES = ("enveda-np-examples", "riken")
AGGREGATOR_SOURCES = frozenset({
    "riken", "mona", "massbank", "msdial", "gnps", "pluskal_ms2", "drug_plus", "masaryk",
})
NEAR_DUP_COSINE = 0.999
HOLDOUT_FRACTION = 0.05
METRIC_RDKIT = "2026.03.3"
KEY_PATTERN = re.compile(r"[A-Z]{14}")
CLASSES = ("class_1", "class_2", "class_3_proxy")
NO_CATALOG = "catalog_unchecked"
SPECTRUM_COUNTS = (1, 2, 3, 4, 6, 10, 16)
SPECTRUM_WEIGHTS = (17, 19, 27, 19, 10, 6, 2)
MASS_RANGE = (157, 1159)
MASS_BANDS = (250, 400, 600)
TANIMOTO_BINS = (
    ("<0.50", 0.50),
    ("0.50-0.70", 0.70),
    ("0.70-0.85", 0.85),
    (">=0.85", math.inf),
)
CLASS_MIX_SCENARIOS = {
    "equal_33_33_33": (1 / 3, 1 / 3, 1 / 3),
    "lib_heavy_50_35_15": (0.50, 0.35, 0.15),
    "catalog_heavy_25_55_20": (0.25, 0.55, 0.20),
    "novel_heavy_20_40_40": (0.20, 0.40, 0.40),
}
META_SOURCE_KEY = b"casmi_source_file"
FP_BITS = 2048
FP_BYTES = FP_BITS // 8


def check_official_rdkit():
    if Chem:
        from rdkit import rdBase
        version = rdBase.rdkitVersion
    else:
        version = None
    if version != METRIC_RDKIT:
        raise ValueError(
            f"Official identity requires RDKit {METRIC_RDKIT}; found {version}. "
            "Use --trust-keys only when all input tables have precomputed connectivity keys."
        )


def identity_for(mode, key_map=None):
    if mode == "provided":
        return lambda _smiles, key: key if key and KEY_PATTERN.fullmatch(key) else None
    check_official_rdkit()
    if key_map is not None:
        return lambda smiles, provided: key_map[smiles] if smiles in key_map else connectivity(smiles, provided)
    return connectivity


def stable_hash(*parts):
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()


def file_record(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"name": path.name, "size": path.stat().st_size, "sha256": digest}


def json_file(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def normalized_instrument(value):
    return "timsTOF" if value and "timstof" in value.casefold() else value or "unknown"


def domain_for(source, instrument):
    if instrument == "timsTOF":
        return "timsTOF_np" if source == "enveda-np-examples" else "timsTOF_synthetic"
    return "public_other"


def tanimoto_bin_for(value):
    if value is None or not math.isfinite(value):
        return "unknown"
    for label, upper in TANIMOTO_BINS:
        if value < upper:
            return label
    return ">=0.85"


def suitable(row, *, query):
    mz = row["precursor_mz"]
    if not mz or not math.isfinite(mz) or mz <= 0 or not row["adduct"]:
        return None
    if query:
        shift = ADDUCT_SHIFT.get(row["adduct"])
        if shift is None or not MASS_RANGE[0] <= mz - shift <= MASS_RANGE[1]:
            return None
    peaks = clean_peaks(row["ms2_mzs"], row["ms2_normalized_intensities"])
    return spectral_signature(peaks) if len(peaks[0]) else None


def data_batches(path, columns, sources=None, *, outside=False):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=columns, batch_size=32768):
        if sources is not None:
            column = pc.cast(batch["ingest_lib"], pa.string())
            included = pc.is_in(column, value_set=pa.array(sorted(sources), type=pa.string()))
            batch = batch.filter(pc.invert(included) if outside else included)
        if batch.num_rows:
            yield batch


# --- Parallel RDKit 2026.03.3 precomputation workers ---

def _canonicalize_train_chunk(smiles_list):
    from rdkit import Chem as _Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    enumerator = rdMolStandardize.TautomerEnumerator()
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    out = []
    for sm in smiles_list:
        mol = _Chem.MolFromSmiles(sm)
        if mol is None:
            out.append((sm, None, None, None))
            continue
        canon = enumerator.Canonicalize(mol)
        key = _Chem.MolToInchiKey(canon).split("-")[0] or None
        mass = float(rdMolDescriptors.CalcExactMolWt(canon))
        fp = np.packbits(fp_gen.GetFingerprintAsNumPy(canon)).tobytes()
        out.append((sm, key, mass, fp))
    return out


def _canonicalize_catalog_chunk(task):
    smiles_list, max_mass = task
    from rdkit import Chem as _Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    enumerator = rdMolStandardize.TautomerEnumerator()
    lfc = rdMolStandardize.LargestFragmentChooser()
    uncharger = rdMolStandardize.Uncharger()
    out = []
    for sm in smiles_list:
        mol = _Chem.MolFromSmiles(sm)
        if mol is None:
            continue
        if "." in sm:
            mol = lfc.choose(mol)
            if mol is None:
                continue
            mol = uncharger.uncharge(mol)
            if mol is None:
                continue
        if mol.GetNumHeavyAtoms() == 0:
            continue
        mass = float(rdMolDescriptors.CalcExactMolWt(mol))
        if not math.isfinite(mass) or mass <= 0:
            continue
        if max_mass is not None and mass > max_mass:
            continue
        canon = enumerator.Canonicalize(mol)
        key = _Chem.MolToInchiKey(canon).split("-")[0] or None
        if not key:
            continue
        mass = float(rdMolDescriptors.CalcExactMolWt(canon))
        if not math.isfinite(mass) or mass <= 0:
            continue
        formula = rdMolDescriptors.CalcMolFormula(canon)
        canon_sm = _Chem.MolToSmiles(canon, canonical=True)
        out.append((key, mass, canon_sm, formula))
    return out


def _chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def prepare_train(args):
    check_official_rdkit()
    if args.output.exists() and not args.overwrite:
        raise ValueError(f"{args.output} exists; pass --overwrite to replace it")
    parquet = pq.ParquetFile(args.train)
    unique_smiles = set()
    for batch in parquet.iter_batches(batch_size=131072, columns=["normalized_smiles"]):
        for sm in batch["normalized_smiles"].to_pylist():
            if sm:
                unique_smiles.add(sm)
    smiles_list = sorted(unique_smiles)
    workers = args.workers or max(1, os.cpu_count() or 1)
    chunk_size = max(256, min(2000, math.ceil(len(smiles_list) / (workers * 8))))
    rows_sm, rows_key, rows_mass, rows_fp = [], [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for chunk_res in pool.map(_canonicalize_train_chunk, _chunked(smiles_list, chunk_size)):
            for sm, key, mass, fp in chunk_res:
                rows_sm.append(sm)
                rows_key.append(key)
                rows_mass.append(mass)
                rows_fp.append(fp)
    table = pa.table({
        "normalized_smiles": pa.array(rows_sm, type=pa.string()),
        "canonical_inchikey14": pa.array(rows_key, type=pa.string()),
        "exact_mass": pa.array(rows_mass, type=pa.float64()),
        "morgan_fp": pa.array(rows_fp, type=pa.binary(FP_BYTES)),
    })
    meta = {
        META_RDKIT_KEY: METRIC_RDKIT.encode(),
        META_SOURCE_KEY: json.dumps(file_record(args.train)).encode(),
    }
    table = table.replace_schema_metadata(meta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output, compression="zstd")
    valid_keys = {k for k in rows_key if k}
    summary = {
        "train": args.train.name,
        "output": str(args.output),
        "rdkit_version": METRIC_RDKIT,
        "unique_smiles": len(smiles_list),
        "unique_canonical_keys": len(valid_keys),
        "invalid_smiles": sum(k is None for k in rows_key),
    }
    print(json.dumps(summary, indent=2))


def load_key_cache(path, train_path=None):
    pf = pq.ParquetFile(path)
    meta = pf.schema_arrow.metadata or {}
    version = (meta.get(META_RDKIT_KEY) or b"").decode()
    if version != METRIC_RDKIT:
        raise ValueError(f"{path}: key cache RDKit version {version!r} != {METRIC_RDKIT!r}")
    if train_path is not None and META_SOURCE_KEY in meta:
        recorded = json.loads(meta[META_SOURCE_KEY].decode())
        if recorded != file_record(train_path):
            raise ValueError(f"{path}: key cache was built for a different train.parquet")
    tbl = pf.read()
    key_map = {}
    fp_by_key = {}
    for sm, key, fp in zip(
        tbl["normalized_smiles"].to_pylist(),
        tbl["canonical_inchikey14"].to_pylist(),
        tbl["morgan_fp"].to_pylist(),
    ):
        key_map[sm] = key
        if key and fp is not None and key not in fp_by_key:
            fp_by_key[key] = fp
    return key_map, fp_by_key


def _iter_raw_catalog(path, smiles_col=None):
    def pick_col(fields, candidates):
        for c in candidates:
            if c in fields:
                return c
        return None

    if path.suffix == ".parquet":
        pf = pq.ParquetFile(path)
        fields = set(pf.schema_arrow.names)
        scol = smiles_col or pick_col(fields, ("smiles", "canonical_smiles", "normalized_smiles"))
        if not scol:
            raise ValueError(f"{path}: could not find SMILES column in {sorted(fields)}")
        kcol = pick_col(fields, ("standard_inchi_key", "inchikey", "inchikey14"))
        cols = [scol] + ([kcol] if kcol else [])
        for batch in pf.iter_batches(columns=cols, batch_size=65536):
            sm_list = batch[scol].to_pylist()
            ik_list = batch[kcol].to_pylist() if kcol else [None] * len(sm_list)
            yield from zip(sm_list, ik_list)
        return

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]
            if not csv_names:
                raise ValueError(f"{path}: zip archive contains no .csv file")
            with zf.open(csv_names[0]) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
                fields = set(reader.fieldnames or [])
                scol = smiles_col or pick_col(fields, ("smiles", "canonical_smiles", "normalized_smiles"))
                if not scol:
                    raise ValueError(f"{path}: could not find SMILES column in {sorted(fields)}")
                kcol = pick_col(fields, ("standard_inchi_key", "inchikey", "inchikey14"))
                for row in reader:
                    yield row[scol], row.get(kcol) if kcol else None
        return

    if path.suffix != ".csv" and not path.name.endswith(".csv.gz"):
        raise ValueError(f"{path}: expected .parquet, .csv, .csv.gz, or .zip")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        scol = smiles_col or pick_col(fields, ("smiles", "canonical_smiles", "normalized_smiles"))
        if not scol:
            raise ValueError(f"{path}: could not find SMILES column in {sorted(fields)}")
        kcol = pick_col(fields, ("standard_inchi_key", "inchikey", "inchikey14"))
        for row in reader:
            yield row[scol], row.get(kcol) if kcol else None


def prepare_catalog(args):
    check_official_rdkit()
    if args.output.exists() and not args.overwrite:
        raise ValueError(f"{args.output} exists; pass --overwrite to replace it")
    # Deduplicate stereoisomers with identical 2D standard InChIKey[:14] before tautomer canonicalization;
    # keep all multi-fragment (salt) records separately so LargestFragmentChooser runs on each.
    by_prekey = {}
    salt_smiles = set()
    no_key_smiles = set()
    raw_rows = 0
    for sm, raw_ik in _iter_raw_catalog(args.input, args.smiles_col):
        raw_rows += 1
        if not sm:
            continue
        if "." in sm:
            salt_smiles.add(sm)
        elif raw_ik and len(raw_ik) >= 14:
            pre = raw_ik[:14]
            if pre not in by_prekey:
                by_prekey[pre] = sm
        else:
            no_key_smiles.add(sm)

    unique_inputs = sorted(set(by_prekey.values()) | salt_smiles | no_key_smiles)
    random.Random(0).shuffle(unique_inputs)
    workers = args.workers or max(1, os.cpu_count() or 1)
    max_mass = getattr(args, "max_mass", None)
    chunk_size = max(128, min(1000, math.ceil(len(unique_inputs) / (workers * 16))))
    chunks = list(_chunked(unique_inputs, chunk_size))
    by_key = {}
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for chunk_res in pool.map(_canonicalize_catalog_chunk, ((c, max_mass) for c in chunks)):
            done += 1
            for key, mass, canon_sm, formula in chunk_res:
                prev = by_key.get(key)
                if prev is None or (len(canon_sm), canon_sm) < (len(prev[1]), prev[1]):
                    by_key[key] = (mass, canon_sm, formula)
            if len(chunks) >= 20 and (done % max(1, len(chunks) // 10) == 0 or done == len(chunks)):
                print(
                    f"[prepare-catalog] {done}/{len(chunks)} chunks ({100 * done / len(chunks):.0f}%), "
                    f"unique_keys={len(by_key):,}",
                    flush=True,
                )

    records = sorted(
        ((mass, key, sm, formula) for key, (mass, sm, formula) in by_key.items()),
        key=lambda x: (x[0], x[1]),
    )
    table = pa.table({
        "smiles": pa.array([r[2] for r in records], type=pa.string()),
        "inchikey14": pa.array([r[1] for r in records], type=pa.string()),
        "exact_mass": pa.array([r[0] for r in records], type=pa.float64()),
        "formula": pa.array([r[3] for r in records], type=pa.string()),
    })
    meta = {
        META_RDKIT_KEY: METRIC_RDKIT.encode(),
        META_SORTED_MASS_KEY: b"true",
        META_SOURCE_KEY: json.dumps(file_record(args.input)).encode(),
    }
    table = table.replace_schema_metadata(meta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output, compression="zstd")
    summary = {
        "input": args.input.name,
        "output": str(args.output),
        "rdkit_version": METRIC_RDKIT,
        "raw_rows": raw_rows,
        "canonicalized_inputs": len(unique_inputs),
        "unique_canonical_structures": len(records),
    }
    print(json.dumps(summary, indent=2))


def catalog_membership(paths, identity, query_keys):
    keys = set()
    for path in paths:
        if is_prepared_catalog(path):
            tbl = pq.read_table(path, columns=["inchikey14"])
            col = tbl["inchikey14"].to_pylist()
            if not col:
                raise ValueError(f"{path}: catalog has no valid connectivity keys")
            keys.update(k for k in col if k in query_keys)
            continue
        valid = 0
        for row in catalog_rows(path):
            key = identity(row["smiles"], row.get("inchikey14")) if row["smiles"] else None
            if key:
                valid += 1
                if key in query_keys:
                    keys.add(key)
        if not valid:
            raise ValueError(f"{path}: catalog has no valid connectivity keys")
    return keys


# --- Split construction & near-duplicate auditing ---

def batch_keys(batch, identity):
    return [
        identity(smiles, key) if smiles else None
        for smiles, key in zip(batch["normalized_smiles"].to_pylist(), batch["inchikey14"].to_pylist())
    ]


def spectrum_groups(train, sources, identity):
    feature_fields = [
        name for name in pq.ParquetFile(train).schema_arrow.names
        if name in {*QUERY_COLUMNS, "instrument_type", "ionization_mode",
                    "base_peak_intensity", "collision_energy_ev", "collision_energy_orig",
                    "collision_energy_orig_units"}
    ]
    queries = defaultdict(lambda: defaultdict(dict))
    signatures, keys, seen_sources = set(), set(), set()
    for batch in data_batches(train, feature_fields + LABEL_COLUMNS, sources):
        ids = batch_keys(batch, identity)
        for key, row in zip(ids, batch.to_pylist()):
            seen_sources.add(row["ingest_lib"])
            if not key:
                continue
            keys.add(key)
            signature = suitable(row, query=True)
            if signature is None or not row["ingest_lib"]:
                continue
            instrument = normalized_instrument(row.get("instrument_type"))
            neutral_mass = row["precursor_mz"] - ADDUCT_SHIFT[row["adduct"]]
            queries[key][row["ingest_lib"], instrument].setdefault(signature, neutral_mass)
            if sources is not None:
                signatures.add(signature)
    missing = set(sources or ()) - seen_sources
    if missing:
        raise ValueError(f"Query sources missing in training data: {', '.join(sorted(missing))}")
    return queries, signatures, keys, feature_fields


def collect_aggregator_query_peaks(train, cases, identity):
    """Collect cleaned query peaks per (key, adduct) for selected queries inside AGGREGATOR_SOURCES."""
    needed = {
        key: case["source"]
        for key, case in cases.items()
        if case["source"] in AGGREGATOR_SOURCES
    }
    if not needed:
        return {}
    peaks_by_key_adduct = defaultdict(list)
    parquet = pq.ParquetFile(train)
    for batch in parquet.iter_batches(batch_size=32768, columns=QUERY_COLUMNS + LABEL_COLUMNS):
        ids = batch_keys(batch, identity)
        libs = batch["ingest_lib"].to_pylist()
        indices = [i for i, (k, lib) in enumerate(zip(ids, libs)) if k in needed and lib == needed[k]]
        if not indices:
            continue
        for i, row in zip(indices, batch.take(indices).to_pylist()):
            peaks = clean_peaks(row["ms2_mzs"], row["ms2_normalized_intensities"])
            if len(peaks[0]) and row["adduct"]:
                peaks_by_key_adduct[ids[i], row["adduct"]].append(peaks)
    return peaks_by_key_adduct


def reference_membership(
    train, sources, identity, query_keys, signatures, all_keys,
    excluded_key_sources, agg_query_peaks=None,
):
    reference_meta = defaultdict(set)
    excluded_row_ids = set()
    near_dupes_excluded = 0
    parquet = pq.ParquetFile(train)
    source_set = set(sources) if sources is not None else None
    row_offset = 0
    for batch in parquet.iter_batches(
        batch_size=32768, columns=QUERY_COLUMNS + LABEL_COLUMNS + ["instrument_type"],
    ):
        batch_len = batch.num_rows
        ids = batch_keys(batch, identity)
        all_keys.update(key for key in ids if key)
        libs = batch["ingest_lib"].to_pylist()
        if source_set is not None:
            indices = [
                i for i, (key, lib) in enumerate(zip(ids, libs))
                if key in query_keys and lib not in source_set
            ]
        else:
            indices = [i for i, key in enumerate(ids) if key in query_keys]
        if indices:
            for index, row in zip(indices, batch.take(indices).to_pylist()):
                key = ids[index]
                ref_lib = row["ingest_lib"]
                global_row = row_offset + index
                if (key, ref_lib) in excluded_key_sources:
                    excluded_row_ids.add(global_row)
                    continue
                mz = row["precursor_mz"]
                if not mz or not math.isfinite(mz) or mz <= 0 or not row["adduct"]:
                    continue
                peaks = clean_peaks(row["ms2_mzs"], row["ms2_normalized_intensities"])
                if not len(peaks[0]):
                    continue
                signature = spectral_signature(peaks)
                if signature in signatures:
                    excluded_row_ids.add(global_row)
                    continue
                if agg_query_peaks and ref_lib in AGGREGATOR_SOURCES:
                    q_list = agg_query_peaks.get((key, row["adduct"]))
                    if q_list and any(cosine(qp, peaks) >= NEAR_DUP_COSINE for qp in q_list):
                        excluded_row_ids.add(global_row)
                        near_dupes_excluded += 1
                        continue
                reference_meta[key].add((
                    ref_lib, row["adduct"], normalized_instrument(row["instrument_type"]),
                ))
        row_offset += batch_len
    return reference_meta, excluded_row_ids, near_dupes_excluded


def source_priority(sources):
    ranks = {source: i for i, source in enumerate(sources)}
    return lambda source: ranks.get(source, len(ranks))


def query_case(key, groups, priority, seed):
    source, instrument = min(
        groups, key=lambda group: (group[1] != "timsTOF", priority(group[0]), group),
    )
    return {
        "molecule_id": "m_" + stable_hash("query", seed, key)[:20],
        "inchikey14": key,
        "source": source,
        "instrument": instrument,
        "domain": domain_for(source, instrument),
        "neutral_mass": float(np.median(list(groups[source, instrument].values()))),
    }


def annotate_novelty(case, references, catalog, catalog_supplied):
    key = case["inchikey14"]
    refs = references.get(key)
    if refs:
        novelty = "class_1"
        if case["instrument"] == "timsTOF":
            tier = "timsTOF_vs_public"
        elif case["source"] not in AGGREGATOR_SOURCES or any(lib not in AGGREGATOR_SOURCES for lib, _, _ in refs):
            tier = "cross_group"
        else:
            tier = "public_deduped"
    elif key in catalog:
        novelty = "class_2"
        tier = None
    else:
        novelty = "class_3_proxy" if catalog_supplied else NO_CATALOG
        tier = None
    return {
        **case,
        "novelty": novelty,
        "class_1_tier": tier,
        "in_catalog": key in catalog if catalog_supplied else None,
    }


def case_stratum(case):
    return (
        case.get("novelty"),
        case["domain"],
        case["source"],
        case["instrument"] == "timsTOF",
        int(np.searchsorted(MASS_BANDS, case["neutral_mass"])),
    )


def select_queries(queries, references, catalog, catalog_supplied, sources, limit, seed):
    priority = source_priority(sources)
    by_class = defaultdict(list)
    for key, groups in queries.items():
        case = annotate_novelty(
            query_case(key, groups, priority, seed), references, catalog, catalog_supplied,
        )
        by_class[case["novelty"]].append(case)

    selected, eligible_counts = {}, {}
    for novelty, cases in sorted(by_class.items()):
        eligible_counts[novelty] = len(cases)
        cases.sort(key=lambda c: (
            c["domain"] != "timsTOF_np", c["instrument"] != "timsTOF",
            priority(c["source"]), stable_hash(seed, c["inchikey14"]),
        ))
        counts = Counter()
        for case in cases[:limit]:
            stratum = case_stratum(case)
            fold = min(("dev", "test"), key=lambda f: (counts[stratum, f], counts[f], f))
            counts[stratum, fold] += 1
            counts[fold] += 1
            selected[case["inchikey14"]] = {**case, "fold": fold}
    if {case["fold"] for case in selected.values()} != {"dev", "test"}:
        raise ValueError("Need enough eligible distinct structures for both dev and local test")
    return selected, eligible_counts


def select_broad_candidates(queries, total_structures, sources, seed):
    """Select the 2x5% holdout pool before reference/catalog annotation; folds are assigned after."""
    requested = max(1, round(HOLDOUT_FRACTION * total_structures))
    target = min(requested, len(queries) // 2)
    if target == 0:
        raise ValueError("Need enough eligible distinct structures for both dev and local test")

    priority = source_priority(sources)
    anchor = sources[0]
    selected = sorted(
        queries, key=lambda key: (
            (anchor, "timsTOF") not in queries[key], stable_hash("broad-split", seed, key),
        ),
    )[:2 * target]
    cases = {key: query_case(key, queries[key], priority, seed) for key in selected}
    return cases, target, requested


def assign_balanced_folds(cases, target, seed):
    """Assign dev vs test stratified by (novelty, domain, source, instrument, mass_band)."""
    ordered = sorted(
        cases.values(),
        key=lambda c: (
            c["novelty"], c["domain"], c["source"],
            int(np.searchsorted(MASS_BANDS, c["neutral_mass"])),
            stable_hash("fold-assign", seed, c["inchikey14"]),
        ),
    )
    counts = Counter()
    assigned = {}
    for case in ordered:
        stratum = case_stratum(case)
        options = [fold for fold in ("dev", "test") if counts[fold] < target]
        fold = min(options, key=lambda f: (counts[stratum, f], counts[case["novelty"], f], counts[f], f))
        counts[stratum, fold] += 1
        counts[case["novelty"], fold] += 1
        counts[fold] += 1
        assigned[case["inchikey14"]] = {**case, "fold": fold}
    return assigned


def tag_focused_panel(cases, sources, limit, seed):
    priority = source_priority(sources)
    by_stratum = defaultdict(list)
    for case in cases.values():
        by_stratum[case["fold"], case["novelty"]].append(case)
    for group in by_stratum.values():
        group.sort(key=lambda c: (
            c["domain"] != "timsTOF_np",
            c["instrument"] != "timsTOF",
            priority(c["source"]),
            stable_hash("focused-panel", seed, c["inchikey14"]),
        ))
        for i, case in enumerate(group):
            case["focused_panel"] = i < limit


def compute_max_tanimoto(query_keys, train_keys, fp_by_key):
    """Exact max Morgan-2048 Tanimoto of each query key against all train keys via BLAS matmul."""
    if not fp_by_key:
        return {}
    q_keys = [k for k in query_keys if k in fp_by_key]
    t_keys = [k for k in train_keys if k in fp_by_key]
    if not q_keys or not t_keys:
        return {}

    def unpack_matrix(keys):
        packed = np.frombuffer(b"".join(fp_by_key[k] for k in keys), dtype=np.uint8).reshape(len(keys), FP_BYTES)
        bits = np.unpackbits(packed, axis=1).astype(np.float32)
        return bits, bits.sum(axis=1)

    q_mat, q_cnt = unpack_matrix(q_keys)
    max_sim = np.zeros(len(q_keys), dtype=np.float32)
    t_chunk = 32768
    q_chunk = 2048
    for t0 in range(0, len(t_keys), t_chunk):
        t_mat, t_cnt = unpack_matrix(t_keys[t0:t0 + t_chunk])
        t_mat_T = np.ascontiguousarray(t_mat.T)
        for q0 in range(0, len(q_keys), q_chunk):
            q1 = min(q0 + q_chunk, len(q_keys))
            inter = q_mat[q0:q1] @ t_mat_T
            denom = q_cnt[q0:q1, None] + t_cnt[None, :] - inter
            with np.errstate(divide="ignore", invalid="ignore"):
                sim = np.where(denom > 0, inter / denom, 0.0)
            np.maximum(max_sim[q0:q1], sim.max(axis=1), out=max_sim[q0:q1])
    return {k: float(s) for k, s in zip(q_keys, max_sim)}


def materialize_query_rows(train, fields, sources, identity, cases, references, tanimoto_map, seed):
    grouped = defaultdict(dict)
    for batch in data_batches(train, fields + LABEL_COLUMNS, sources):
        keys = batch_keys(batch, identity)
        indices = [
            i for i, (key, source, instrument) in enumerate(zip(
                keys, batch["ingest_lib"].to_pylist(), batch["instrument_type"].to_pylist(),
            ))
            if key in cases and source == cases[key]["source"]
            and normalized_instrument(instrument) == cases[key]["instrument"]
        ]
        if not indices:
            continue
        for i, row in zip(indices, batch.take(indices).to_pylist()):
            signature = suitable(row, query=True)
            if signature is not None:
                grouped[keys[i]].setdefault(signature, {field: row[field] for field in fields})

    features, labels = defaultdict(list), defaultdict(list)
    for key, case in sorted(cases.items()):
        rng = random.Random(stable_hash(seed, key, "spectra"))
        target, = rng.choices(SPECTRUM_COUNTS, weights=SPECTRUM_WEIGHTS)
        rows = [row for _, row in sorted(grouped[key].items())]
        rng.shuffle(rows)
        # Prefer distinct adduct and collision-energy combinations where available.
        conditions, extra = {}, []
        for row in rows:
            energy = row.get("collision_energy_ev") or row.get("collision_energy_orig")
            condition = (row["adduct"], str(energy))
            if condition in conditions:
                extra.append(row)
            else:
                conditions[condition] = row
        sampled = (list(conditions.values()) + extra)[:target]
        fold = case["fold"]
        for i, row in enumerate(sampled, 1):
            molecule_id = case["molecule_id"]
            features[fold].append({"molecule_id": molecule_id, "spectrum_id": f"{molecule_id}_{i}", **row})
        refs = references.get(key, ())
        matching_adducts = {adduct for _, adduct, _ in refs}
        max_tan = tanimoto_map.get(key)
        labels[fold].append({
            **case,
            "spectra": len(sampled),
            "same_adduct_reference": bool(matching_adducts & {row["adduct"] for row in sampled}),
            "same_instrument_reference": any(
                instrument == case["instrument"] for _, _, instrument in refs
            ),
            "max_tanimoto_to_train": max_tan,
            "tanimoto_bin": tanimoto_bin_for(max_tan),
        })
    return features, labels


def _fold_summary(rows, feature_rows):
    return {
        "molecules": len(rows),
        "spectra": len(feature_rows),
        "by_class": dict(Counter(row["novelty"] for row in rows)),
        "by_domain": dict(Counter(row["domain"] for row in rows)),
        "by_source": dict(Counter(row["source"] for row in rows)),
        "by_instrument": dict(Counter(row["instrument"] for row in rows)),
        "by_tanimoto_bin": dict(Counter(row["tanimoto_bin"] for row in rows)),
        "class_1_by_tier": dict(Counter(row["class_1_tier"] for row in rows if row["class_1_tier"])),
        "focused_panel_by_class": dict(Counter(
            row["novelty"] for row in rows if row.get("focused_panel", False)
        )),
        "focused_panel_by_domain": dict(Counter(
            row["domain"] for row in rows if row.get("focused_panel", False)
        )),
        "spectra_per_molecule": dict(Counter(str(row["spectra"]) for row in rows)),
        "median_neutral_mass": float(np.median([row["neutral_mass"] for row in rows])),
    }


def build_split(args):
    mode = "provided" if args.trust_keys else "official"
    key_map, fp_by_key = (None, {})
    if getattr(args, "key_cache", None):
        if mode != "official":
            raise ValueError("--key-cache requires official identity mode (do not pass --trust-keys)")
        key_map, fp_by_key = load_key_cache(args.key_cache, args.train)
    identity = identity_for(mode, key_map)
    train, catalogs = args.train, args.catalog
    out = args.output
    if out.exists():
        raise ValueError(f"{out} exists; choose a new output directory to preserve frozen splits")
    sources = tuple(dict.fromkeys(args.query_source or ANCHOR_SOURCES))
    split_mode = args.split_mode
    scan_sources = sources if split_mode == "focused" else None
    input_files = {"train": file_record(train), "catalogs": [file_record(p) for p in catalogs]}
    queries, signatures, all_keys, fields = spectrum_groups(train, scan_sources, identity)
    eligible_structures = len(queries)
    excluded_key_sources = set()
    agg_query_peaks = None
    target_per_holdout = None
    broad_target = None
    if split_mode == "broad":
        cases, broad_target, target_per_holdout = select_broad_candidates(
            queries, len(all_keys), sources, args.seed,
        )
        queries = {key: queries[key] for key in cases}
        excluded_key_sources = {(key, case["source"]) for key, case in cases.items()}
        signatures = {
            signature
            for key, case in cases.items()
            for (source, _), group in queries[key].items() if source == case["source"]
            for signature in group
        }
        agg_query_peaks = collect_aggregator_query_peaks(train, cases, identity)

    references, excluded_row_ids, near_dupes_excluded = reference_membership(
        train, scan_sources, identity, queries, signatures, all_keys,
        excluded_key_sources, agg_query_peaks,
    )
    present = catalog_membership(catalogs, identity, queries)
    if split_mode == "broad":
        cases = {
            key: annotate_novelty(case, references, present, bool(catalogs))
            for key, case in cases.items()
        }
        cases = assign_balanced_folds(cases, broad_target, args.seed)
        tag_focused_panel(cases, sources, max(1, args.max_per_class // 2), args.seed)
        counts = None
    else:
        cases, counts = select_queries(
            queries, references, present, bool(catalogs), sources, args.max_per_class, args.seed,
        )

    train_keys = all_keys - cases.keys()
    tanimoto_map = compute_max_tanimoto(sorted(cases), sorted(train_keys), fp_by_key) if fp_by_key else {}
    features, labels = materialize_query_rows(
        train, fields, scan_sources, identity, cases, references, tanimoto_map, args.seed,
    )
    fit_excluded = {
        signature for key in cases for group in queries[key].values() for signature in group
    }
    manifest = {
        "version": 3,
        "identity": mode,
        "seed": args.seed,
        "inputs": input_files,
        "split_mode": split_mode,
        "query_sources": list(sources),
        "aggregator_sources": sorted(AGGREGATOR_SOURCES),
        "near_dup_cosine_threshold": NEAR_DUP_COSINE,
        "aggregator_near_dupes_excluded": near_dupes_excluded,
        "excluded_reference_rows": len(excluded_row_ids),
        "eligible_by_class": counts,
        "eligible_structures": eligible_structures,
        "target_per_holdout": target_per_holdout,
        "max_per_class": args.max_per_class,
        "query_neutral_mass_range": list(MASS_RANGE),
        "spectrum_count_sampling": dict(zip(SPECTRUM_COUNTS, SPECTRUM_WEIGHTS)),
        "fit_structures": len(train_keys),
        "has_tanimoto": bool(tanimoto_map),
        "splits": {fold: _fold_summary(rows, features[fold]) for fold, rows in labels.items()},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent, prefix=".casmi-split-") as tmp:
        stage = Path(tmp) / "split"
        stage.mkdir()
        for fold in ("dev", "test"):
            pq.write_table(pa.Table.from_pylist(features[fold]), stage / f"{fold}_queries.parquet")
            pq.write_table(pa.Table.from_pylist(labels[fold]), stage / f"{fold}_labels.parquet")
        keys = sorted(all_keys)
        pq.write_table(pa.table({
            "inchikey14": keys,
            "fold": [cases[key]["fold"] if key in cases else "train" for key in keys],
            "query_source": pa.array(
                [cases[key]["source"] if key in cases else None for key in keys], type=pa.string(),
            ),
        }), stage / "assignments.parquet")
        for name, items in (("excluded_reference_spectra", signatures), ("excluded_fit_spectra", fit_excluded)):
            pq.write_table(pa.table({"signature": pa.array(sorted(items), type=pa.binary(16))}),
                           stage / f"{name}.parquet")
        pq.write_table(
            pa.table({"row_id": pa.array(sorted(excluded_row_ids), type=pa.int64())}),
            stage / "excluded_reference_row_ids.parquet",
        )
        json_file(stage / "manifest.json", manifest)
        stage.rename(out)
    print(json.dumps(manifest, indent=2))


def attach_catalog(args):
    """Refresh catalog coverage labels on an already-frozen split without changing fold assignments."""
    split = args.split
    manifest = load_manifest(split)
    if manifest["version"] < 2:
        raise ValueError("attach-catalog requires manifest version >= 2")
    identity = identity_for(manifest["identity"])
    all_labels = {}
    query_keys = set()
    for fold in ("dev", "test"):
        rows = pq.read_table(split / f"{fold}_labels.parquet").to_pylist()
        all_labels[fold] = rows
        query_keys.update(r["inchikey14"] for r in rows)
    present = catalog_membership(args.catalog, identity, query_keys)
    cases = {}
    for fold, rows in all_labels.items():
        for row in rows:
            in_cat = row["inchikey14"] in present
            row["in_catalog"] = in_cat
            if row["novelty"] != "class_1":
                row["novelty"] = "class_2" if in_cat else "class_3_proxy"
            cases[row["inchikey14"]] = row
    if manifest.get("split_mode") == "broad":
        tag_focused_panel(cases, tuple(manifest["query_sources"]), max(1, manifest["max_per_class"] // 2), manifest["seed"])
    for fold, rows in all_labels.items():
        pq.write_table(pa.Table.from_pylist(rows), split / f"{fold}_labels.parquet")
        feature_count = pq.ParquetFile(split / f"{fold}_queries.parquet").metadata.num_rows
        manifest["splits"][fold] = {
            **manifest["splits"][fold],
            "spectra": feature_count,
            "by_class": dict(Counter(r["novelty"] for r in rows)),
            "focused_panel_by_class": dict(Counter(r["novelty"] for r in rows if r.get("focused_panel", False))),
            "focused_panel_by_domain": dict(Counter(r["domain"] for r in rows if r.get("focused_panel", False))),
        }
    manifest["inputs"]["catalogs"] = [file_record(p) for p in args.catalog]
    json_file(split / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def load_manifest(split):
    manifest = json.loads((split / "manifest.json").read_text())
    if manifest["version"] not in (1, 2, 3):
        raise ValueError("Unsupported split manifest version")
    return manifest


# --- Metrics & reporting ---

def metrics(rows, rank_lists, coverage=None):
    ranks, found = [], []
    for row in rows:
        molecule_id, truth = row["molecule_id"], row["inchikey14"]
        candidates = rank_lists.get(molecule_id, ())
        ranks.append(next((i for i, k in enumerate(candidates[:TOP_K], 1) if k == truth), 0))
        if coverage is not None:
            found.append(truth in coverage.get(molecule_id, ()))
    count = len(ranks)
    result = {
        "molecules": count,
        "mrr_at_25": sum(1 / rank for rank in ranks if rank) / count,
        "top_1": ranks.count(1) / count,
        "recall_at_25": sum(bool(rank) for rank in ranks) / count,
    }
    if coverage is not None:
        result["candidate_recall"] = sum(found) / count
        sizes = [len(coverage.get(row["molecule_id"], ())) for row in rows]
        result["candidate_count"] = {
            "median": float(np.median(sizes)),
            "p95": float(np.percentile(sizes, 95)),
            "max": max(sizes),
        }
    return result


def score_groups(rows, rank_lists, coverage=None):
    def breakdown(field):
        groups = defaultdict(list)
        for row in rows:
            if field in row and row[field] is not None:
                groups[str(row[field])].append(row)
        return {key: metrics(group, rank_lists, coverage) for key, group in sorted(groups.items())}

    by_class = breakdown("novelty")
    class_1 = [row for row in rows if row["novelty"] == "class_1"]

    def reference_overlap(field, same, different):
        groups = defaultdict(list)
        for row in class_1:
            groups[same if row[field] else different].append(row)
        return {key: metrics(group, rank_lists, coverage) for key, group in sorted(groups.items())}

    macro = None
    sensitivity = None
    if all(label in by_class for label in CLASSES):
        mrrs = [by_class[label]["mrr_at_25"] for label in CLASSES]
        macro = sum(mrrs) / len(CLASSES)
        sensitivity = {
            "class_counts": {label: by_class[label]["molecules"] for label in CLASSES},
            "proxy_note": "class_2/class_3_proxy reflect local reference and attached-catalog coverage; hidden Kaggle class proportions are undisclosed.",
            "weighted_mrr_at_25": {
                name: sum(w * m for w, m in zip(weights, mrrs))
                for name, weights in CLASS_MIX_SCENARIOS.items()
            },
        }
    scores = {
        "observed_mixture": metrics(rows, rank_lists, coverage),
        "equal_class_mrr_at_25": macro,
        "class_mix_sensitivity": sensitivity,
        "by_class": by_class,
        "by_domain": breakdown("domain"),
        "by_source": breakdown("source"),
        "by_instrument": breakdown("instrument"),
        "by_tanimoto_bin": breakdown("tanimoto_bin"),
        "class_1_by_tier": {
            key: metrics(group, rank_lists, coverage)
            for key, group in sorted(
                defaultdict(list, {
                    t: [r for r in class_1 if r.get("class_1_tier") == t]
                    for t in {r.get("class_1_tier") for r in class_1 if r.get("class_1_tier")}
                }).items()
            )
        },
        "class_1_by_adduct_overlap": reference_overlap(
            "same_adduct_reference", "same_adduct", "different_adduct_only",
        ),
        "class_1_by_instrument_overlap": reference_overlap(
            "same_instrument_reference", "same_instrument", "different_instrument_only",
        ),
    }
    focused = [row for row in rows if row.get("focused_panel", False)]
    if 0 < len(focused) < len(rows):
        scores["focused_panel"] = score_groups(focused, rank_lists, coverage)
    return scores


def fold_labels(split, fold, panel, subset="all"):
    labels = pq.read_table(split / f"{fold}_labels.parquet").to_pylist()
    if panel == "focused":
        labels = [row for row in labels if row.get("focused_panel", False)]
        if not labels:
            raise ValueError("No focused panel in this fold; build a broad split or select --panel all")
    if subset == "in_catalog":
        labels = [row for row in labels if row.get("in_catalog")]
    elif subset == "timstof":
        labels = [row for row in labels if row.get("instrument") == "timsTOF"]
    elif subset == "timstof_np":
        labels = [row for row in labels if row.get("domain") == "timsTOF_np"]
    if not labels:
        raise ValueError(f"No molecules matched fold={fold}, panel={panel}, subset={subset}")
    return labels


def evaluate(args):
    split, fold = args.split, args.fold
    manifest = load_manifest(split)
    key_map = None
    if getattr(args, "key_cache", None):
        key_map, _ = load_key_cache(args.key_cache, args.train)
    identity = identity_for(manifest["identity"], key_map)
    if file_record(args.train) != manifest["inputs"]["train"]:
        raise ValueError("Training parquet differs from the frozen manifest")
    if [file_record(path) for path in args.catalog] != manifest["inputs"]["catalogs"]:
        raise ValueError("Pass the frozen catalog files in the same order as build")
    additional_catalogs = getattr(args, "additional_catalog", ())
    retrieval_catalogs = [*args.catalog, *additional_catalogs]
    panel = args.panel
    formulas_path = getattr(args, "formulas", None)
    if formulas_path is not None and not retrieval_catalogs:
        raise ValueError("Formula evidence requires a structure catalog")
    query_path = split / f"{fold}_queries.parquet"
    formulas = {}
    if formulas_path is not None:
        query_ids = pq.read_table(query_path, columns=["molecule_id"])["molecule_id"].to_pylist()
        formulas = load_formulas(formulas_path, set(query_ids))
    labels = fold_labels(split, fold, panel)
    masked = set()
    if args.simulate != "none":
        labels = [
            {**row, "novelty": "masked_catalog_only" if args.simulate == "catalog-only" else "masked_catalog_absent"}
            for row in labels if row.get("in_catalog")
        ]
        if not labels:
            raise ValueError("Masking needs structures present in the frozen catalog")
        masked = {row["inchikey14"] for row in labels}
    filters = None
    if panel == "focused" or masked:
        filters = [("molecule_id", "in", [row["molecule_id"] for row in labels])]
    query_rows = pq.read_table(query_path, filters=filters).to_pylist()

    molecules = defaultdict(Molecule)
    for row in query_rows:
        molecules[row["molecule_id"]].add_spectrum(row)
    excluded = set(pq.read_table(split / "excluded_reference_spectra.parquet")["signature"].to_pylist())
    excluded_row_ids = None
    row_id_file = split / "excluded_reference_row_ids.parquet"
    if row_id_file.exists():
        excluded_row_ids = set(pq.read_table(row_id_file)["row_id"].to_pylist())
    excluded_sources = manifest["query_sources"]
    excluded_key_sources = set()
    if manifest.get("split_mode") == "broad":
        excluded_sources = ()
        assignment = pq.read_table(split / "assignments.parquet", columns=["inchikey14", "query_source"])
        excluded_key_sources = {
            (key, source)
            for key, source in zip(assignment["inchikey14"].to_pylist(), assignment["query_source"].to_pylist())
            if source is not None
        }
    search_library(
        args.train, list(molecules.values()), excluded_sources=excluded_sources,
        excluded_keys=masked, excluded_spectra=excluded,
        excluded_key_sources=excluded_key_sources, excluded_row_ids=excluded_row_ids,
        strict_only=True, identity=identity,
    )
    library_lists = {key: [hit[2] for hit in molecule.ranked()] for key, molecule in molecules.items()}
    library_pool = {key: molecule.candidates for key, molecule in molecules.items()}
    result = {
        "fold": fold, "panel": panel, "simulation": args.simulate, "identity": manifest["identity"],
        "catalogs": manifest["inputs"]["catalogs"],
        "library": score_groups(labels, library_lists, library_pool),
    }
    if additional_catalogs:
        result["retrieval_catalogs"] = [
            *result["catalogs"], *(file_record(path) for path in additional_catalogs),
        ]
    filtered_pool = {}
    if retrieval_catalogs:
        hidden_catalog = masked if args.simulate == "catalog-absent" else ()
        mass_pool, filtered_pool = search_catalogs(
            retrieval_catalogs, query_rows, identity=identity, ppm_limit=args.ppm,
            excluded_keys=hidden_catalog, formulas=formulas,
        )
        result["catalog_ppm"] = args.ppm

        def score_catalog(pool, catalog_name, combined_name):
            catalog_lists = {key: mass_order(hits)[:TOP_K] for key, hits in pool.items()}
            combined = {key: merge_candidates(ranks, catalog_lists.get(key, ()), TOP_K)
                        for key, ranks in library_lists.items()}
            combined_pool = {key: hits.keys() | pool.get(key, {}).keys() for key, hits in library_pool.items()}
            result[catalog_name] = score_groups(labels, catalog_lists, pool)
            result[combined_name] = score_groups(labels, combined, combined_pool)

        score_catalog(mass_pool, "catalog_mass_only", "library_plus_catalog")
        if formulas_path is not None:
            result["formula_evidence"] = {
                "input": file_record(formulas_path), "molecules": len(formulas.keys() & molecules.keys()),
            }
            score_catalog(filtered_pool, "catalog_mass_and_formula", "library_plus_catalog_formula")
    candidate_output = getattr(args, "candidate_output", None)
    if candidate_output is not None:
        export_candidates(candidate_output, molecules, filtered_pool)
    if args.output:
        json_file(args.output, result)
    print(json.dumps(result, indent=2))


def export_train(args):
    manifest = load_manifest(args.split)
    key_map = None
    if getattr(args, "key_cache", None):
        key_map, _ = load_key_cache(args.key_cache, args.train)
    identity = identity_for(manifest["identity"], key_map)
    if file_record(args.train) != manifest["inputs"]["train"]:
        raise ValueError("Training parquet differs from the frozen manifest")
    assignment = pq.read_table(args.split / "assignments.parquet")
    allowed = set(assignment.filter(pc.equal(assignment["fold"], "train"))["inchikey14"].to_pylist())
    excluded = set(pq.read_table(args.split / "excluded_fit_spectra.parquet")["signature"].to_pylist())
    parquet = pq.ParquetFile(args.train)
    if args.output.exists():
        raise ValueError(f"{args.output} already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent, prefix=".casmi-fit-") as tmp:
        temp = Path(tmp) / "train.parquet"
        with pq.ParquetWriter(temp, parquet.schema_arrow, compression="zstd") as writer:
            for batch in parquet.iter_batches(batch_size=32768):
                indices = [
                    i for i, (smiles, key) in enumerate(zip(
                        batch["normalized_smiles"].to_pylist(), batch["inchikey14"].to_pylist(),
                    )) if smiles and identity(smiles, key) in allowed
                ]
                if indices:
                    batch = batch.take(indices)
                    keep = [
                        suitable(row, query=False) not in excluded
                        for row in batch.select(QUERY_COLUMNS).to_pylist()
                    ]
                    writer.write_batch(batch.filter(keep))
        temp.rename(args.output)


def _read_molecule_lists(path, column, ids, identity, max_items=None):
    out = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"molecule_id", column} <= set(reader.fieldnames or []):
            raise ValueError(f"{path}: must contain molecule_id and {column}")
        for row in reader:
            key = row["molecule_id"]
            if key not in ids:
                continue
            if key in out:
                raise ValueError(f"Duplicate molecule_id: {key}")
            values = row[column].split(";") if row[column] else []
            if max_items is not None and len(values) > max_items:
                raise ValueError(f"More than {max_items} guesses for molecule: {key}")
            # Invalid and repeated guesses retain their submitted rank positions.
            out[key] = [
                identity(value, value if column == "connectivity_keys" else None)
                if value.strip() else None for value in values
            ]
    return out


def score_predictions(args):
    manifest = load_manifest(args.split)
    key_map = None
    if getattr(args, "key_cache", None):
        key_map, _ = load_key_cache(args.key_cache)
    identity = identity_for(manifest["identity"], key_map)
    subset = getattr(args, "subset", "all")
    labels = fold_labels(args.split, args.fold, args.panel, subset=subset)
    ids = {row["molecule_id"] for row in labels}
    column = "connectivity_keys" if manifest["identity"] == "provided" else "smiles"
    # Verify no unexpected molecule IDs when scoring full fold/panel
    rankings = {}
    with args.predictions.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"molecule_id", column} <= set(reader.fieldnames or []):
            raise ValueError(f"Predictions must contain molecule_id and {column}")
        for row in reader:
            key = row["molecule_id"]
            values = row[column].split(";") if row[column] else []
            if subset == "all" and key not in ids:
                raise ValueError(f"Unexpected/duplicate molecule or more than 25 guesses: {key}")
            if key not in ids:
                continue
            if key in rankings or len(values) > TOP_K:
                raise ValueError(f"Unexpected/duplicate molecule or more than 25 guesses: {key}")
            rankings[key] = [
                identity(value, value if column == "connectivity_keys" else None)
                if value.strip() else None for value in values
            ]
    if set(rankings) != ids:
        raise ValueError("Predictions must contain every molecule in the selected fold")
    coverage = None
    if getattr(args, "candidates", None):
        cand_lists = _read_molecule_lists(args.candidates, column, ids, identity, max_items=None)
        coverage = {k: {c for c in v if c} for k, v in cand_lists.items()}
    result = score_groups(labels, rankings, coverage)
    if getattr(args, "output", None):
        json_file(args.output, result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prep_train = commands.add_parser("prepare-train", help="parallel RDKit 2026.03.3 key + Morgan FP cache for train.parquet")
    prep_train.add_argument("--train", type=Path, required=True)
    prep_train.add_argument("--output", type=Path, required=True)
    prep_train.add_argument("--workers", type=int)
    prep_train.add_argument("--overwrite", action="store_true")
    prep_train.set_defaults(run=prepare_train)

    prep_cat = commands.add_parser("prepare-catalog", help="desalt, canonicalize, and mass-sort a structure catalog")
    prep_cat.add_argument("--input", type=Path, required=True)
    prep_cat.add_argument("--output", type=Path, required=True)
    prep_cat.add_argument("--smiles-col")
    prep_cat.add_argument("--max-mass", type=float, help="skip molecules heavier than this neutral mass (Da) before tautomerization")
    prep_cat.add_argument("--workers", type=int)
    prep_cat.add_argument("--overwrite", action="store_true")
    prep_cat.set_defaults(run=prepare_catalog)

    build = commands.add_parser("build", help="freeze a split and its input snapshots")
    build.add_argument("--train", type=Path, required=True)
    build.add_argument("--key-cache", type=Path, help="precomputed train_canonical.parquet from prepare-train")
    build.add_argument("--catalog", type=Path, action="append", default=[])
    build.add_argument("--query-source", action="append", default=[])
    build.add_argument("--split-mode", choices=("broad", "focused"), default="broad",
                       help="broad 90/5/5 folds or the original capped source-held-out benchmark")
    build.add_argument("--trust-keys", action="store_true")
    build.add_argument("--max-per-class", type=int, default=250,
                       help="combined dev and test class cap for the focused benchmark or nested panels")
    build.add_argument("--seed", type=int, default=2026)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(run=build_split)

    attach = commands.add_parser("attach-catalog", help="refresh catalog coverage labels on a frozen split")
    attach.add_argument("--split", type=Path, required=True)
    attach.add_argument("--catalog", type=Path, action="append", required=True)
    attach.set_defaults(run=attach_catalog)

    evaluation = commands.add_parser("evaluate", help="run the CPU baselines on a frozen fold")
    evaluation.add_argument("--split", type=Path, required=True)
    evaluation.add_argument("--train", type=Path, required=True)
    evaluation.add_argument("--key-cache", type=Path)
    evaluation.add_argument("--catalog", type=Path, action="append", default=[])
    evaluation.add_argument("--additional-catalog", type=Path, action="append", default=[],
                            help="retrieve from this extra catalog while keeping the frozen cohort labels")
    evaluation.add_argument("--fold", choices=("dev", "test"), default="dev")
    evaluation.add_argument("--panel", choices=("all", "focused"), default="all")
    evaluation.add_argument("--simulate", choices=("none", "catalog-only", "catalog-absent"), default="none")
    evaluation.add_argument("--ppm", type=float, default=10)
    evaluation.add_argument("--formulas", type=Path, help="CSV with independent molecule_id,formula predictions")
    evaluation.add_argument("--candidate-output", type=Path, help="write all retrieved candidates to Parquet")
    evaluation.add_argument("--output", type=Path)
    evaluation.set_defaults(run=evaluate)

    export = commands.add_parser("export-train", help="materialize the structure-disjoint training fold")
    export.add_argument("--split", type=Path, required=True)
    export.add_argument("--train", type=Path, required=True)
    export.add_argument("--key-cache", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.set_defaults(run=export_train)

    score = commands.add_parser("score", help="score a model's molecule-level predictions")
    score.add_argument("--split", type=Path, required=True)
    score.add_argument("--fold", choices=("dev", "test"), default="dev")
    score.add_argument("--panel", choices=("all", "focused"), default="all")
    score.add_argument("--subset", choices=("all", "in_catalog", "timstof", "timstof_np"), default="all")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--candidates", type=Path, help="optional candidate pool CSV for candidate_recall")
    score.add_argument("--key-cache", type=Path)
    score.add_argument("--output", type=Path)
    score.set_defaults(run=score_predictions)

    args = parser.parse_args()
    if getattr(args, "max_per_class", 1) < 1:
        parser.error("--max-per-class must be positive")
    ppm = getattr(args, "ppm", 1)
    if not math.isfinite(ppm) or ppm <= 0:
        parser.error("--ppm must be positive and finite")
    try:
        args.run(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
