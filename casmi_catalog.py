"""Retrieve catalog structures using precursor m/z and optional formula evidence."""

import csv
import gzip
import io
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcMolFormula
except ImportError:
    Chem = None


PROTON = 1.007276466621
WATER = 18.010564684
ADDUCT_SHIFT = {
    "[M+H]+": PROTON,
    "[M+NH4]+": 17.026549101 + PROTON,
    "[M-H2O+H]+": PROTON - WATER,
    "[M-2H2O+H]+": PROTON - 2 * WATER,
    "[M+Na]+": 22.989220702,
    "[M+K]+": 38.963157907,
    "[M-H]-": -PROTON,
    "[M-H2O-H]-": -PROTON - WATER,
    "[M+CH2O2-H]-": 46.005479304 - PROTON,
    "[M+Cl]-": 34.969401262,
}
CATALOG_COLUMNS = {
    "smiles": ("smiles", "canonical_smiles", "normalized_smiles"),
    "exact_mass": ("exact_mass", "exact_molecular_weight", "monoisotopic_mass"),
    "formula": ("formula", "molecular_formula"),
    "inchikey14": ("inchikey14",),
}
BATCH_SIZE = 32768
META_RDKIT_KEY = b"casmi_canonical_rdkit"
META_SORTED_MASS_KEY = b"casmi_sorted_mass"
OFFICIAL_RDKIT = "2026.03.3"


def is_prepared_catalog(path):
    if path.suffix != ".parquet":
        return False
    pf = pq.ParquetFile(path)
    meta = pf.schema_arrow.metadata or {}
    return (meta.get(META_RDKIT_KEY) or b"").decode() == OFFICIAL_RDKIT


def neutral_mass(mz, adduct):
    shift = ADDUCT_SHIFT.get(adduct)
    if shift is not None and mz is not None and math.isfinite(mz) and mz > 0:
        mass = mz - shift
        if mass > 0:
            return mass
    return None


def catalog_columns(path, fields):
    columns = {
        target: next((name for name in aliases if name in fields), None)
        for target, aliases in CATALOG_COLUMNS.items()
    }
    if columns["smiles"] is None:
        raise ValueError(f"{path}: catalog needs a smiles or canonical_smiles column")
    return {target: source for target, source in columns.items() if source is not None}


def csv_catalog(path, handle):
    # Full COCONUT CSV exports can contain large annotation fields.
    csv.field_size_limit(max(csv.field_size_limit(), 16 * 1024 * 1024))
    reader = csv.DictReader(handle)
    columns = catalog_columns(path, reader.fieldnames or ())
    for row in reader:
        yield {target: row[source] for target, source in columns.items()}


def catalog_rows(path):
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        columns = catalog_columns(path, parquet.schema_arrow.names)
        for batch in parquet.iter_batches(columns=list(columns.values()), batch_size=BATCH_SIZE):
            for row in batch.to_pylist():
                yield {target: row[source] for target, source in columns.items()}
    elif path.suffix == ".zip":
        with ZipFile(path) as archive:
            entries = [name for name in archive.namelist()
                       if not name.startswith("__MACOSX/") and name.lower().endswith(".csv")]
            if len(entries) != 1:
                raise ValueError(f"{path}: expected exactly one CSV in the catalog ZIP")
            with archive.open(entries[0]) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as handle:
                yield from csv_catalog(path, handle)
    elif path.suffix == ".csv" or path.name.endswith(".csv.gz"):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
            yield from csv_catalog(path, handle)
    else:
        raise ValueError(f"{path}: expected .parquet, .csv, .csv.gz, or a single-CSV .zip")


def load_formulas(path, molecule_ids):
    """Read an optional allowlist of independently assigned neutral formulas."""
    if path is None:
        return {}
    formulas = defaultdict(set)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"molecule_id", "formula"} <= set(reader.fieldnames or ()):
            raise ValueError(f"{path}: expected molecule_id and formula columns")
        for row in reader:
            molecule_id, formula = row["molecule_id"], (row["formula"] or "").strip()
            if molecule_id not in molecule_ids or not formula:
                raise ValueError(f"{path}:{reader.line_num}: expected a query molecule ID and neutral formula")
            formulas[molecule_id].add(formula)
    return formulas


@dataclass(frozen=True)
class CatalogHit:
    smiles: str
    exact_mass: float
    ppm: float
    formula: str | None
    catalog: str


def search_catalogs(paths, query_rows, *, identity, ppm_limit=10, excluded_keys=(), formulas=None):
    """Return mass-only hits and hits passing the optional formula allowlists."""
    if not math.isfinite(ppm_limit) or ppm_limit <= 0:
        raise ValueError("Catalog ppm tolerance must be positive and finite")
    windows = []
    for row in query_rows:
        mz = row["precursor_mz"]
        mass = neutral_mass(mz, row["adduct"])
        if mass is not None:
            windows.append((mass, mz, row["molecule_id"]))
    mass_only = defaultdict(dict)
    if not windows:
        return mass_only, mass_only
    formulas = formulas or {}
    filtered = defaultdict(dict) if formulas else mass_only
    windows.sort()
    query_masses = np.array([mass for mass, _, _ in windows])
    max_delta = max(mz for _, mz, _ in windows) * ppm_limit * 1e-6

    for path in paths:
        if is_prepared_catalog(path):
            tbl = pq.read_table(path, columns=[c for c in ("smiles", "inchikey14", "exact_mass", "formula")
                                               if c in pq.ParquetFile(path).schema_arrow.names])
            cat_masses = tbl["exact_mass"].to_numpy(zero_copy_only=False)
            cat_keys = tbl["inchikey14"].to_pylist()
            cat_smiles = tbl["smiles"].to_pylist()
            cat_formulas = tbl["formula"].to_pylist() if "formula" in tbl.column_names else [None] * len(cat_keys)
            meta = tbl.schema.metadata or {}
            if meta.get(META_SORTED_MASS_KEY) != b"true":
                order = np.argsort(cat_masses)
                cat_masses = cat_masses[order]
                cat_keys = [cat_keys[i] for i in order]
                cat_smiles = [cat_smiles[i] for i in order]
                cat_formulas = [cat_formulas[i] for i in order]
            for measured, mz, molecule_id in windows:
                delta = mz * ppm_limit * 1e-6
                lo = int(np.searchsorted(cat_masses, measured - delta))
                hi = int(np.searchsorted(cat_masses, measured + delta, side="right"))
                if lo == hi:
                    continue
                allowed = formulas.get(molecule_id)
                for idx in range(lo, hi):
                    key = cat_keys[idx]
                    if not key or key in excluded_keys:
                        continue
                    mass = float(cat_masses[idx])
                    ppm = abs(mass - measured) / mz * 1e6
                    if ppm > ppm_limit:
                        continue
                    formula = (cat_formulas[idx] or "").strip() or None
                    hit = CatalogHit(cat_smiles[idx], mass, ppm, formula, path.name)
                    keep_best(mass_only[molecule_id], key, hit)
                    if formulas and (not allowed or formula in allowed):
                        keep_best(filtered[molecule_id], key, hit)
            continue
        rows = catalog_rows(path)
        while batch := list(islice(rows, BATCH_SIZE)):
            masses = np.full(len(batch), np.nan)
            molecules = {}
            for i, row in enumerate(batch):
                if not row["smiles"]:
                    continue
                raw_mass = row.get("exact_mass")
                if raw_mass not in (None, ""):
                    try:
                        masses[i] = float(raw_mass)
                    except (TypeError, ValueError) as error:
                        raise ValueError(f"{path}: invalid exact mass {raw_mass!r}") from error
                elif Chem is not None:
                    molecule = Chem.MolFromSmiles(row["smiles"])
                    if molecule is not None:
                        masses[i] = CalcExactMolWt(molecule)
                        molecules[i] = molecule
                else:
                    raise ValueError(f"{path}: exact_mass is required when RDKit is unavailable")

            starts = np.searchsorted(query_masses, masses - max_delta)
            stops = np.searchsorted(query_masses, masses + max_delta, side="right")
            for i in np.flatnonzero((starts < stops) & np.isfinite(masses) & (masses > 0)):
                row, mass = batch[i], float(masses[i])
                smiles = row["smiles"]
                key = identity(smiles, row.get("inchikey14"))
                if not key or key in excluded_keys:
                    continue
                formula = (row.get("formula") or "").strip() or None
                for measured, mz, molecule_id in windows[starts[i]:stops[i]]:
                    ppm = abs(mass - measured) / mz * 1e6
                    if ppm > ppm_limit:
                        continue
                    allowed = formulas.get(molecule_id)
                    if allowed and formula is None:
                        if Chem is None:
                            raise ValueError(f"{path}: formula is required for formula filtering without RDKit")
                        molecule = molecules.get(i)
                        if molecule is None:
                            molecule = Chem.MolFromSmiles(smiles)
                        if molecule is not None:
                            formula = CalcMolFormula(molecule)
                    hit = CatalogHit(smiles, mass, ppm, formula, path.name)
                    keep_best(mass_only[molecule_id], key, hit)
                    if formulas and (not allowed or formula in allowed):
                        keep_best(filtered[molecule_id], key, hit)
    return mass_only, filtered


def keep_best(candidates, key, hit):
    previous = candidates.get(key)
    if previous is None or (hit.ppm, hit.catalog, hit.smiles) < (previous.ppm, previous.catalog, previous.smiles):
        candidates[key] = hit


def mass_order(hits):
    return sorted(hits, key=lambda key: (hits[key].ppm, key))


def merge_candidates(library, ranked_catalog, limit):
    return list(dict.fromkeys([*library, *ranked_catalog]))[:limit]


def export_candidates(path, molecules, catalog):
    """Write unlabeled candidate structures and score features for a downstream ranker."""
    schema = pa.schema([
        ("molecule_id", pa.string()), ("inchikey14", pa.string()), ("smiles", pa.string()),
        ("spectral_cosine", pa.float64()), ("spectral_ppm", pa.float64()),
        ("catalog_ppm", pa.float64()), ("exact_mass", pa.float64()),
        ("formula", pa.string()), ("catalog", pa.string()),
    ])
    with TemporaryDirectory(dir=path.parent, prefix=".casmi-candidates-") as tmp:
        stage = Path(tmp) / path.name
        with pq.ParquetWriter(stage, schema, compression="zstd") as writer:
            for molecule_id, molecule in sorted(molecules.items()):
                hits = catalog.get(molecule_id, {})
                rows = []
                for key in sorted(molecule.candidates.keys() | hits.keys()):
                    spectral, structure = molecule.candidates.get(key), hits.get(key)
                    rows.append({
                        "molecule_id": molecule_id, "inchikey14": key,
                        "smiles": structure.smiles if structure else spectral[3],
                        "spectral_cosine": spectral[0] if spectral else None,
                        "spectral_ppm": spectral[1] if spectral else None,
                        "catalog_ppm": structure.ppm if structure else None,
                        "exact_mass": structure.exact_mass if structure else None,
                        "formula": structure.formula if structure else None,
                        "catalog": structure.catalog if structure else None,
                    })
                if rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        stage.replace(path)
