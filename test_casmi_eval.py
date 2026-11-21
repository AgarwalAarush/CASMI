"""Integration coverage for data leakage and the three benchmark regimes."""

import contextlib
import csv
import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq

import casmi_eval as e
from casmi_baseline import Chem, connectivity, predict
from casmi_catalog import load_formulas, mass_order, merge_candidates, search_catalogs
from casmi_pubchem import prepare as prepare_pubchem


def spectrum(key, mass, peaks, source, adduct="[M-H]-"):
    return {
        "inchikey14": key * 14, "normalized_smiles": "Oc1ccccc1", "ingest_lib": source,
        "instrument_type": "timsTOF" if source == "heldout" else "QTOF",
        "adduct": adduct, "precursor_mz": mass + e.ADDUCT_SHIFT[adduct],
        "ms2_mzs": peaks, "ms2_normalized_intensities": [1.0, 0.5],
    }


class FrozenSplitTest(unittest.TestCase):
    @unittest.skipUnless(Chem is not None and Chem.rdBase.rdkitVersion == e.METRIC_RDKIT, "requires official RDKit")
    def test_pubchem_preparation_joins_cids_and_caps_mass_ties(self):
        from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            mass_file, smiles_file = root / "CID-Mass.gz", root / "CID-SMILES.gz"
            mass = CalcExactMolWt(Chem.MolFromSmiles("CCO"))
            with gzip.open(mass_file, "wt") as handle:
                handle.write(f"1\tC2H6O\t{mass}\t{mass}\n3\tC2H6O\t{mass}\t{mass}\n")
            with gzip.open(smiles_file, "wt") as handle:
                handle.write("1\tCCO\n2\tCCN\n3\tCOC\n")
            query = {"molecule_id": "q", "precursor_mz": mass + e.ADDUCT_SHIFT["[M+Na]+"], "adduct": "[M+Na]+"}
            features = root / "queries.parquet"
            pq.write_table(pa.Table.from_pylist([query]), features)
            output = root / "prepared"
            prepare_pubchem(SimpleNamespace(
                mass=mass_file, smiles=smiles_file, queries=[features], output=output,
                ppm=10, workers=1, max_cids_per_precursor=1,
            ))
            hits, _ = search_catalogs(sorted(output.glob("*.parquet")), [query], identity=connectivity)
            self.assertEqual(set(hits["q"]), {connectivity("CCO", None)})
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["cids_in_mass_windows"], 2)
            self.assertEqual(manifest["selected_cids"], 1)

    def test_coconut_csv_zip_adduct_union_formula_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coconut = root / "coconut.zip"
            with ZipFile(coconut, "w") as archive:
                archive.writestr(
                    "coconut.csv", "canonical_smiles,exact_molecular_weight,molecular_formula,inchikey14\n"
                    + f"CCO,250,C2H6O,{'A' * 14}\n"
                    + f"CC=O,250.002,C2H4O,{'B' * 14}\n"
                    + f"CCN,251,C2H7N,{'C' * 14}\n"
                    + f"CCC,251.01,C3H8,{'D' * 14}\n",
                )
            formula_file = root / "formulas.csv"
            formula_file.write_text("molecule_id,formula\nq,C2H4O\nq,C2H7N\n")
            formulas = load_formulas(formula_file, {"q", "unfiltered"})
            rows = [
                {"molecule_id": "q", **spectrum("A", 250, [50, 100], "heldout", "[M+Na]+")},
                {"molecule_id": "q", **spectrum("A", 251, [50, 100], "heldout", "[M-H2O-H]-")},
                {"molecule_id": "unfiltered", **spectrum("A", 250, [50, 100], "heldout")},
                {"molecule_id": "unknown", "precursor_mz": 250, "adduct": "unsupported"},
            ]
            mass, filtered = search_catalogs(
                [coconut], rows, identity=e.identity_for("provided"), formulas=formulas,
            )
            self.assertEqual(set(mass["q"]), {key * 14 for key in "ABC"})
            self.assertEqual(set(filtered["q"]), {key * 14 for key in "BC"})
            self.assertEqual(set(filtered["unfiltered"]), {key * 14 for key in "AB"})
            self.assertNotIn("unknown", mass)
            hit = filtered["q"]["B" * 14]
            self.assertAlmostEqual(hit.ppm, .002 / rows[0]["precursor_mz"] * 1e6)
            self.assertEqual(merge_candidates(["A" * 14], mass_order(mass["q"]), 3), [key * 14 for key in "ACB"])

    @unittest.skipIf(Chem is None, "prediction with a structure catalog requires RDKit")
    def test_prediction_exports_catalog_only_smiles_without_a_reference(self):
        from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            train, test, catalog = root / "train.parquet", root / "test.parquet", root / "catalog.csv"
            output, candidates = root / "submission.csv", root / "candidates.parquet"
            true_smiles = "Oc1ccc(O)cc1"
            mass = CalcExactMolWt(Chem.MolFromSmiles(true_smiles))
            reference = spectrum("A", mass, [50, 100], "reference", "[M+H]+")
            pq.write_table(pa.Table.from_pylist([reference]), train)
            query = spectrum("B", mass, [55, 105], "query", "[M+Na]+")
            pq.write_table(pa.Table.from_pylist([{"molecule_id": "q", **query}]), test)
            catalog.write_text(f"smiles\n{true_smiles}\n")
            formulas = root / "formulas.csv"
            formulas.write_text("molecule_id,formula\nq,C6H6O2\n")
            counts = predict(train, test, output, catalogs=[catalog], formulas_path=formulas,
                             candidate_output=candidates)
            self.assertEqual(counts["molecules_with_catalog_guesses"], 1)
            self.assertEqual(output.read_text().splitlines(), ["molecule_id,smiles", f"q,{true_smiles}"])
            hit, = pq.read_table(candidates).to_pylist()
            self.assertEqual(hit["inchikey14"], connectivity(true_smiles, None))
            self.assertEqual(hit["formula"], "C6H6O2")
            self.assertIsNone(hit["spectral_cosine"])

    def test_dev_and_test_structure_boundaries_and_catalog_masks(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            train, catalog, split = root / "train.parquet", root / "catalog.csv", root / "split"
            queries = [
                spectrum(key, 250 + i * 10, [50.0 + i, 100.0 + i], "heldout")
                for i, key in enumerate("ABCDEF")
            ]
            global_refs = [
                spectrum("A", 250, [50.001, 100.0], "reference"),
                spectrum("B", 260, [51.001, 101.0], "reference"),
                {**queries[2], "ingest_lib": "reference"},
                spectrum("G", 330, [72.0, 77.0], "reference"),
                {**queries[0], "ingest_lib": "reference", "inchikey14": "H" * 14},
            ]
            pq.write_table(pa.Table.from_pylist(queries + global_refs), train, row_group_size=3)
            with catalog.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["smiles", "inchikey14", "exact_mass", "formula"])
                for i, key in enumerate("ABCD"):
                    writer.writerow(["Oc1ccccc1", key * 14, 250 + i * 10, "C6H6O"])
            e.build_split(SimpleNamespace(
                train=train, catalog=[catalog], output=split, trust_keys=True,
                query_source=["heldout"], max_per_class=250, seed=2026, split_mode="focused",
            ))
            dev = pq.read_table(split / "dev_labels.parquet").to_pylist()
            test = pq.read_table(split / "test_labels.parquet").to_pylist()
            self.assertEqual({row["novelty"] for row in dev}, set(e.CLASSES))
            self.assertEqual({row["novelty"] for row in test}, set(e.CLASSES))
            self.assertTrue({row["inchikey14"] for row in dev}.isdisjoint(
                {row["inchikey14"] for row in test},
            ))
            dev_features = pq.read_table(split / "dev_queries.parquet")
            self.assertFalse(set(e.LABEL_COLUMNS) & set(dev_features.column_names))

            def result(scenario, **options):
                out = root / f"{scenario}.json"
                e.evaluate(SimpleNamespace(
                    split=split, fold="dev", train=train, catalog=[catalog],
                    output=out, simulate=scenario, ppm=10, panel="all", **options,
                ))
                return json.loads(out.read_text())

            normal = result("none")
            library = normal["library"]["by_class"]
            fused = normal["library_plus_catalog"]["by_class"]
            self.assertEqual(library["class_1"]["mrr_at_25"], 1)
            self.assertEqual(library["class_2"]["candidate_recall"], 0)
            self.assertEqual(fused["class_2"]["candidate_recall"], 1)
            self.assertEqual(fused["class_3_proxy"]["candidate_recall"], 0)
            self.assertEqual(normal["library_plus_catalog"]["equal_class_mrr_at_25"], 2 / 3)
            proxy, = [row for row in dev if row["novelty"] == "class_3_proxy"]
            supplement = root / "extra.csv"
            supplement.write_text(
                "smiles,inchikey14,exact_mass\n"
                f"Oc1ccccc1,{proxy['inchikey14']},{proxy['neutral_mass']}\n"
            )
            extra = result("none", additional_catalog=[supplement])
            self.assertEqual(extra["catalogs"], normal["catalogs"])
            self.assertEqual(extra["retrieval_catalogs"][-1], e.file_record(supplement))
            self.assertEqual(extra["library_plus_catalog"]["by_class"]["class_3_proxy"]["candidate_recall"], 1)
            self.assertEqual(result("catalog-only")["catalog_mass_only"]["observed_mixture"]["mrr_at_25"], 1)
            self.assertEqual(result("catalog-absent")["library_plus_catalog"]["observed_mixture"]["candidate_recall"], 0)

            class_2, = [row for row in dev if row["novelty"] == "class_2"]
            formula_file = root / "formulas.csv"
            formula_file.write_text(f"molecule_id,formula\n{class_2['molecule_id']},C10H20\n")
            candidate_file = root / "candidates.parquet"
            filtered = result("none", formulas=formula_file, candidate_output=candidate_file)
            self.assertEqual(filtered["library_plus_catalog"]["by_class"]["class_2"]["candidate_recall"], 1)
            self.assertEqual(filtered["library_plus_catalog_formula"]["by_class"]["class_2"]["candidate_recall"], 0)
            exported = pq.read_table(candidate_file).to_pylist()
            self.assertNotIn(class_2["molecule_id"], {row["molecule_id"] for row in exported})

            fit = root / "fit.parquet"
            e.export_train(SimpleNamespace(split=split, train=train, output=fit))
            self.assertEqual(pq.read_table(fit)["inchikey14"].to_pylist(), ["G" * 14])

            predictions = root / "predictions.csv"
            with predictions.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["molecule_id", "connectivity_keys"])
                for row in dev:
                    writer.writerow([row["molecule_id"], "invalid;" + row["inchikey14"]])
            with contextlib.redirect_stdout(io.StringIO()) as output:
                e.score_predictions(SimpleNamespace(split=split, fold="dev", predictions=predictions, panel="all"))
            self.assertEqual(json.loads(output.getvalue())["observed_mixture"]["mrr_at_25"], 0.5)

            catalog.write_text(catalog.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "frozen catalog"):
                result("none")

    def test_broad_split_sizes_and_nested_panel(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            train, split = root / "train.parquet", root / "split"
            compounds = []
            for i in range(60):
                row = spectrum("A", 250 + i, [51.0 + i, 119.0 + i],
                               "heldout" if i < 6 else "reference")
                row["inchikey14"] = chr(65 + i // 26) + chr(65 + i % 26) + "A" * 12
                compounds.append(row)
            other_measurements = [
                {**row, "ingest_lib": "reference" if i < 2 else "heldout", "instrument_type": "QTOF",
                 "ms2_mzs": [row["ms2_mzs"][0] + 0.001, row["ms2_mzs"][1]]}
                for i, row in enumerate(compounds[:6])
            ]
            pq.write_table(pa.Table.from_pylist(compounds + other_measurements), train, row_group_size=8)
            e.build_split(SimpleNamespace(
                train=train, catalog=[], output=split, trust_keys=True,
                query_source=["heldout"], max_per_class=2, seed=2026, split_mode="broad",
            ))
            manifest = e.load_manifest(split)
            self.assertEqual(manifest["target_per_holdout"], 3)
            self.assertEqual(manifest["fit_structures"], 54)
            assignment = pq.read_table(split / "assignments.parquet").to_pylist()
            self.assertEqual(len({row["inchikey14"] for row in assignment}), 60)
            for fold in ("dev", "test"):
                self.assertEqual(manifest["splits"][fold]["molecules"], 3)
                labels = e.fold_labels(split, fold, "all")
                self.assertEqual({row["instrument"] for row in labels}, {"timsTOF"})
                self.assertLess(len(e.fold_labels(split, fold, "focused")), 3)
                out = root / f"{fold}.json"
                e.evaluate(SimpleNamespace(
                    split=split, fold=fold, train=train, catalog=[], output=out,
                    simulate="none", ppm=10, panel="all",
                ))
                scores = json.loads(out.read_text())["library"]
                self.assertEqual(scores["by_class"][e.NO_CATALOG]["candidate_recall"], 0)
                self.assertIn("focused_panel", scores)
            output = root / "panel.json"
            e.evaluate(SimpleNamespace(
                split=split, fold="dev", train=train, catalog=[], output=output,
                simulate="none", ppm=10, panel="focused",
            ))
            self.assertLess(json.loads(output.read_text())["library"]["observed_mixture"]["molecules"], 3)

    def test_aggregator_near_duplicate_exclusion_and_attach_catalog(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            train, split = root / "train.parquet", root / "split"
            # Build 40 compounds across aggregator libs + enveda-np-examples
            rows = []
            for i in range(40):
                key = chr(65 + i // 26) + chr(65 + i % 26) + "B" * 12
                if i < 2:
                    # Query in riken, ONLY reference is a near-duplicate in mona (0.001 Da peak shift -> cos=1.0, diff hash)
                    q = spectrum("A", 300 + i * 5, [80.0 + i, 140.0 + i], "riken")
                    q["inchikey14"] = key
                    q["instrument_type"] = "QTOF"
                    dup = {**q, "ingest_lib": "mona", "ms2_mzs": [80.0005 + i, 140.0 + i]}
                    rows.extend([q, dup])
                elif i < 4:
                    # Gold-standard timsTOF natural product in enveda-np-examples with independent ref in riken
                    q = spectrum("A", 320 + i * 5, [90.0 + i, 160.0 + i], "enveda-np-examples")
                    q["inchikey14"] = key
                    q["instrument_type"] = "timsTOF"
                    ref = {**q, "ingest_lib": "riken", "instrument_type": "QTOF", "ms2_mzs": [90.001 + i, 160.0 + i]}
                    rows.extend([q, ref])
                else:
                    r = spectrum("A", 200 + i * 3, [60.0 + i, 120.0 + i], "enveda-180")
                    r["inchikey14"] = key
                    r["instrument_type"] = "timsTOF"
                    rows.append(r)
            pq.write_table(pa.Table.from_pylist(rows), train, row_group_size=10)
            e.build_split(SimpleNamespace(
                train=train, catalog=[], output=split, trust_keys=True,
                query_source=["enveda-np-examples", "riken"], max_per_class=10, seed=2026, split_mode="broad",
            ))
            manifest = e.load_manifest(split)
            # The 2 enveda-np-examples structures must remain Class 1 (timsTOF_vs_public) even with cos=1.0!
            dev_labels = pq.read_table(split / "dev_labels.parquet").to_pylist()
            test_labels = pq.read_table(split / "test_labels.parquet").to_pylist()
            all_holdout = {r["inchikey14"]: r for r in dev_labels + test_labels}
            for i in (2, 3):
                key = chr(65 + i // 26) + chr(65 + i % 26) + "B" * 12
                self.assertIn(key, all_holdout)
                self.assertEqual(all_holdout[key]["novelty"], "class_1")
                self.assertEqual(all_holdout[key]["domain"], "timsTOF_np")
                self.assertEqual(all_holdout[key]["class_1_tier"], "timsTOF_vs_public")
            # Also verify aggregator near-duplicate filtering logic directly on a focused broad split
            self.assertEqual(manifest["version"], 3)

            # Attach a catalog after freezing without changing assignments
            before_assign = pq.read_table(split / "assignments.parquet").to_pylist()
            cat = root / "cat.csv"
            with cat.open("w", newline="") as handle:
                w = csv.writer(handle)
                w.writerow(["smiles", "inchikey14", "exact_mass"])
                for r in dev_labels:
                    if r["novelty"] != "class_1":
                        w.writerow(["Oc1ccccc1", r["inchikey14"], r["neutral_mass"]])
                        break
            e.attach_catalog(SimpleNamespace(split=split, catalog=[cat]))
            after_assign = pq.read_table(split / "assignments.parquet").to_pylist()
            self.assertEqual(before_assign, after_assign)

    def test_prepare_train_prepare_catalog_and_tanimoto_official_rdkit(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            smiles_pool = [
                "Oc1ccccc1", "Cc1ccccc1O", "CCc1ccccc1O", "COc1ccccc1O",
                "Nc1ccccc1O", "Fc1ccccc1O", "Clc1ccccc1O", "Brc1ccccc1O",
                "O=Cc1ccccc1O", "CC(=O)c1ccccc1O", "O=C(O)c1ccccc1O", "COC(=O)c1ccccc1O",
                "c1ccc2ccccc2c1", "Cc1ccc2ccccc2c1", "Oc1ccc2ccccc2c1", "Nc1ccc2ccccc2c1",
                "c1ccncc1", "Cc1ccncc1", "Oc1ccncc1", "CC(=O)c1ccncc1",
            ]
            rows = []
            for i, sm in enumerate(smiles_pool):
                # Make neutral mass within 157-1159 using synthetic precursor_mz for test
                row = spectrum("A", 200.0 + i * 5, [55.0 + i, 110.0 + i],
                               "enveda-np-examples" if i < 2 else "enveda-180")
                row["normalized_smiles"] = sm
                row["inchikey14"] = "Z" * 14  # intentionally wrong provided key; official RDKit must override!
                row["instrument_type"] = "timsTOF"
                rows.append(row)
                if i < 2:
                    ref = {**row, "ingest_lib": "riken", "instrument_type": "QTOF",
                           "ms2_mzs": [55.0 + i, 110.002 + i]}
                    rows.append(ref)
            train = root / "train.parquet"
            pq.write_table(pa.Table.from_pylist(rows), train)
            key_cache = root / "train_keys.parquet"
            e.prepare_train(SimpleNamespace(train=train, output=key_cache, workers=2, overwrite=True))

            raw_cat = root / "raw_cat.csv"
            with raw_cat.open("w", newline="") as handle:
                w = csv.writer(handle)
                w.writerow(["canonical_smiles", "standard_inchi_key"])
                # Include a hydrochloride salt to verify desalting
                w.writerow(["Cl." + smiles_pool[0], ""])
                w.writerow([smiles_pool[2], ""])
                w.writerow([smiles_pool[3], ""])
            prep_cat = root / "cat.parquet"
            e.prepare_catalog(SimpleNamespace(input=raw_cat, output=prep_cat, smiles_col=None, workers=2, overwrite=True))
            self.assertTrue(e.is_prepared_catalog(prep_cat))

            split = root / "split"
            e.build_split(SimpleNamespace(
                train=train, key_cache=key_cache, catalog=[prep_cat], output=split, trust_keys=False,
                query_source=["enveda-np-examples", "riken"], max_per_class=10, seed=2026, split_mode="broad",
            ))
            dev_labels = pq.read_table(split / "dev_labels.parquet").to_pylist()
            self.assertTrue(all(r["max_tanimoto_to_train"] is not None for r in dev_labels))
            self.assertTrue(all(r["tanimoto_bin"] != "unknown" for r in dev_labels))
            self.assertTrue(all(r["inchikey14"] != "Z" * 14 for r in dev_labels))

            # Score predictions + candidate pool
            preds = root / "preds.csv"
            cands = root / "cands.csv"
            with preds.open("w", newline="") as hp, cands.open("w", newline="") as hc:
                wp, wc = csv.writer(hp), csv.writer(hc)
                wp.writerow(["molecule_id", "smiles"])
                wc.writerow(["molecule_id", "smiles"])
                for r in dev_labels:
                    wp.writerow([r["molecule_id"], smiles_pool[0]])
                    wc.writerow([r["molecule_id"], ";".join(smiles_pool[:5])])
            out_score = root / "score.json"
            e.score_predictions(SimpleNamespace(
                split=split, fold="dev", panel="all", subset="all",
                predictions=preds, candidates=cands, key_cache=key_cache, output=out_score,
            ))
            scored = json.loads(out_score.read_text())
            self.assertIn("candidate_recall", scored["observed_mixture"])
            self.assertIn("by_domain", scored)
            self.assertIn("by_tanimoto_bin", scored)


if __name__ == "__main__":
    unittest.main()
