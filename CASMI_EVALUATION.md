# CASMI development and local test

Use the labeled CASMI 2026 spectra for the main benchmark. The default is a 90/5/5 split by molecular connectivity, with a smaller CASMI-like panel inside each held-out fold. Keep independent reference measurements for Class 1. Use actual gaps in that library for Class 2. Use structure-catalog gaps for a Class 3 proxy.

The competition reports about 400 hidden molecules, roughly 1,500 timsTOF spectra, and a median of three spectra per molecule. The class proportions are withheld. The downloadable `test.parquet` is an example file. Kaggle scores the hidden queries. [Competition data](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data) · [Host clarification](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741404)

## Build the benchmark

Use Python 3.11–3.13, NumPy, PyArrow, and RDKit `2026.03.3` for competition-matched structure identity. Because `train.parquet`'s provided `inchikey14` differs from RDKit `2026.03.3` tautomer-canonical connectivity on ~1.8% of structures (collapsing 1,615 tautomer pairs across 277,566 unique SMILES), precompute the official RDKit key/fingerprint cache and desalted mass-sorted catalog once:

```sh
python casmi_eval.py prepare-train \
  --train data/train.parquet \
  --output data/train_canonical.parquet

python casmi_eval.py prepare-catalog \
  --input data/coconut_csv_lite-11-2026.zip \
  --output data/coconut_2026_11_canonical.parquet \
  --max-mass 1300

python casmi_eval.py build \
  --train data/train.parquet \
  --key-cache data/train_canonical.parquet \
  --catalog data/coconut_2026_11_canonical.parquet \
  --output splits/v3_official_coconut

python casmi_eval.py evaluate \
  --train data/train.parquet \
  --key-cache data/train_canonical.parquet \
  --catalog data/coconut_2026_11_canonical.parquet \
  --split splits/v3_official_coconut --fold dev \
  --candidate-output splits/v3_official_coconut/dev_candidates.parquet \
  --output splits/v3_official_coconut/dev_metrics.json

python casmi_eval.py export-train \
  --train data/train.parquet \
  --key-cache data/train_canonical.parquet \
  --split splits/v3_official_coconut \
  --output splits/v3_official_coconut/fit.parquet
```

Repeat `--catalog` to include additional snapshots (such as a prepared PubChem export), or run `casmi_eval.py attach-catalog --split splits/v3_official_coconut --catalog ...` later to refresh catalog coverage labels without altering the frozen `train`/`dev`/`test` molecule assignments. `manifest.json` (version 3) records all input hashes, near-duplicate counts, coverage counts, domain mix, and Tanimoto similarity bins.

The broad split targets 5% of all official tautomer-canonical keys in each holdout (`13,710` dev / `13,710` test / `246,775` fit structures). It preserves all query-eligible `enveda-np-examples` timsTOF natural products (`123` dev / `122` test), filters out republished cross-aggregator near-duplicate reference spectra (`cos >= 0.999` within `AGGREGATOR_SOURCES`, saved in `excluded_reference_row_ids.parquet`), annotates reference & catalog coverage first, and then assigns `dev` vs `test` balanced by `(novelty, domain, source, instrument, mass_band)`. It also records exact max Morgan-2048 Tanimoto similarity to `fit.parquet` (`max_tanimoto_to_train`, `tanimoto_bin`).

### Add PubChem

For a curated PubChem supplement, prepare the published PubChemLite CSV with `prepare-catalog --smiles-col SMILES`. Attach COCONUT and the resulting Parquet together to refresh the coverage labels. The original COCONUT split and the PubChemLite experiment are saved separately under `splits/v3_official_coconut` and `splits/v3_official_coconut_pubchemlite`. [PubChemLite release](https://zenodo.org/records/22143400)

Build canonical shards from the NCBI bulk snapshot for a batch of query precursors. [NCBI bulk files](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/)

```sh
python casmi_pubchem.py \
  --mass data/pubchem_2026_11_20/CID-Mass.gz \
  --smiles data/pubchem_2026_11_20/CID-SMILES.gz \
  --queries data/pubchem_2026_11_20/dev_focused_precursors.parquet \
  --output data/pubchem_2026_11_20/dev_focused_catalog_256 \
  --max-cids-per-precursor 256
```

Preparation uses measured precursor m/z and adducts to select PubChem CIDs and recalculates exact mass and the competition's canonical identity for single-fragment structures. The optional CID cap keeps the nearest records by mass before canonicalization. Omit it to process the full mass windows. `evaluate --additional-catalog PATH` adds each shard to retrieval while keeping the original novelty labels and panel IDs for a paired comparison. `casmi_baseline.py` accepts shards with repeated `--catalog`. Query-scoped or capped shards describe coverage for that run and cannot measure membership in all of PubChem.

To score any model's predictions (and optional candidate pool for `candidate_recall`), run:

```sh
python casmi_eval.py score \
  --split splits/v3_official_coconut --fold dev \
  --predictions predictions.csv \
  [--candidates candidates.csv] [--panel focused] [--subset all|in_catalog|timstof|timstof_np] \
  --key-cache data/train_canonical.parquet --output dev_model_metrics.json
```

The nested panel prioritizes timsTOF and the preferred sources, `enveda-np-examples` followed by `riken` by default. Use `--query-source` to change those preferences. The defaults allow up to 125 cases per novelty class per fold. Use `evaluate --panel focused` for a quicker run. The earlier, capped, source-held-out benchmark is available with `build --split-mode focused`.

| Panel | Exact structure has an independent spectrum in the frozen reference library | Exact structure in the attached catalog |
|---|---|---|
| Class 1 | Yes | Optional |
| Class 2 | No | Yes |
| Class 3 proxy | No | No |

`assignments.parquet` assigns one fold per connectivity key. Train exports exclude every dev and test structure across all acquisition sources, plus matching copies of their query peak lists. In the broad benchmark, reference search removes the chosen query source for each held-out structure and rounded duplicate query peak lists. In the earlier focused split, reference search removes the designated sources globally. Distinct library labels and peak lists are evidence for a separate reference acquisition. Republished near-duplicates can still require a provenance audit.

The query and label tables are separate. Use `dev_queries.parquet` during development and save `test_queries.parquet` for a locked comparison after selecting an approach. To score another model, pass its CSV with `molecule_id,smiles` to `casmi_eval.py score --split splits/v2 --fold dev --predictions predictions.csv`. Runs built with `--trust-keys` expect `connectivity_keys` in place of SMILES.

## Interpret dev and Kaggle together

The catalog retriever subtracts the adduct mass shift and keeps a structure if its monoisotopic mass matches any supported observation within 10 ppm of the observed precursor m/z. It deduplicates by connectivity and ranks each catalog candidate by its smallest mass error. The combined baseline places cosine-ranked spectral hits first and fills remaining slots with catalog hits. `--candidate-output` writes the full deduplicated candidate pool with SMILES, spectral cosine, and mass evidence so a reranker can score candidates before the 25-guess cutoff.

The evaluator reports candidate recall, candidate counts, top 1, recall@25, and MRR@25 by class, source, and instrument. It also breaks out Class 1 by reference adduct and instrument overlap. It compares the strict library search, mass-only catalog retrieval, and library results supplemented by catalog results. The equal-class mean is a diagnostic when all three classes have cases. The overall mean reflects the selected local mixture.

Pass `--formulas PATH` only when you have independently inferred neutral formulas. The CSV uses `molecule_id,formula` headers and accepts multiple Hill-notation formulas per molecule. Molecules without formula evidence use the mass window alone. Evaluation reports the mass-only and formula-filtered variants side by side. The optional candidate export uses the filtered catalog pool when this flag is supplied. Keep the training parquet's ground-truth `molecular_formula` out of this input.

Use the CPU prediction entry point to run the same retrieval stage on unlabeled queries.

```sh
python casmi_baseline.py data --catalog data/coconut_csv_lite-11-2026.zip \
  --candidate-output candidates.parquet --output submission.csv
```

Compare changes on the same fixed molecules and inspect gains by class and on timsTOF. Class counts and domain mix matter when comparing across splits. Treat the downloadable sample as a format check. The public leaderboard is a separate measurement and the hidden class mix can produce a different absolute score.

For controlled stress tests, repeat the dev command with `--simulate catalog-only` to remove exact library references from catalog-covered Class 1 queries, or `--simulate catalog-absent` to remove their exact catalog entries too. These diagnostics preserve measured query spectra and the remaining analogs. Record their metrics separately from the primary panels.

## Library confidence experiment

Freeze the catalog and library candidates. Compare unconditional library-first ordering, catalog-first ordering on the same candidates, and a rule that promotes a single library hit when its cosine and lead over the runner-up meet thresholds. Catalog-only ordering provides the historical control.

Prepare the COCONUT and PubChemLite snapshots with the commands above. Pass both catalogs in the same order to `build`, `evaluate` and `calibrate`. Export `fit.parquet` and `dev_candidates.parquet` before running:

```sh
casmi_split=splits/v3_official_coconut_pubchemlite
mkdir -p runs
python casmi_library_gate.py calibrate --split "$casmi_split" \
  --catalog data/coconut_2026_11_canonical.parquet \
  --catalog data/pubchem_2026_11_20/PubChemLite_exposomics_20260828_canonical.parquet \
  --key-cache data/train_canonical.parquet --output runs/library_thresholds.json
python casmi_library_gate.py evaluate --split "$casmi_split" \
  --candidates "$casmi_split/dev_candidates.parquet" \
  --thresholds runs/library_thresholds.json --output runs/library_gate_metrics.json
```

The fit-only calibration selected a minimum cosine of **0.55** and a minimum margin of **0.00** on 301 molecules and their truth-masked counterparts. The score cutoff is a heuristic learned from a mix of source holdouts and simulated catalog-only queries.

| Development slice | Library first | Confidence gated | Paired MRR@25 change (95% CI) |
| --- | ---: | ---: | ---: |
| Class 1 (1,278) | 0.7835 | 0.7363 | −0.0472 [−0.0574, −0.0371] |
| Class 2 (1,003) | 0.1178 | 0.2049 | +0.0871 [+0.0760, +0.0982] |

The gate recovered Class 2 rank while losing 14 net Class 1 first-place hits. The standalone experiment leaves submission ranking at the existing baseline.

## External benchmark for generation

Revisit this when the pipeline can generate structures. Freeze the reference library, catalogs, and model data to before an independent discovery study disclosed its compounds. Use date and provenance records to verify that the structure was absent before disclosure. Score measured spectra with independently confirmed structures, preferably NMR-confirmed. Audit overlap with any pretrained model where its data is known, and report uncertainty where it is unknown. [BUDDY discovery examples](https://doi.org/10.1038/s41592-023-01850-x)

For an external Class 2 panel, start with CASMI 2022. The organizers described its 500 measured compounds as outside public spectral libraries at the time. Audit each exact connectivity against the current frozen reference and catalog snapshots before labeling it Class 2. Track instrument shift separately from the main CASMI 2026 dev set. [CASMI 2022 raw data and provenance](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000089239)

## Initial verification

The measured smoke run used 10 MassBank records across four structures, with BAFG as the query source and RIKEN and CASMI 2016 as the reference sources. The small catalog was built from the four public structure records. Dev had two molecules and three spectra. Local test had two molecules and four spectra. These are pipeline checks on QTOF queries with a tiny catalog. [MassBank source records](https://github.com/MassBank/MassBank-data)

| Measured smoke check | Molecules | Library MRR@25 | Library plus catalog MRR@25 |
|---|---:|---:|---:|
| Dev | 2 | 0.50 | 1.00 |
| Local test | 2 | 0.25 | 0.75 |

The synthetic integration test exercises all three cases, both explicit masking modes, duplicate removal, the fit export, and scorer rank handling. Run it with `python -m unittest -v test_casmi_eval`.

The Kaggle library-only control used the parquet's supplied connectivity keys. The focused source holdout scored 250 molecules in each fold, with 125 Class 1 cases per fold. Class 1 MRR@25 was 0.891 on dev and 0.913 on local test. The broad split with the same identity mode contains 13,790 molecules per fold. The catalog comparison requires a frozen COCONUT snapshot.
