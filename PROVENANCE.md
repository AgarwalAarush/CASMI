# CASMI input provenance

The repository contains first-party code. To reproduce the pipeline, obtain the inputs from their original sources and prepare catalog indexes with `casmi_eval.py prepare-catalog`. The code package contains only the runner, CPU baseline, catalog search and dependency pins.

| Input | Source and version | Use |
| --- | --- | --- |
| `train.parquet`, `test.parquet` | Enveda CASMI 2026 competition data. Record file hashes when downloading. | CPU spectral library and queries. The official train file also supplies the local evaluation splits. |
| COCONUT CSV Lite | COCONUT November 2026 snapshot, `coconut_csv_lite-11-2026.zip` | Optional mass-indexed structures. |
| PubChemLite for Exposomics | 2026-08-28 snapshot, `PubChemLite_exposomics_20260828.csv` | Optional mass-indexed structures. |

Review each source’s terms for the intended submission use and record the exact download, release, terms and SHA-256 with that submission. The documented local metrics are from prepared COCONUT and PubChemLite catalogs. The published historical Kaggle score used the competition data for library search.

[Competition data](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data) · [COCONUT](https://coconut.naturalproducts.net/) · [Local preparation](CASMI_EVALUATION.md)
