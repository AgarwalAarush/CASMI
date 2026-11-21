# CASMI input provenance

The repository contains first-party code. To reproduce the pipeline, obtain the inputs from their original sources and prepare catalog indexes with `casmi_eval.py prepare-catalog`. The code package contains only the runner, CPU baseline, catalog search and dependency pins.

| Input | Source and version | Use |
| --- | --- | --- |
| `train.parquet`, `test.parquet` | Enveda CASMI 2026 competition data. Record file hashes when downloading. | CPU spectral library and queries. The official train file also supplies the local evaluation splits. |
| COCONUT CSV Lite | [November 2026 snapshot](https://coconut.naturalproducts.net/download), `coconut_csv_lite-11-2026.zip` | Optional mass-indexed structures. |
| PubChemLite for Exposomics | [Version 3.2.0](https://zenodo.org/records/22143400), `PubChemLite_exposomics_20260828.csv` | Optional mass-indexed structures. |

Review each source’s terms for the intended submission use and record the exact download, release, terms and SHA-256 with that submission. The documented local metrics are from prepared COCONUT and PubChemLite catalogs. The published historical Kaggle score used the competition data for library search.

[Competition data](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data) · [COCONUT](https://coconut.naturalproducts.net/) · [Local preparation](CASMI_EVALUATION.md)

### SHA-256 for the local benchmark inputs

```text
9423f90ec08b941a9646b535bf0606183e86ff0c5a297978f5a70693d41bcd98  train.parquet
81f3edc1731f01da289fe6b84d162667863e331d52041a781565e1af2eb79939  coconut_csv_lite-11-2026.zip
ca8b89045f57b50702ef917d6a7713453972034e5b92fd33e59ec1d03ff7b1ce  PubChemLite_exposomics_20260828.csv
```
