# Data

Raw data are not redistributed in this repository. Run:

```text
python reproduce.py data --dataset all
```

| Dataset | Paper role | Prepared path |
|---|---|---|
| FEMNIST | main writer-level task, non-IID study, capacity grid | `data/raw/femnist/emnist_all.sqlite` |
| Synthea 10k COVID-19 | main synthetic-patient task | `data/raw/synthea/10k_synthea_covid19_csv/` |
| Sentiment140 | supplementary account-level task | `data/raw/sent140/training.1600000.processed.noemoticon.csv` |
| MovieLens 1M | supplementary balanced account-level task | `data/raw/movielens1m/ml-1m/` |
| FEBRL1-3 | linkage and WUR boundary study | `data/processed/febrl/` |

The downloader checks the archive SHA-256 values in
`DATASET_SOURCE_PROVENANCE.json`. It also checks the decompressed FEMNIST
database hash. `data/raw/download_manifest.json` records the files actually
used on the local machine.

FEBRL1-3 are supplied by the pinned `recordlinkage` package. The `all` command
materializes them as six-silo inputs. To prepare only FEBRL, run:

```text
python reproduce.py data --dataset febrl
```

The NCVR experiment uses the North Carolina State Board of Elections snapshot
dated 2026-07-13. That historical object is not redistributed, and the public
URL now points to the current statewide file. `scripts/download_ncvr_snapshots.ps1`
checks the recorded byte length, but a current download is not a replacement
for the frozen paper snapshot. The aggregate paper results are retained under
`results/mechanism_studies/ncvr_boundary/`.

The frozen ZIP is 517,980,796 bytes with SHA-256
`dfe78db2658339766a4c688c60a9db7a762ba2cd2a00d54265787bad7a7c9195`.
This value was recorded by the original statewide run and rechecked against the
locally retained 2026-07-13 snapshot during release preparation.

The earlier county-scale run used the board's public `ncvoter1.zip` object from
`https://s3.amazonaws.com/dl.ncsbe.gov/data/ncvoter1.zip`. That mutable snapshot
is also not redistributed. The frozen county ZIP has SHA-256
`b36c62507d04f3c21c6c4985ef1ea3aac45ad65d8ba70bdb2832fbe58125db6d`.

Detailed source URLs, hashes, identity semantics, and preprocessing roles are
in `DATASET_PROVENANCE_AND_ROLES.md`.
