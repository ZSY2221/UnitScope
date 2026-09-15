# Dataset provenance

Source files were checked on 2026-07-14. The learning experiments use writer,
synthetic-patient, or account identifiers. Their roles are listed below.

## Core task 1: writer-grouped Federated EMNIST

- Source: TensorFlow Federated public dataset
- URL: `https://storage.googleapis.com/tff-datasets-public/emnist_all.sqlite.lzma`
- Upstream description: LEAF-derived Extended MNIST grouped by writer
- Compressed bytes: `170507172`
- Compressed SHA-256: `c3aaf91b4cf4e69d53d81627ad119eccdc878ff906f68e66063ff10f67faad48`
- Decompressed bytes: `4708990976`
- Decompressed SHA-256: `239b48fb3b3d69e70406d18a0605c821d4db3c803443c64788736bcb9f79ac56`
- Public metadata: 3,400 writers, 671,585 training examples, and 77,483 test
  examples for the all-character task.
- Experiment role: writer-level learning with simulated silos. Writers are
  split before silo assignment, and records are not duplicated.
- Scope: the writer groups are real; the silo topology is simulated.

## Core task 2: Synthea 10k COVID-19 CSV population

- Source: Synthetic Health official sample-data repository
- URL: `https://github.com/synthetichealth/synthea-sample-data/raw/refs/heads/main/downloads/10k_synthea_covid19_csv.zip`
- Compressed bytes: `56851927`
- SHA-256: `559757dc849f4361a328f456d2c0a20c6df72419068321c753c6be787161e937`
- Task: use encounters in the 365 days before 2020-01-01 to predict
  whether the same patient has an inpatient encounter in the following 365
  days.
- Adapter output: 25,428 eligible historical encounter records, 8,536
  patients, and 1,825 positive patients. Under the frozen six-silo organization
  hash, 37.24% of patients reach at least two silos, with a mean of 1.41 silos
  and a maximum of five.
- Experiment role: patient-level learning and the health-card evidence view.
- Scope: Synthea supplies synthetic patients and encounters. Results do not
  establish clinical validity or real-patient prevalence.

## Supplementary task: Sentiment140

- Source: Stanford Sentiment140 archive referenced by LEAF
- URL: `http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip`
- Compressed bytes: `81363704`
- SHA-256: `004a3772c8a7ff9bbfeb875880f47f0679d93fc63e5cf9cff72d54a8a6162e57`
- Adapter output: 1,600,000 records, 659,775 accounts, 254,498 accounts with at
  least two records, and 1,194,723 records owned by those multi-record accounts.
- Experiment role: account-level text classification with simulated silos.
- Scope: account groups are used as contribution units; they are not verified
  natural persons.

## Supplementary task: MovieLens 1M

- Source: GroupLens stable MovieLens 1M release
- URL: `https://files.grouplens.org/datasets/movielens/ml-1m.zip`
- Compressed bytes: `5917549`
- Official MD5: `c4d9eecfca2ab87c1945afe126590906`
- Compressed SHA-256: `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`
- Public metadata: 1,000,209 ratings from 6,040 anonymized accounts on
  3,706 movies.
- Experiment role: account-level recommendation with simulated silos. Accounts
  are split before silo assignment, and ratings are not duplicated.
- Scope: account identifiers are not verified natural persons, and the silo
  topology is not a real multi-platform deployment.
- Redistribution: the artifact stores the official downloader, source URLs,
  hashes, and preprocessing code. It does not redistribute GroupLens raw data.

## Identity/WUR-only datasets

FEBRL and NCVR are used for identity quality and WUR only. They are not part of
the learning comparison. Ground-truth entity identifiers are used only for
evaluation and are excluded from matcher features.

## Integrity

`download_manifest.json` records the size and SHA-256 hash of each downloaded
or extracted file. The download script enforces a 5 GiB per-file limit.
