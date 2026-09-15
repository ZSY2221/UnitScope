# UnitScope

This repository contains the UnitScope compiler, experiment drivers, released
run directories, and the tables used in the paper and supplementary material.

## Setup

Python 3.12 is used for the release checks. Create an environment and install
the pinned packages:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Linux or macOS, use the environment's `bin/python` path:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

This installs the CPU build of PyTorch. It is sufficient for the smoke checks,
compiler examples, and CPU analysis commands. To run the multi-hour `main`,
`references`, `extensions`, or `nonprivate` jobs with `--device cuda`, replace
it with a CUDA build. For example, PyTorch 2.8.0 with CUDA 12.8 can be installed
with:

```text
python -m pip install --force-reinstall --no-deps torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

The CUDA build is not installed by `requirements.txt`. Choose the PyTorch
wheel index that matches the local driver and CUDA environment. Use the
environment's Python executable in the commands below.

Local environments, caches, downloaded or prepared data, and generated outputs
are not version controlled. The excluded directories are `.venv/`,
`__pycache__/`, `.pytest_cache/`, `data/raw/`, `data/processed/`, `reproduced/`,
and `figures/`. The same directories are excluded from `HASHES.md`.

## Data

The raw datasets are downloaded from their original hosts and stored under the
ignored `data/raw/` directory:

```text
python reproduce.py data --dataset all
```

The command verifies every archive, decompresses FEMNIST, and extracts
Synthea, Sent140, and MovieLens into the paths expected by the trainers.
FEMNIST occupies about 4.7 GB after decompression. Allow at least 8 GB for the
data and more space for training outputs. See `data/README.md` for dataset
roles, paths, and the FEBRL and NCVR notes.

## Run

Inspect the run counts without starting a job:

```text
python reproduce.py plan
```

Run the release checks:

```text
python reproduce.py smoke
```

Recompute the main, reference, extension, MovieLens, and non-private summaries
from the released manifests, JSONL files, clusters, and trajectories:

```text
python reproduce.py analyze
```

The main two-level bootstrap is the slow part of this command and can take
more than ten minutes on a laptop.

### Learning experiments

Run the 40-cell FEMNIST and Synthea experiment and rebuild its registry and
tables:

```text
python reproduce.py main --device cuda
```

Run the supplementary learning experiments for epsilon 8, non-IID FEMNIST,
Sent140, and balanced MovieLens:

```text
python reproduce.py extensions --device cuda
```

After the main run, reproduce Complete-ID, ULDP-AVG-w, Group-2, and the
record-level reference on the same 40 paired configurations:

```text
python reproduce.py references --device cuda
```

Run the 60-cell FEMNIST validation grid followed by the five selected test
runs:

```text
python reproduce.py nonprivate --device cuda
```

These are multi-hour GPU jobs. `main` must finish before `references`; the
other two commands are independent after their datasets are prepared. Outputs
go to `reproduced/main/`, `reproduced/references/`,
`reproduced/extensions/`, and `reproduced/femnist_nonprivate/`. Completed run
directories are reused.

### Allocation studies

The released lambda, evidence, and adaptive directories contain the original
per-run files. Recompute their tables with:

```text
python reproduce.py lambda-grid
python reproduce.py evidence-grid
python reproduce.py adaptive
```

These CPU analyses take a few seconds and write to
`reproduced/lambda_frontier/`, `reproduced/evidence_grid/`, and
`reproduced/adaptive_allocation/`. They do not need raw datasets or a new main
run. The adaptive command analyzes the released 90-configuration confirmation;
it does not rerun its multi-hour GPU training stage.

Run the vector mechanism studies from scratch:

```text
python -m experiments.unitscope.run_b_aware_gate --output reproduced/b_aware_gate
python -m experiments.unitscope.run_contract_parameter_sweep --output reproduced/contract_parameter_sweep
python -m experiments.unitscope.run_gradient_geometry --output reproduced/gradient_geometry
```

They are independent CPU jobs. B-aware and the contract sweep usually finish
within a few minutes; gradient geometry can take several minutes. Each command
writes raw rows, summaries, and a manifest under its output directory.

### Operational studies

Prepare Synthea before running the database workload:

```text
python reproduce.py data --dataset synthea
python -m experiments.unitscope.run_db_workload --data data/raw/synthea/10k_synthea_covid19_csv --declaration declarations/db_workload.md --output reproduced/db_workload
```

The workload takes about a minute and writes 90 releases plus its audit under
`reproduced/db_workload/analysis/`. The remaining operational studies are
independent CPU jobs:

```text
python -m experiments.unitscope.run_failure_injection --output reproduced/failure_injection
python -m experiments.unitscope.run_dynamic_lifecycle --output reproduced/dynamic_lifecycle
python -m experiments.unitscope.benchmark_compiler --output reproduced/compiler_benchmark --users 1000 10000 100000 1000000 --seed 31001
python -m experiments.unitscope.benchmark_mpc_backend --output reproduced/mpc_backend
```

Failure injection and lifecycle finish in seconds. The compiler command above
reproduces the frozen four-point scale with the recorded seed. Its one-million
user point needs about 3 GB of memory and can take several minutes. The MPC
benchmark usually takes a few minutes.

### Linkage studies

The NCVR snapshot is not redistributed. If the exact snapshot described in
`data/README.md` is available locally, run:

```text
python -m experiments.unitscope.run_ncvr_entity_resolution --zip path/to/ncvoter_2026-07-13.zip --maximum-people 250000 --selection-modulus 16 --scales 50000 100000 250000 --output reproduced/ncvr_statewide
```

The statewide run can take tens of minutes. Its manifest records the ZIP hash.
Do not use the current NCVR download as a substitute for the frozen snapshot.

PER uses its upstream implementation and a separate Python 3.10 environment
with `pyjedai==0.2.0`. Check out commit
`09eeb4a27dacf41eb6db11d9ae898b4ac3e17bbd`, then run:

```text
python adapters/per2025/prepare_febrl.py --output reproduced/per_input
python adapters/per2025/run_per_febrl.py --input reproduced/per_input --official-repo path/to/PER-Design-Space-Exploration --output reproduced/per_febrl
```

The full FEBRL1-3 run takes a few minutes. `adapters/README.md` records the
upstream repository and commit. `pyjedai` is intentionally not in the main
requirements because its Windows Python 3.12 dependency chain needs a local C
compiler.

### Figures and combined run

Rebuild figures from the frozen tables with:

```text
python reproduce.py figures
```

`python reproduce.py all --device cuda` runs data preparation, the main,
reference, extension, MovieLens, and non-private capacity experiments. It then
plots those new learning results and uses frozen tables for linkage, allocation,
mechanism, and operational studies. It does not rerun those latter studies.
Full training is a multi-hour GPU job.

Generated outputs are written to `reproduced/`. The `figures` command writes
to `figures/`; `all` writes figures to `reproduced/figures/`. Neither output
directory is part of the release. Published tables remain under `results/`.

#### Paper figure toolchain

`plotting/plot_paper_figures.py` consolidates the final paper-local plotting
code and rebuilds the paper's data-driven figures from the released CSV tables.
Run it through `python reproduce.py figures`, or choose explicit inputs and an
output directory with:

```text
python plotting/plot_paper_figures.py --results-root results --figures-root reproduced/figures
```

The lambda-frontier and evidence-coverage panels use the final parameters from
the paper-local plotting tools. The contract sweep retains the original
three-panel canvas and common panel crop, and the non-IID panels use the
archived trajectory analyzer's mean and 95% SEM rendering. The three conceptual
figures `Figure1a_naive_total_2C.pdf`, `Figure1b_unitscope_contract.pdf`, and
`Figure2_unitscope_workflow.pdf` are drawn manually. Their editable sources are
the matching files under the paper's `visio-fig/` directory, and this script
does not generate them.

PDF files contain backend metadata and are not expected to be byte-identical.
To verify a rebuild, rasterize the published and regenerated PDFs at 200 dpi
with `pdftoppm -r 200 -png`, then compare image dimensions and pixels. Small
differences confined to antialiased glyph and line edges are renderer effects;
data geometry, labels, and panel framing should otherwise agree.

### Optional paper-code consistency check

The repository does not include the paper manuscript. Authors, or readers who
already have the manuscript text, can run an additional check that compares
reported values with the released results:

```text
python reproduce.py validity --paper path/to/main.tex
```

This optional check also needs prepared FEMNIST and Synthea data. It is not an
experiment or a required part of the reproduction workflow. All other commands
and repository functions work without `main.tex`; readers reproducing the
experiments can skip this section. Passing the same `--paper` option to `all`
runs this check after the learning experiments.

Floating-point results can vary slightly across CUDA and library versions.
Seeds, splits, schedules, data hashes, and per-run configuration hashes are
recorded so that those differences can be audited. Recomputed hierarchical
bootstrap confidence-interval bounds can differ by about `2e-4` across numerical
libraries and hardware. This does not affect the reported conclusions.

## License

UnitScope code is released under the MIT License. Dataset licenses and terms
remain with their respective owners.
