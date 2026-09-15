# Results

The directories in this folder correspond to the experiments in the paper and
supplementary material.

- `main/`: fixed-allocation FEMNIST and Synthea results.
- `femnist_nonprivate_tuning/`: all 60 validation runs, the 12-cell grid, and
  the five selected test runs.
- `reference_methods/`: complete-identity, ULDP-FL, Group-2, and record-level
  reference results.
- `learning_extensions/`: Sent140, non-IID, and privacy-budget studies.
- `movielens_task/`: balanced MovieLens study.
- `residual_audit/`, `adaptive_allocation/`, `allocation_tradeoff/`, and
  `shifted_residual_signal/`: residual-path and allocation analyses.
- `contract_parameter_sweep/`, `b_aware_gate/`, `failure_injection/`, and
  `db_workload/`: contract and operational studies.
- `linkage_boundary/`: FEBRL and PER boundary summaries and deletion witnesses.
- `mechanism_studies/`: evidence coverage, gradient geometry, compiler scaling,
  protected execution, lifecycle, and NCVR results.

The main, reference, extension, and MovieLens learning families include a
`registry.json` and a `runs/` directory. The FEMNIST non-private tuning family
retains its validation and test runs without a separate registry. Run
`python reproduce.py analyze` from the repository root to rebuild the learning
summaries. Raw datasets are not stored here.

Frozen run files are copied byte-for-byte. Internal identifiers in those files
keep the names used when the experiments were run.
