# Database workload declaration

This file fixes the database experiment before execution. Its payload hash is
checked by `run_db_workload.py`.

## Data and query

- Data: official Synthea 10k COVID-19 CSV sample.
- Source SHA-256: `559757dc849f4361a328f456d2c0a20c6df72419068321c753c6be787161e937`.
- Task: `build_synthea_future_inpatient_task` with seed 20260714, six silos,
  index date 2020-01-01, a 365-day history, and a 365-day future window.
- Evidence: regional health-card handles issued on 2019-07-01 for `silo-0`
  through `silo-3`, with one handle domain, H=1, B=6, C=1, and epoch
  `db-workload-v1`.
- Query: count distinct patients with a positive future-inpatient label among
  records covered by valid health-card handles.

Oracle patient IDs are used only to check handle purity and the distinct-count
reference. The UnitScope query groups records by validated handle.

Before a result is accepted, the implementation checks that claims are pure,
each query record belongs to one claim, each claimed patient maps to one
UnitScope unit, the deterministic UnitScope and Naive counts match the oracle
count, and a multi-silo fixture is counted once.

## Routes

1. **Stable-Local:** de-duplicate within each silo and sum the silo counts. Its
   sensitivity is C. A patient present in several silos is counted several
   times.
2. **Naive-2C:** compute the person-level count with both handle and local
   contribution paths. Its sensitivity is 2C.
3. **UnitScope:** group by validated Partial-Handle units. Its sensitivity is C.

Each route adds Laplace noise to one scalar release. For sensitivity Delta, the
release is `count + (Delta/epsilon) * Z`, where Z is a standard Laplace draw.
The output is not clipped. Epsilon is 1, 4, or 16.

## Runs and analysis

- Seeds: 42001 through 42010.
- The standard Laplace draw for a seed is shared across routes and epsilon
  values.
- Runs: 3 routes x 3 epsilon values x 10 seeds = 90 releases.
- Primary metric: squared error from the oracle count.
- Secondary metric: absolute relative error.
- Intervals: paired percentile bootstrap with 20,000 resamples and seed
  20270724.
- Stable-Local duplicate bias is also reported before noise.

UnitScope should have lower error than Naive-2C because the deterministic counts
match while the Laplace scales are C/epsilon and 2C/epsilon. Stable-Local should
show positive duplicate-count bias. All 90 releases are retained.

## Integrity

The payload hash covers the UTF-8 bytes before the line below.

Declaration payload SHA-256: `56664307c8c0208c18c8fd1971f5f80640a15f7a0a86ead019b6c77e2471e9c2`
