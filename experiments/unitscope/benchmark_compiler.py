from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import pandas as pd

from .audit import audit_plan_transition
from .compiler import compile_plan, validate_plan_with_oracle
from .evidence import EvidencePolicy, generate_evidence_view
from .manifest import atomic_write_json
from .model import GroundTruth, MethodId, PublicConfig
from .synthetic import make_vector_population


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark UnitScope compilation and adjacent-plan audit"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--users", type=int, nargs="+", default=[1000, 10_000, 100_000, 1_000_000]
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=31001)
    args = parser.parse_args()
    if args.dry_run:
        args.users = [20, 100]
    rows = []
    for n_users in args.users:
        config = PublicConfig(
            "compiler-scaling",
            "frozen",
            1.0,
            2,
            6,
            0.5,
            float(n_users),
            1.0,
            1,
            1e-5,
            6,
        )
        population = make_vector_population(
            n_users=n_users,
            n_silos=6,
            dimension=8,
            overlap_probability=0.5,
            gradient_correlation=0.5,
            norm=0.6,
            seed=args.seed,
            records_per_silo=2,
        )
        tracemalloc.start()
        started = time.perf_counter()
        evidence = generate_evidence_view(
            population.records,
            population.truth,
            config,
            EvidencePolicy(0.5, 0.5, handles_per_user=2),
            args.seed + 1,
        )
        evidence_seconds = time.perf_counter() - started
        started = time.perf_counter()
        plan = compile_plan(
            MethodId.UNITSCOPE_TUNED,
            population.records,
            evidence.evidence,
            evidence.registry,
            config,
        )
        compile_seconds = time.perf_counter() - started
        validation = validate_plan_with_oracle(
            plan, population.records, population.truth, config
        )
        if not validation.valid:
            raise RuntimeError(validation.errors)
        truth = population.truth.as_dict()
        removed_user = sorted(set(truth.values()))[0]
        remaining = tuple(
            record
            for record in population.records
            if truth[record.record_id] != removed_user
        )
        remaining_truth = GroundTruth(
            tuple((record.record_id, truth[record.record_id]) for record in remaining)
        )
        started = time.perf_counter()
        adjacent_plan = compile_plan(
            MethodId.UNITSCOPE_TUNED,
            remaining,
            evidence.evidence,
            evidence.registry,
            config,
        )
        adjacent_validation = validate_plan_with_oracle(
            adjacent_plan, remaining, remaining_truth, config
        )
        if not adjacent_validation.valid:
            raise RuntimeError(adjacent_validation.errors)
        witness = audit_plan_transition(plan, adjacent_plan)
        audit_seconds = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        row = {
            "users": n_users,
            "records": len(population.records),
            "plan_units": len(plan.units),
            "partial_handle_units": sum(
                unit.kind.value == "partial-handle" for unit in plan.units
            ),
            "evidence_seconds": evidence_seconds,
            "compile_seconds": compile_seconds,
            "adjacent_recompile_and_audit_seconds": audit_seconds,
            "peak_traced_memory_mib": peak / (1024 * 1024),
            "deletion_witness_wur_over_c": witness.value / config.C,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / "compiler_scaling.csv", index=False)
    atomic_write_json(
        args.output / "compiler_scaling_manifest.json",
        {
            "dry_run": args.dry_run,
            "users": args.users,
            "seed": args.seed,
            "population": {
                "n_silos": 6,
                "dimension": 8,
                "overlap_probability": 0.5,
                "gradient_correlation": 0.5,
                "norm": 0.6,
                "records_per_silo": 2,
                "mixed_local_slots": False,
            },
            "evidence_policy": {
                "q_user": 0.5,
                "q_record_given_handle": 0.5,
                "handles_per_user": 2,
                "seed_offset": 1,
            },
            "public_config": {
                "task_domain": "compiler-scaling",
                "epoch": "frozen",
                "clip_norm": 1.0,
                "max_handle_groups": 2,
                "max_fallback_slots": 6,
                "lambda": 0.5,
                "public_normalizer": "n_users",
                "noise_multiplier": 1.0,
                "rounds": 1,
                "delta": 1e-5,
                "n_silos": 6,
            },
            "method": MethodId.UNITSCOPE_TUNED.value,
            "warning": "Oracle validation is an experiment assertion and is excluded from compile_seconds.",
        },
    )


if __name__ == "__main__":
    main()
