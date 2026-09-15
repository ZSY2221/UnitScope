from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import numpy as np
import pandas as pd

from .baselines import PRIMARY_METHODS
from .compiler import compile_plan, validate_plan_with_oracle, with_oracle_validation
from .evidence import EvidencePolicy, generate_evidence_view
from .executor import execute_release
from .manifest import atomic_write_json, build_method_fields
from .model import MethodId, PublicConfig, UnitKind
from .synthetic import make_vector_population


MAIN_METHODS = (
    MethodId.STABLE_LOCAL_FALLBACK,
    MethodId.HANDLE_ONLY,
    MethodId.UNITSCOPE_EQUAL,
    MethodId.UNITSCOPE_TUNED,
    MethodId.NAIVE_RECALIBRATED_2C,
    MethodId.FULL_IDENTITY,
    MethodId.NON_PRIVATE,
)


def _full_identity_evidence(generated, population, config, seed):
    complete_policy = EvidencePolicy(0.0, 0.0, complete_user_probability=1.0)
    return generate_evidence_view(
        population.records, population.truth, config, complete_policy, seed
    )


def _reported_lambda(method: MethodId, config: PublicConfig) -> float | None:
    if method == MethodId.STABLE_LOCAL_FALLBACK:
        return 0.0
    if method == MethodId.HANDLE_ONLY:
        return 1.0
    if method == MethodId.UNITSCOPE_EQUAL:
        return config.equal_lambda
    if method == MethodId.UNITSCOPE_TUNED:
        return config.lambda_
    return None


def run_configuration(
    config: PublicConfig,
    *,
    n_users: int,
    dimension: int,
    overlap_probability: float,
    gradient_correlation: float,
    vector_norm: float,
    q_user: float,
    q_record_given_handle: float,
    handles_per_user: int,
    records_per_silo: int,
    seed: int,
    mixed_local_slots: bool = False,
) -> pd.DataFrame:
    population = make_vector_population(
        n_users=n_users,
        n_silos=config.n_silos,
        dimension=dimension,
        overlap_probability=overlap_probability,
        gradient_correlation=gradient_correlation,
        norm=vector_norm,
        seed=seed,
        mixed_local_slots=mixed_local_slots,
        records_per_silo=records_per_silo,
    )
    evidence_policy = EvidencePolicy(
        q_user, q_record_given_handle, handles_per_user=handles_per_user
    )
    generated = generate_evidence_view(
        population.records, population.truth, config, evidence_policy, seed + 300
    )
    complete = _full_identity_evidence(generated, population, config, seed + 400)
    noise = np.random.default_rng(seed + 500).normal(size=dimension)
    rows: List[Dict[str, object]] = []
    releases = {}
    for method in MAIN_METHODS + (MethodId.NAIVE_CLAIMED_C,):
        started = perf_counter()
        if method == MethodId.FULL_IDENTITY:
            method_evidence, registry = complete.evidence, complete.registry
        else:
            method_evidence, registry = generated.evidence, generated.registry
        plan = compile_plan(
            method, population.records, method_evidence, registry, config
        )
        validation = validate_plan_with_oracle(
            plan, population.records, population.truth, config
        )
        plan = with_oracle_validation(plan, validation)
        if not validation.valid:
            raise RuntimeError(
                f"plan {method.value} failed oracle validation: {validation.errors}"
            )
        release = execute_release(
            population.records,
            population.vectors,
            plan,
            config,
            None if method == MethodId.NON_PRIVATE else noise,
            allow_unsafe_diagnostic=method == MethodId.NAIVE_CLAIMED_C,
        )
        releases[method] = release
        partial_records = sum(
            len(unit.record_ids)
            for unit in plan.units
            if unit.kind == UnitKind.PARTIAL_HANDLE
        )
        residual_records = sum(
            len(unit.record_ids)
            for unit in plan.units
            if unit.kind in (UnitKind.LOCAL_ATOM, UnitKind.LOCAL_MIXED)
        )
        used_records = len(population.records) - release.diagnostics.dropped_records
        partial_handle_units = sum(
            unit.kind == UnitKind.PARTIAL_HANDLE for unit in plan.units
        )
        local_atom_units = sum(unit.kind == UnitKind.LOCAL_ATOM for unit in plan.units)
        local_mixed_units = sum(
            unit.kind == UnitKind.LOCAL_MIXED for unit in plan.units
        )
        spec = PRIMARY_METHODS[method]
        rows.append(
            {
                **build_method_fields(spec),
                "seed": seed,
                "config_hash": config.config_hash,
                "lambda": _reported_lambda(method, config),
                "H": config.H,
                "B": config.B,
                "C": config.C,
                "q_user": q_user,
                "q_record_given_handle": q_record_given_handle,
                "realized_q_user": generated.realized_handle_user_fraction,
                "realized_q_record_given_handle": generated.realized_record_coverage_given_handle,
                "overlap_probability": overlap_probability,
                "realized_overlap": population.realized_overlap,
                "gradient_correlation": gradient_correlation,
                "mixed_local_slots": mixed_local_slots,
                "realized_pairwise_cosine_mean": population.realized_pairwise_cosine_mean,
                "realized_pairwise_cosine_p10": population.realized_pairwise_cosine_p10,
                "realized_pairwise_cosine_p50": population.realized_pairwise_cosine_p50,
                "realized_pairwise_cosine_p90": population.realized_pairwise_cosine_p90,
                "certified_sensitivity": None
                if plan.certified_sensitivity is None
                else plan.certified_sensitivity.value,
                "structural_sensitivity_bound": plan.structural_sensitivity_bound,
                "certificate_type": plan.certificate_type.value,
                "noise_sensitivity": plan.noise_sensitivity,
                "effective_record_utilization": used_records / len(population.records),
                "partial_handle_record_fraction": partial_records
                / len(population.records),
                "residual_record_fraction": residual_records / len(population.records),
                "partial_handle_units": partial_handle_units,
                "local_atom_units": local_atom_units,
                "local_mixed_units": local_mixed_units,
                "active_units": len(plan.active_units),
                "clip_fraction": release.diagnostics.clip_fraction,
                "aggregate_snr": release.diagnostics.aggregate_snr,
                "wall_time_seconds": perf_counter() - started,
            }
        )
    oracle = releases[MethodId.FULL_IDENTITY]
    for row in rows:
        method = MethodId(row["method"])
        difference = releases[method].deterministic_value - oracle.deterministic_value
        noisy_difference = releases[method].value - oracle.deterministic_value
        row["deterministic_distortion"] = float(np.dot(difference, difference))
        row["noisy_aggregate_mse"] = float(np.dot(noisy_difference, noisy_difference))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UnitScope 128-dimensional mechanism experiment"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-users", type=int, default=256)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--n-silos", type=int, default=6)
    parser.add_argument("--H", type=int, default=1)
    parser.add_argument("--B", type=int, default=6)
    parser.add_argument("--lambda-value", type=float, default=0.5)
    parser.add_argument("--q-user", type=float, default=0.75)
    parser.add_argument("--q-record-given-handle", type=float, default=0.5)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--correlation", type=float, default=0.5)
    parser.add_argument("--records-per-silo", type=int, default=2)
    parser.add_argument("--handles-per-user", type=int)
    args = parser.parse_args()
    if args.dry_run:
        args.n_users = min(args.n_users, 12)
        args.dimension = min(args.dimension, 8)
        args.seeds = args.seeds[:1]
    config = PublicConfig(
        task_domain="vector-mechanism",
        epoch="frozen-v1",
        clip_norm=1.0,
        max_handle_groups=args.H,
        max_fallback_slots=args.B,
        lambda_=args.lambda_value,
        public_normalizer=float(args.n_users),
        noise_multiplier=1.0,
        rounds=1,
        delta=1e-6,
        n_silos=args.n_silos,
    )
    frames = [
        run_configuration(
            config,
            n_users=args.n_users,
            dimension=args.dimension,
            overlap_probability=args.overlap,
            gradient_correlation=args.correlation,
            vector_norm=0.6,
            q_user=args.q_user,
            q_record_given_handle=args.q_record_given_handle,
            handles_per_user=min(
                args.H,
                args.handles_per_user
                if args.handles_per_user is not None
                else 2 * args.records_per_silo,
            ),
            records_per_silo=args.records_per_silo,
            seed=seed,
        )
        for seed in args.seeds
    ]
    output = pd.concat(frames, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output / "vector_results.csv", index=False)
    atomic_write_json(
        args.output / "manifest.json",
        {
            "config": asdict(config),
            "config_hash": config.config_hash,
            "dry_run": args.dry_run,
            "rows": len(output),
            "scope": "bounded-vector mechanism study",
        },
    )
    print(
        json.dumps(
            {
                "rows": len(output),
                "config_hash": config.config_hash,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
