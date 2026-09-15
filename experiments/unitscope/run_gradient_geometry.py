from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .model import PublicConfig
from .vector_experiment import run_configuration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--n-users", type=int, default=2000)
    parser.add_argument("--dimension", type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run_dir = args.output / "runs"
    run_dir.mkdir(exist_ok=True)
    correlations = (-0.9, -0.5, 0.0, 0.5, 0.9)
    lambdas = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
    completed = 0
    for correlation in correlations:
        for lambda_value in lambdas:
            path = run_dir / f"rho_{correlation:+.1f}_lambda_{lambda_value:.2f}.csv"
            if path.exists():
                completed += 1
                continue
            config = PublicConfig(
                task_domain="gradient-geometry-v1",
                epoch="frozen-v1",
                clip_norm=1.0,
                max_handle_groups=1,
                max_fallback_slots=6,
                lambda_=lambda_value,
                public_normalizer=float(args.n_users),
                noise_multiplier=1.0,
                rounds=1,
                delta=1e-5,
                n_silos=6,
            )
            frames = [
                run_configuration(
                    config,
                    n_users=args.n_users,
                    dimension=args.dimension,
                    overlap_probability=0.5,
                    gradient_correlation=correlation,
                    vector_norm=0.6,
                    q_user=0.5,
                    q_record_given_handle=0.5,
                    handles_per_user=1,
                    records_per_silo=2,
                    seed=seed,
                )
                for seed in args.seeds
            ]
            pd.concat(frames, ignore_index=True).to_csv(path, index=False)
            completed += 1
            print(
                json.dumps(
                    {"completed": completed, "total": len(correlations) * len(lambdas)}
                ),
                flush=True,
            )
    all_rows = pd.concat(
        [pd.read_csv(path) for path in sorted(run_dir.glob("*.csv"))], ignore_index=True
    )
    all_rows.to_csv(args.output / "gradient_geometry_results.csv", index=False)
    summary = all_rows.groupby(
        ["gradient_correlation", "lambda", "method"], as_index=False
    ).agg(
        seeds=("seed", "nunique"),
        realized_cosine=("realized_pairwise_cosine_mean", "mean"),
        deterministic_distortion_mean=("deterministic_distortion", "mean"),
        deterministic_distortion_std=("deterministic_distortion", "std"),
        noisy_mse_mean=("noisy_aggregate_mse", "mean"),
        record_utilization=("effective_record_utilization", "mean"),
        aggregate_snr=("aggregate_snr", "mean"),
    )
    summary.to_csv(args.output / "gradient_geometry_summary.csv", index=False)
    manifest = {
        "seeds": args.seeds,
        "n_users": args.n_users,
        "dimension": args.dimension,
        "correlations": correlations,
        "lambdas": lambdas,
        "cells": len(correlations) * len(lambdas),
        "rows": len(all_rows),
        "warning": "Vector-query mechanism ablation, not trained-model accuracy.",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
