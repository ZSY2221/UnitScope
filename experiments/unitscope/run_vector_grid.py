from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import pandas as pd

from .manifest import atomic_write_json
from .model import PublicConfig
from .vector_experiment import run_configuration


def _hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cells() -> list[dict[str, float | int | str]]:
    cells = []
    for q_user, q_record, H, overlap in product(
        (0.25, 0.5, 0.75),
        (0.25, 0.5, 0.75),
        (1, 2, 4),
        (0.2, 0.5, 0.8),
    ):
        cells.append(
            {
                "stage": "coverage-h-overlap",
                "q_user": q_user,
                "q_record": q_record,
                "H": H,
                "overlap": overlap,
                "correlation": 0.5,
                "lambda": 0.5,
            }
        )
    for correlation, lambda_value in product(
        (-0.5, 0.0, 0.5, 0.9), (0.0, 0.25, 0.5, 0.75, 1.0)
    ):
        cells.append(
            {
                "stage": "lambda-geometry",
                "q_user": 0.5,
                "q_record": 0.5,
                "H": 1,
                "overlap": 0.5,
                "correlation": correlation,
                "lambda": lambda_value,
            }
        )
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen UnitScope vector-mechanism grid"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-users", type=int, default=10_000)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--n-silos", type=int, default=6)
    parser.add_argument("--B", type=int, default=6)
    parser.add_argument("--records-per-silo", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    cells = _cells()
    if args.dry_run:
        args.n_users = min(args.n_users, 32)
        args.dimension = min(args.dimension, 16)
        args.seeds = args.seeds[:1]
        cells = [
            cells[0],
            next(cell for cell in cells if cell["stage"] == "lambda-geometry"),
        ]
    args.output.mkdir(parents=True, exist_ok=True)
    run_directory = args.output / "runs"
    run_directory.mkdir(parents=True, exist_ok=True)
    completed = 0
    for cell in cells:
        specification = {
            **cell,
            "n_users": args.n_users,
            "dimension": args.dimension,
            "n_silos": args.n_silos,
            "B": args.B,
            "records_per_silo": args.records_per_silo,
            "seeds": args.seeds,
        }
        config_hash = _hash(specification)
        path = run_directory / f"{config_hash}.csv"
        if path.exists():
            completed += 1
            continue
        config = PublicConfig(
            task_domain=f"vector-grid-{cell['stage']}",
            epoch="frozen",
            clip_norm=1.0,
            max_handle_groups=int(cell["H"]),
            max_fallback_slots=args.B,
            lambda_=float(cell["lambda"]),
            public_normalizer=float(args.n_users),
            noise_multiplier=1.0,
            rounds=1,
            delta=1e-5,
            n_silos=args.n_silos,
        )
        rows = []
        for seed in args.seeds:
            frame = run_configuration(
                config,
                n_users=args.n_users,
                dimension=args.dimension,
                overlap_probability=float(cell["overlap"]),
                gradient_correlation=float(cell["correlation"]),
                vector_norm=0.6,
                q_user=float(cell["q_user"]),
                q_record_given_handle=float(cell["q_record"]),
                handles_per_user=min(int(cell["H"]), 2 * args.records_per_silo),
                records_per_silo=args.records_per_silo,
                seed=seed,
            )
            frame["grid_stage"] = cell["stage"]
            frame["grid_config_hash"] = config_hash
            rows.append(frame)
        pd.concat(rows, ignore_index=True).to_csv(path, index=False)
        completed += 1
        print(
            json.dumps(
                {
                    "completed": completed,
                    "total": len(cells),
                    "config_hash": config_hash,
                }
            ),
            flush=True,
        )
    atomic_write_json(
        args.output / "grid_manifest.json",
        {
            "dry_run": args.dry_run,
            "cells": len(cells),
            "seeds": args.seeds,
            "n_users": args.n_users,
            "dimension": args.dimension,
            "records_per_silo": args.records_per_silo,
            "warning": "This grid measures vector-query mechanisms, not trained-model accuracy.",
        },
    )


if __name__ == "__main__":
    main()
