"""Evaluate the B-aware lambda gate on the bounded-vector grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .accounting import calibrate_noise_multiplier
from .model import MethodId, PublicConfig
from .stats import paired_seed_bootstrap
from .vector_experiment import run_configuration


SEEDS = tuple(range(600, 610))
GRID = tuple((B, mixed) for B in (2, 4, 6, 8) for mixed in (False, True))


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _config(B: int, mixed: bool, lam: float) -> PublicConfig:
    return PublicConfig(
        task_domain=f"b-aware-gate-B{B}-{'mixed' if mixed else 'pure'}",
        epoch="b-aware-gate-v1",
        clip_norm=1.0,
        max_handle_groups=2,
        max_fallback_slots=B,
        lambda_=lam,
        public_normalizer=1000.0,
        noise_multiplier=calibrate_noise_multiplier(8.0, rounds=1, delta=1e-6),
        rounds=1,
        delta=1e-6,
        n_silos=6,
    )


def _run_cell(B: int, mixed: bool, seed: int, output: Path) -> list[dict[str, object]]:
    gated_lambda = 0.0 if B == 2 and not mixed else 0.75
    specification = {
        "B": B,
        "mixed_local_slots": mixed,
        "seed": seed,
        "gated_lambda": gated_lambda,
        "ungated_lambda": 0.75,
        "n_users": 1000,
        "dimension": 64,
        "protocol": "one-release fixed weighted sum of bounded vectors",
    }
    cell_hash = _hash(specification)
    destination = output / "runs" / f"{cell_hash}.csv"
    if destination.exists():
        return pd.read_csv(destination).to_dict("records")

    common = dict(
        n_users=1000,
        dimension=64,
        overlap_probability=0.75,
        gradient_correlation=0.5,
        vector_norm=0.6,
        q_user=0.75,
        q_record_given_handle=0.5,
        handles_per_user=2,
        records_per_silo=2,
        seed=seed,
        mixed_local_slots=mixed,
    )
    ungated = run_configuration(_config(B, mixed, 0.75), **common)
    gated = run_configuration(_config(B, mixed, gated_lambda), **common)
    stable = ungated[ungated["method"] == MethodId.STABLE_LOCAL_FALLBACK.value].copy()
    stable["reported_method"] = "stable-local-fallback"
    stable["gate_lambda"] = 0.0
    stable["cell_hash"] = cell_hash
    stable["seed"] = seed
    stable["B"] = B
    stable["mixed_local_slots"] = mixed

    ungated_row = ungated[ungated["method"] == MethodId.UNITSCOPE_TUNED.value].copy()
    ungated_row["reported_method"] = "ungated-unitscope-lambda075"
    ungated_row["gate_lambda"] = 0.75
    ungated_row["cell_hash"] = cell_hash
    ungated_row["seed"] = seed
    ungated_row["B"] = B
    ungated_row["mixed_local_slots"] = mixed

    gated_row = gated[gated["method"] == MethodId.UNITSCOPE_TUNED.value].copy()
    gated_row["reported_method"] = "gated-unitscope"
    gated_row["gate_lambda"] = gated_lambda
    gated_row["cell_hash"] = cell_hash
    gated_row["seed"] = seed
    gated_row["B"] = B
    gated_row["mixed_local_slots"] = mixed
    frame = pd.concat([stable, ungated_row, gated_row], ignore_index=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame.to_dict("records")


def _mean_ci(values: pd.Series, seed: int) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(20_000, len(array)))].mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(array.mean()), float(lower), float(upper)


def analyze(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = ["B", "mixed_local_slots", "reported_method"]
    summary_rows = []
    for index, (keys, group) in enumerate(raw.groupby(groups, sort=True)):
        row = dict(zip(groups, keys))
        for metric in ("deterministic_distortion", "noisy_aggregate_mse"):
            mean, lower, upper = _mean_ci(group[metric], 96000 + index)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
        row["seeds"] = int(group.seed.nunique())
        row["gate_lambda"] = float(group.gate_lambda.iloc[0])
        row["certified_sensitivity_max"] = float(group.certified_sensitivity.max())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(groups).reset_index(drop=True)

    paired_rows = []
    for index, (keys, group) in enumerate(
        raw.groupby(["B", "mixed_local_slots"], sort=True)
    ):
        pivot = group.pivot(
            index="seed", columns="reported_method", values="noisy_aggregate_mse"
        )
        row = {"B": keys[0], "mixed_local_slots": keys[1], "seeds": len(pivot)}
        for label, comparator in (
            ("gated_vs_stable", "stable-local-fallback"),
            ("gated_vs_ungated", "ungated-unitscope-lambda075"),
        ):
            differences = (pivot["gated-unitscope"] - pivot[comparator]).to_numpy(
                dtype=float
            )
            interval = paired_seed_bootstrap(
                differences, n_resamples=20_000, seed=97000 + index
            )
            row[f"{label}_mean"] = interval.estimate
            row[f"{label}_ci_lower"] = interval.lower
            row[f"{label}_ci_upper"] = interval.upper
            row[f"{label}_max_abs"] = float(np.max(np.abs(differences)))
        paired_rows.append(row)
    return summary, pd.DataFrame(paired_rows).sort_values(
        ["B", "mixed_local_slots"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for B, mixed in GRID:
        for seed in SEEDS:
            rows.extend(_run_cell(B, mixed, seed, args.output))
    raw = pd.DataFrame(rows)
    raw.to_csv(args.output / "raw_results.csv", index=False)
    summary, paired = analyze(raw)
    summary.to_csv(args.output / "summary.csv", index=False)
    paired.to_csv(args.output / "paired_effects.csv", index=False)
    manifest = {
        "family_id": "b-aware-gate-v1",
        "seeds": list(SEEDS),
        "grid": [{"B": B, "mixed_local_slots": mixed} for B, mixed in GRID],
        "rows": len(raw),
        "config_hash": _hash(
            {"seeds": SEEDS, "grid": GRID, "protocol": "b-aware-gate-v1"}
        ),
        "expected": "gated equals stable at B=2 pure and equals ungated elsewhere",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({"rows": len(raw), "config_hash": manifest["config_hash"]}, indent=2)
    )


if __name__ == "__main__":
    main()
