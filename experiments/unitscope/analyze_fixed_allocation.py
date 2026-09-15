from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .stats import (
    hierarchical_cluster_effect_bootstrap,
    holm_adjust,
    paired_ratio_bootstrap,
    paired_seed_bootstrap,
)


FIXED = "unitscope-lambda-sweep"
FALLBACK = "stable-local-fallback"
HANDLE = "handle-only-drop-residual"
NAIVE = "naive-recalibrated-2c"
METHODS = (FALLBACK, FIXED, HANDLE, NAIVE)


def _one_sided_sign_flip(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        raise ValueError("at least one finite paired effect is required")
    observed = float(array.mean())
    if len(array) <= 20:
        exceed = 0
        total = 2 ** len(array)
        for signs in product((-1.0, 1.0), repeat=len(array)):
            statistic = float(np.mean(array * np.asarray(signs)))
            exceed += statistic >= observed - 1e-15
        return exceed / total
    rng = np.random.default_rng(20260717)
    signs = rng.choice((-1.0, 1.0), size=(200_000, len(array)))
    statistics = (signs * array[None, :]).mean(axis=1)
    return float((1 + np.sum(statistics >= observed - 1e-15)) / (1 + len(statistics)))


def _resolve_result_directory(registry_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else registry_path.parent / path


def _load(registry_path: Path) -> tuple[Mapping[str, object], pd.DataFrame, dict]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = []
    clusters: dict[str, dict[str, dict[int, Sequence[Mapping[str, object]]]]] = {
        "femnist": {method: {} for method in METHODS},
        "synthea": {method: {} for method in METHODS},
    }
    for entry in registry["entries"]:
        directory = _resolve_result_directory(
            registry_path, str(entry["result_directory"])
        )
        result_path = directory / "learning_results.jsonl"
        if not result_path.exists():
            raise FileNotFoundError(result_path)
        result_rows = [
            json.loads(line)
            for line in result_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if {row["method"] for row in result_rows} != set(METHODS):
            raise ValueError(f"{directory} does not contain exactly the frozen methods")
        if any(row["config_hash"] != entry["config_hash"] for row in result_rows):
            raise ValueError(f"config hash mismatch in {directory}")
        for row in result_rows:
            row = dict(row)
            row["replicate"] = int(entry["replicate"])
            rows.append(row)
            cluster_path = directory / "clusters" / f"{row['run_id']}.json"
            payload = json.loads(cluster_path.read_text(encoding="utf-8"))
            clusters[str(row["dataset"])][str(row["method"])][
                int(entry["replicate"])
            ] = payload["clusters"]
    frame = pd.DataFrame(rows)
    if len(frame) != 2 * 20 * len(METHODS):
        raise ValueError("the replication result grid is incomplete")
    return registry, frame, clusters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the fixed-allocation experiment"
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    registry, frame, clusters = _load(args.registry.resolve())
    margin = float(registry["noninferiority_margin"])

    method_rows = []
    effect_rows = []
    raw_p_values = {}
    test_rows = []
    for dataset in ("femnist", "synthea"):
        subset = frame[frame.dataset == dataset]
        pivot = subset.pivot(
            index="replicate", columns="method", values="metric_value"
        ).sort_index()
        if list(pivot.index) != list(range(20)):
            raise ValueError(f"{dataset} is missing a replicate")
        for method in METHODS:
            interval = paired_seed_bootstrap(pivot[method].to_numpy(), seed=20260717)
            meta = subset[subset.method == method]
            method_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "metric_mean": interval.estimate,
                    "metric_ci_lower": interval.lower,
                    "metric_ci_upper": interval.upper,
                    "effective_record_utilization_mean": float(
                        meta.effective_record_utilization.mean()
                    ),
                    "certified_sensitivity_over_c": float(
                        meta.certified_sensitivity.mean()
                    ),
                    "replicates": len(meta),
                }
            )

        gain = (pivot[FIXED] - pivot[FALLBACK]).to_numpy(dtype=np.float64)
        gap_handle = (pivot[FIXED] - pivot[HANDLE]).to_numpy(dtype=np.float64)
        noninferiority = gap_handle + margin
        gap_naive = (pivot[FIXED] - pivot[NAIVE]).to_numpy(dtype=np.float64)
        handle_gain = (pivot[HANDLE] - pivot[FALLBACK]).to_numpy(dtype=np.float64)
        for name, values in (
            ("gain_over_fallback", gain),
            ("difference_from_handle_only", gap_handle),
            ("noninferiority_transformed", noninferiority),
            ("difference_from_naive_2c", gap_naive),
        ):
            interval = paired_seed_bootstrap(values, seed=20260717)
            effect_rows.append(
                {
                    "dataset": dataset,
                    "effect": name,
                    "estimate": interval.estimate,
                    "ci_lower": interval.lower,
                    "ci_upper": interval.upper,
                    "replicates": len(values),
                }
            )
        ratio = paired_ratio_bootstrap(gain, handle_gain, seed=20260717)
        effect_rows.append(
            {
                "dataset": dataset,
                "effect": "aggregate_handle_gain_retention",
                "estimate": ratio.estimate,
                "ci_lower": ratio.lower,
                "ci_upper": ratio.upper,
                "replicates": len(gain),
            }
        )

        raw_p_values[f"{dataset}|gain_over_fallback"] = _one_sided_sign_flip(gain)
        raw_p_values[f"{dataset}|noninferior_to_handle_only"] = _one_sided_sign_flip(
            noninferiority
        )

        metric = "accuracy" if dataset == "femnist" else "auprc"
        cluster_gain = hierarchical_cluster_effect_bootstrap(
            clusters[dataset], FIXED, [FALLBACK], metric=metric, seed=20260717
        )
        cluster_handle = hierarchical_cluster_effect_bootstrap(
            clusters[dataset], FIXED, [HANDLE], metric=metric, seed=20260718
        )
        effect_rows.extend(
            [
                {
                    "dataset": dataset,
                    "effect": "hierarchical_gain_over_fallback",
                    "estimate": cluster_gain.estimate,
                    "ci_lower": cluster_gain.lower,
                    "ci_upper": cluster_gain.upper,
                    "replicates": cluster_gain.seeds,
                },
                {
                    "dataset": dataset,
                    "effect": "hierarchical_difference_from_handle_only",
                    "estimate": cluster_handle.estimate,
                    "ci_lower": cluster_handle.lower,
                    "ci_upper": cluster_handle.upper,
                    "replicates": cluster_handle.seeds,
                },
            ]
        )

    adjusted = holm_adjust(raw_p_values)
    for name, raw in sorted(raw_p_values.items()):
        dataset, hypothesis = name.split("|", 1)
        test_rows.append(
            {
                "dataset": dataset,
                "hypothesis": hypothesis,
                "raw_p": raw,
                "holm_p": adjusted[name],
                "reject_at_0.05": adjusted[name] <= 0.05,
            }
        )

    method_frame = pd.DataFrame(method_rows)
    effect_frame = pd.DataFrame(effect_rows)
    test_frame = pd.DataFrame(test_rows)
    method_frame.to_csv(args.output / "method_summary.csv", index=False)
    effect_frame.to_csv(args.output / "paired_effects.csv", index=False)
    test_frame.to_csv(args.output / "hypothesis_tests.csv", index=False)
    frame.to_csv(args.output / "run_results.csv", index=False)

    summary = {
        "profile": registry["profile"],
        "frozen_at_utc": registry["frozen_at_utc"],
        "noninferiority_margin": margin,
        "method_summary": method_rows,
        "effects": effect_rows,
        "tests": test_rows,
    }
    (args.output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
