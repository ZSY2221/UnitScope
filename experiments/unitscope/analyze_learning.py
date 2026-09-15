from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .manifest import atomic_write_json
from .model import MethodId
from .stats import (
    hierarchical_cluster_effect_bootstrap,
    hierarchical_cluster_recovery_bootstrap,
    holm_adjust,
    paired_ratio_bootstrap,
    paired_seed_bootstrap,
    paired_sign_flip_test,
)


def _load_jsonl(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("no learning result rows found")
    frame = pd.DataFrame(rows)
    duplicates = frame.duplicated(["run_id"], keep=False)
    if duplicates.any():
        raise ValueError(
            f"duplicate run IDs detected: {frame.loc[duplicates, 'run_id'].tolist()[:5]}"
        )
    return frame


def _load_cluster_runs(group: pd.DataFrame, cluster_directories: list[Path]):
    runs = {}
    for row in group.itertuples(index=False):
        candidates = [
            directory / f"{row.run_id}.json" for directory in cluster_directories
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(candidates[0])
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs.setdefault(str(row.method), {})[int(row.seed)] = payload["clusters"]
    return runs


def analyze(
    frame: pd.DataFrame,
    metric: str,
    cluster_directories: list[Path] | None = None,
    confirmatory_config_hashes: set[str] | None = None,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    if confirmatory_config_hashes is not None and metric != "metric_value":
        raise ValueError(
            "confirmatory analysis must use the per-row metric_value field"
        )
    required = {"dataset", "config_hash", "seed", "method", metric}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"result data misses {sorted(missing)}")
    summaries = []
    tests: Dict[str, float] = {}
    group_columns = ["dataset", "config_hash"]
    for group_key, group in frame.groupby(group_columns, dropna=False):
        cluster_runs = (
            _load_cluster_runs(group, cluster_directories)
            if cluster_directories is not None
            else None
        )
        pivot = group.pivot(index="seed", columns="method", values=metric)
        competitors = (
            MethodId.STABLE_LOCAL_FALLBACK.value,
            MethodId.HANDLE_ONLY.value,
            MethodId.NAIVE_RECALIBRATED_2C.value,
            MethodId.FULL_IDENTITY.value,
        )
        for unit_method in (
            MethodId.UNITSCOPE_EQUAL.value,
            MethodId.UNITSCOPE_TUNED.value,
        ):
            needed = (unit_method, *competitors)
            if any(column not in pivot for column in needed):
                continue
            complete = pivot[list(needed)].dropna()
            if complete.empty:
                continue
            unit = complete[unit_method].to_numpy()
            fallback = complete[MethodId.STABLE_LOCAL_FALLBACK.value].to_numpy()
            handle = complete[MethodId.HANDLE_ONLY.value].to_numpy()
            naive = complete[MethodId.NAIVE_RECALIBRATED_2C.value].to_numpy()
            full = complete[MethodId.FULL_IDENTITY.value].to_numpy()
            effects = {
                "delta_fallback": unit - fallback,
                "interior_gain": unit - np.maximum(fallback, handle),
                "safe_competitor_gain": unit
                - np.maximum.reduce([fallback, handle, naive]),
                "full_identity_gap": full - fallback,
            }
            row = {
                "dataset": group_key[0],
                "config_hash": group_key[1],
                "unit_method": unit_method,
                "metric": metric,
                "paired_seeds": len(complete),
            }
            for name, values in effects.items():
                row[f"{name}_mean"] = float(values.mean())
                row[f"{name}_positive_seed_fraction"] = float(np.mean(values > 0))
                if len(values) >= 2:
                    interval = paired_seed_bootstrap(values)
                    row[f"{name}_ci_lower"] = interval.lower
                    row[f"{name}_ci_upper"] = interval.upper
                else:
                    row[f"{name}_ci_lower"] = np.nan
                    row[f"{name}_ci_upper"] = np.nan
            denominator = effects["full_identity_gap"]
            denominator_mean = float(denominator.mean())
            denominator_valid = (
                denominator_mean >= 0.005
                and row["full_identity_gap_ci_lower"] > 0
                and row["full_identity_gap_ci_upper"] > 0
            )
            if denominator_valid:
                try:
                    recovery = paired_ratio_bootstrap(
                        effects["delta_fallback"], denominator
                    )
                    row["recovery_mean"] = recovery.estimate
                    row["recovery_ci_lower"] = recovery.lower
                    row["recovery_ci_upper"] = recovery.upper
                    row["recovery_status"] = "reported"
                except ValueError as error:
                    row["recovery_mean"] = np.nan
                    row["recovery_ci_lower"] = np.nan
                    row["recovery_ci_upper"] = np.nan
                    row["recovery_status"] = f"N/A: {error}"
            else:
                row["recovery_mean"] = np.nan
                row["recovery_ci_lower"] = np.nan
                row["recovery_ci_upper"] = np.nan
                row["recovery_status"] = (
                    "N/A: unstable or sub-threshold full-identity gap"
                )
            exact_confirmatory_seeds = len(complete) == 10 and set(
                map(int, complete.index)
            ) == set(range(10))
            confirmatory_allowed = (
                exact_confirmatory_seeds
                and unit_method == MethodId.UNITSCOPE_TUNED.value
                and str(group_key[0]) in {"femnist", "synthea"}
                and confirmatory_config_hashes is not None
                and str(group_key[1]) in confirmatory_config_hashes
            )
            row["inference_status"] = (
                "preregistered-confirmatory"
                if confirmatory_allowed
                else "descriptive-only"
            )
            if confirmatory_allowed:
                for name in ("interior_gain", "safe_competitor_gain"):
                    key = f"{group_key[0]}|{group_key[1]}|{unit_method}|{name}"
                    tests[key] = paired_sign_flip_test(effects[name])
            if cluster_runs is not None and len(complete) >= 2:
                cluster_metric = (
                    "auprc" if str(group_key[0]) == "synthea" else "accuracy"
                )
                for name, competitors_for_effect in (
                    ("delta_fallback", (MethodId.STABLE_LOCAL_FALLBACK.value,)),
                    (
                        "interior_gain",
                        (
                            MethodId.STABLE_LOCAL_FALLBACK.value,
                            MethodId.HANDLE_ONLY.value,
                        ),
                    ),
                    (
                        "safe_competitor_gain",
                        (
                            MethodId.STABLE_LOCAL_FALLBACK.value,
                            MethodId.HANDLE_ONLY.value,
                            MethodId.NAIVE_RECALIBRATED_2C.value,
                        ),
                    ),
                ):
                    interval = hierarchical_cluster_effect_bootstrap(
                        cluster_runs,
                        unit_method,
                        competitors_for_effect,
                        metric=cluster_metric,
                    )
                    row[f"{name}_writer_user_ci_lower"] = interval.lower
                    row[f"{name}_writer_user_ci_upper"] = interval.upper
                    row[f"{name}_minimum_clusters"] = interval.minimum_clusters_per_seed
                full_gap_interval = hierarchical_cluster_effect_bootstrap(
                    cluster_runs,
                    MethodId.FULL_IDENTITY.value,
                    (MethodId.STABLE_LOCAL_FALLBACK.value,),
                    metric=cluster_metric,
                )
                row["full_identity_gap_writer_user_ci_lower"] = full_gap_interval.lower
                row["full_identity_gap_writer_user_ci_upper"] = full_gap_interval.upper
                cluster_denominator_valid = (
                    full_gap_interval.estimate >= 0.005 and full_gap_interval.lower > 0
                )
                if denominator_valid and cluster_denominator_valid:
                    try:
                        recovery_interval = hierarchical_cluster_recovery_bootstrap(
                            cluster_runs,
                            unit_method,
                            MethodId.STABLE_LOCAL_FALLBACK.value,
                            MethodId.FULL_IDENTITY.value,
                            metric=cluster_metric,
                        )
                        row["recovery_writer_user_ci_lower"] = recovery_interval.lower
                        row["recovery_writer_user_ci_upper"] = recovery_interval.upper
                    except ValueError as error:
                        row["recovery_writer_user_ci_lower"] = np.nan
                        row["recovery_writer_user_ci_upper"] = np.nan
                        row["recovery_status"] = f"N/A: {error}"
                else:
                    row["recovery_writer_user_ci_lower"] = np.nan
                    row["recovery_writer_user_ci_upper"] = np.nan
                    row["recovery_status"] = (
                        "N/A: cluster full-identity gap is unstable or sub-threshold"
                    )
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    adjusted = holm_adjust(tests) if tests else {}
    if not summary.empty:
        for index, result in summary.iterrows():
            if result.get("inference_status") != "preregistered-confirmatory":
                continue
            for name in ("interior_gain", "safe_competitor_gain"):
                key = f"{result['dataset']}|{result['config_hash']}|{result['unit_method']}|{name}"
                adjusted_value = adjusted.get(key, np.nan)
                summary.at[index, f"{name}_holm_p"] = adjusted_value
                lower = result.get(f"{name}_writer_user_ci_lower", np.nan)
                summary.at[index, f"{name}_confirmatory_positive"] = bool(
                    np.isfinite(adjusted_value)
                    and adjusted_value < 0.05
                    and np.isfinite(lower)
                    and lower > 0
                )
    return summary, {
        "raw_p_values": tests,
        "holm_adjusted_p_values": adjusted,
        "scope": "Seeds 0 through 9 are confirmatory; other cells are descriptive.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze frozen UnitScope paired comparisons"
    )
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", default="user_macro_accuracy")
    parser.add_argument("--cluster-dirs", type=Path, nargs="+")
    parser.add_argument(
        "--confirmatory-registry",
        type=Path,
        help="Frozen JSON registry whose config_hashes define the confirmatory family",
    )
    args = parser.parse_args()
    frame = _load_jsonl(args.inputs)
    hashes = None
    if args.confirmatory_registry is not None:
        if args.metric != "metric_value":
            raise ValueError(
                "--metric metric_value is mandatory with --confirmatory-registry"
            )
        registry = json.loads(args.confirmatory_registry.read_text(encoding="utf-8"))
        hashes = set(registry.get("config_hashes", []))
    summary, tests = analyze(frame, args.metric, args.cluster_dirs, hashes)
    if hashes is not None:
        expected = 2 * len(hashes)
        if len(tests["raw_p_values"]) != expected:
            raise ValueError(
                f"confirmatory registry implies {expected} tests but analysis produced "
                f"{len(tests['raw_p_values'])}"
            )
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "paired_effects.csv", index=False)
    atomic_write_json(args.output / "paired_tests.json", tests)
    print(
        json.dumps(
            {"rows": len(summary), "tests": len(tests["raw_p_values"])}, indent=2
        )
    )


if __name__ == "__main__":
    main()
