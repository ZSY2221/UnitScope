from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from .stats import paired_seed_bootstrap


METHODS = (
    "full-identity-reference",
    "uldp-avg-weighted-oracle",
    "group-privacy-k2-person-corrected",
    "record-level-dp-sgd-reference",
)
MAIN_METHODS = (
    "stable-local-fallback",
    "unitscope-lambda-sweep",
    "handle-only-drop-residual",
    "naive-recalibrated-2c",
)
DISPLAY = {
    "stable-local-fallback": "Stable-Local",
    "unitscope-lambda-sweep": "UnitScope-Fixed",
    "handle-only-drop-residual": "Handle-Only",
    "naive-recalibrated-2c": "Naive-2C",
    "full-identity-reference": "Complete-ID Joint Oracle",
    "uldp-avg-weighted-oracle": "ULDP-FL AVG-w Oracle",
    "group-privacy-k2-person-corrected": "Group-2",
    "record-level-dp-sgd-reference": "Record-DP",
}
ASSUMPTIONS = {
    "full-identity-reference": (
        "Complete shared user map, one jointly clipped user unit. Assumption "
        "reference only and excluded from the C-certified ranking."
    ),
    "uldp-avg-weighted-oracle": (
        "Complete common user IDs and per-silo record counts, with ULDP-FL "
        "AVG-w weighting in the common trainer. Assumption reference only."
    ),
    "group-privacy-k2-person-corrected": (
        "Complete user IDs and a public stable cap of two records per person. "
        "Overflow records are dropped and noise is calibrated to 2C."
    ),
    "record-level-dp-sgd-reference": (
        "No cross-silo identity. Each record is clipped at C. The guarantee is "
        "record-level DP and not user-level DP."
    ),
}


def _load_rows(registry_path: Path) -> tuple[Mapping[str, object], pd.DataFrame]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = []
    for entry in registry["entries"]:
        directory = (registry_path.parent / str(entry["result_directory"])).resolve()
        result_rows = [
            json.loads(line)
            for line in (directory / "learning_results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        if {str(row["method"]) for row in result_rows} != set(METHODS):
            raise ValueError(f"wrong method set in {directory}")
        for row in result_rows:
            if row["config_hash"] != entry["config_hash"]:
                raise ValueError(f"config hash mismatch in {directory}")
            row["replicate"] = int(entry["replicate"])
            rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) != 160:
        raise ValueError("reference-method study must contain all 160 result rows")
    return registry, frame


def _load_main_trajectories(registry_path: Path) -> pd.DataFrame:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = []
    for entry in registry["entries"]:
        directory = (registry_path.parent / str(entry["result_directory"])).resolve()
        for path in sorted((directory / "trajectories").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["method"] not in MAIN_METHODS:
                continue
            for point in payload["trajectory"]:
                metric_name = (
                    "patient_or_user_auprc"
                    if payload["dataset"] == "synthea"
                    else "user_macro_accuracy"
                )
                rows.append(
                    {
                        "dataset": payload["dataset"],
                        "replicate": int(entry["replicate"]),
                        "seed": int(payload["seed"]),
                        "method": payload["method"],
                        "round": int(point["round"]),
                        "epsilon": point["epsilon"],
                        "metric_name": metric_name,
                        "metric_value": float(point[metric_name]),
                    }
                )
    frame = pd.DataFrame(rows)
    expected = 2 * 20 * len(MAIN_METHODS) * 6
    if len(frame) != expected:
        raise ValueError(
            f"main trajectory grid has {len(frame)} rows, expected {expected}"
        )
    return frame


def _interval(values: Sequence[float]) -> tuple[float, float, float]:
    interval = paired_seed_bootstrap(values, seed=20260717)
    return interval.estimate, interval.lower, interval.upper


def _plot_trajectories(summary: pd.DataFrame, output: Path) -> None:
    colors = {
        "stable-local-fallback": "#6B7280",
        "unitscope-lambda-sweep": "#176B87",
        "handle-only-drop-residual": "#D97706",
        "naive-recalibrated-2c": "#8B5CF6",
    }
    markers = {
        "stable-local-fallback": "s",
        "unitscope-lambda-sweep": "o",
        "handle-only-drop-residual": "v",
        "naive-recalibrated-2c": "P",
    }
    for dataset in ("femnist", "synthea"):
        fig, axis = plt.subplots(figsize=(4.15, 2.75))
        subset = summary[summary.dataset == dataset]
        for method in MAIN_METHODS:
            method_frame = subset[subset.method == method].sort_values("round")
            axis.plot(
                method_frame["round"],
                method_frame["metric_mean"],
                color=colors[method],
                marker=markers[method],
                markersize=3.8,
                linewidth=1.35,
                label=DISPLAY[method],
            )
            axis.fill_between(
                method_frame["round"],
                method_frame["ci_lower"],
                method_frame["ci_upper"],
                color=colors[method],
                alpha=0.11,
                linewidth=0,
            )
        axis.set_xlabel("Training round")
        axis.set_ylabel(
            "Writer macro accuracy" if dataset == "femnist" else "Patient AUPRC"
        )
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.55, alpha=0.7)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("#111827")
        axis.legend(frameon=False, fontsize=7, ncol=2, loc="best")
        fig.tight_layout(pad=0.45)
        fig.savefig(output / f"{dataset}_main_convergence.png", dpi=300)
        fig.savefig(output / f"{dataset}_main_convergence.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the reference methods")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--main-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry_path = args.registry.resolve()
    _, frame = _load_rows(registry_path)

    method_rows = []
    for dataset in ("femnist", "synthea"):
        subset = frame[frame.dataset == dataset]
        for method in METHODS:
            method_frame = subset[subset.method == method].sort_values("replicate")
            if list(method_frame.replicate) != list(range(20)):
                raise ValueError(f"missing replicate for {dataset} {method}")
            mean, lower, upper = _interval(method_frame.metric_value.to_numpy())
            utilization = method_frame.effective_record_utilization.to_numpy(
                dtype=float
            )
            util_mean, util_lower, util_upper = _interval(utilization)
            method_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "display_name": DISPLAY[method],
                    "metric_mean": mean,
                    "metric_ci_lower": lower,
                    "metric_ci_upper": upper,
                    "effective_record_utilization_mean": util_mean,
                    "effective_record_utilization_ci_lower": util_lower,
                    "effective_record_utilization_ci_upper": util_upper,
                    "overflow_drop_fraction_mean": 1.0 - util_mean
                    if method == "group-privacy-k2-person-corrected"
                    else 0.0,
                    "certified_sensitivity_over_c": float(
                        method_frame.certified_sensitivity.mean()
                    ),
                    "privacy_semantics": str(method_frame.iloc[0].privacy_semantics),
                    "replicates": len(method_frame),
                }
            )
    method_summary = pd.DataFrame(method_rows)
    method_summary.to_csv(output / "method_summary.csv", index=False)

    log_rows = []
    for (dataset, method), group in frame.groupby(["dataset", "method"]):
        log_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "successful_runs": len(group),
                "unique_seeds": group.seed.nunique(),
                "nan_metrics": int(group.metric_value.isna().sum()),
                "validation_pass_all": bool(group.validation_pass.all()),
                "epsilon_values": ",".join(
                    map(str, sorted(group.epsilon.dropna().unique()))
                ),
                "noise_multiplier_values": ",".join(
                    map(str, sorted(group.noise_multiplier.unique()))
                ),
                "wall_time_seconds_total": float(group.wall_time_seconds.sum()),
                "wall_time_seconds_mean": float(group.wall_time_seconds.mean()),
                "record_utilization_mean": float(
                    group.effective_record_utilization.mean()
                ),
            }
        )
    pd.DataFrame(log_rows).to_csv(output / "run_log_summary.csv", index=False)

    latex_lines = []
    for row in method_rows:
        dataset = "FEMNIST" if row["dataset"] == "femnist" else "Synthea"
        sensitivity = (
            "record $C$"
            if row["privacy_semantics"] == "record-dp"
            else f"{row['certified_sensitivity_over_c']:.0f}$C$"
        )
        latex_lines.append(
            f"{dataset} & {row['display_name']} & {sensitivity} & "
            f"{100 * row['effective_record_utilization_mean']:.1f}\\% & "
            f"{row['metric_mean']:.4f} "
            f"[{row['metric_ci_lower']:.4f}, {row['metric_ci_upper']:.4f}] \\\\"
        )
    (output / "latex_rows.tex").write_text(
        "\n".join(latex_lines) + "\n", encoding="utf-8"
    )
    (output / "assumption_notes.md").write_text(
        "# Assumption differences\n\n"
        + "\n".join(f"- **{DISPLAY[key]}:** {ASSUMPTIONS[key]}" for key in METHODS)
        + "\n",
        encoding="utf-8",
    )

    trajectory = _load_main_trajectories(args.main_registry.resolve())
    trajectory_rows = []
    for (dataset, method, round_index, metric_name), group in trajectory.groupby(
        ["dataset", "method", "round", "metric_name"]
    ):
        mean, lower, upper = _interval(group.sort_values("replicate").metric_value)
        trajectory_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "round": round_index,
                "metric_name": metric_name,
                "metric_mean": mean,
                "ci_lower": lower,
                "ci_upper": upper,
                "replicates": len(group),
            }
        )
    trajectory_summary = pd.DataFrame(trajectory_rows)
    trajectory.to_csv(output / "main_trajectory_long.csv", index=False)
    trajectory_summary.to_csv(output / "main_trajectory_summary.csv", index=False)
    _plot_trajectories(trajectory_summary, output)
    print(
        json.dumps(
            {
                "result_rows": len(frame),
                "method_summary_rows": len(method_summary),
                "trajectory_rows": len(trajectory),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
