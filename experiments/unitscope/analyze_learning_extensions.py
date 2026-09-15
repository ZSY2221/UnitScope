from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stats import paired_seed_bootstrap


DISPLAY = {
    "stable-local-fallback": "Stable-Local",
    "unitscope-lambda-sweep": "UnitScope-Fixed",
    "handle-only-drop-residual": "Handle-Only",
    "naive-recalibrated-2c": "Naive-2C",
    "full-identity-reference": "Complete-ID Oracle",
}
COLORS = {
    "stable-local-fallback": "#6B7280",
    "unitscope-lambda-sweep": "#176B87",
    "handle-only-drop-residual": "#D97706",
    "naive-recalibrated-2c": "#8B5CF6",
    "full-identity-reference": "#2A9D8F",
}
MARKERS = {
    "stable-local-fallback": "s",
    "unitscope-lambda-sweep": "o",
    "handle-only-drop-residual": "v",
    "naive-recalibrated-2c": "P",
    "full-identity-reference": "D",
}


def _interval(values) -> tuple[float, float, float]:
    result = paired_seed_bootstrap(values, seed=20260718)
    return result.estimate, result.lower, result.upper


def _load(registries: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    trajectories = []
    for registry_path in registries:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        for entry in registry["entries"]:
            directory = (registry_path.parent / entry["result_directory"]).resolve()
            result_rows = [
                json.loads(line)
                for line in (directory / "learning_results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            expected = set(entry["methods"])
            if {row["method"] for row in result_rows} != expected:
                raise ValueError(f"incomplete method set in {directory}")
            for row in result_rows:
                if row["config_hash"] != entry["config_hash"]:
                    raise ValueError(f"config hash mismatch in {directory}")
                row["family"] = entry["family"]
                row["dirichlet_alpha"] = entry["dirichlet_alpha"]
                rows.append(row)
            for path in sorted((directory / "trajectories").glob("*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                for point in payload["trajectory"]:
                    metric_name = (
                        "patient_or_user_auprc"
                        if payload["dataset"] == "synthea"
                        else "user_macro_accuracy"
                    )
                    trajectories.append(
                        {
                            "family": entry["family"],
                            "dataset": payload["dataset"],
                            "dirichlet_alpha": entry["dirichlet_alpha"],
                            "seed": payload["seed"],
                            "method": payload["method"],
                            "round": point["round"],
                            "metric_name": metric_name,
                            "metric_value": point[metric_name],
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(trajectories)


def _plot_trajectories(trajectory: pd.DataFrame, output: Path) -> None:
    group_fields = ["family", "dataset", "dirichlet_alpha"]
    for keys, family_frame in trajectory.groupby(group_fields, dropna=False):
        family, dataset, alpha = keys
        fig, axis = plt.subplots(figsize=(3.25, 2.2))
        for method, method_frame in family_frame.groupby("method"):
            summary = (
                method_frame.groupby("round")["metric_value"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            sem95 = 1.96 * summary["std"].fillna(0.0) / np.sqrt(summary["count"])
            axis.plot(
                summary["round"],
                summary["mean"],
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=3.0,
                linewidth=1.15,
                label=DISPLAY[method],
            )
            axis.fill_between(
                summary["round"],
                summary["mean"] - sem95,
                summary["mean"] + sem95,
                color=COLORS[method],
                alpha=0.1,
                linewidth=0,
            )
        axis.set_xlabel("Training round")
        axis.set_ylabel(
            "Patient AUPRC"
            if dataset == "synthea"
            else "Account macro accuracy"
            if dataset in {"sent140", "movielens1m"}
            else "Writer macro accuracy"
        )
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.5, alpha=0.7)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("#111827")
        axis.legend(frameon=False, fontsize=6.1, ncol=2, loc="best")
        fig.tight_layout(pad=0.45)
        suffix = "" if pd.isna(alpha) else f"_alpha{float(alpha):.1f}".replace(".", "p")
        stem = f"{family}_{dataset}{suffix}_convergence"
        fig.savefig(output / f"{stem}.png", dpi=300)
        fig.savefig(output / f"{stem}.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the learning extensions")
    parser.add_argument("--registries", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame, trajectory = _load([path.resolve() for path in args.registries])
    if frame.metric_value.isna().any():
        raise ValueError("a successful learning run has a NaN primary metric")
    summary_rows = []
    effects = []
    group_fields = ["family", "dataset", "dirichlet_alpha"]
    for keys, group in frame.groupby(group_fields, dropna=False):
        family, dataset, alpha = keys
        seeds = sorted(group.seed.unique())
        pivot = group.pivot(index="seed", columns="method", values="metric_value")
        for method, method_frame in group.groupby("method"):
            method_frame = method_frame.sort_values("seed")
            estimate, lower, upper = _interval(method_frame.metric_value.to_numpy())
            util, util_lower, util_upper = _interval(
                method_frame.effective_record_utilization.to_numpy()
            )
            summary_rows.append(
                {
                    "family": family,
                    "dataset": dataset,
                    "dirichlet_alpha": alpha,
                    "method": method,
                    "metric_name": method_frame.iloc[0].metric_name,
                    "metric_mean": estimate,
                    "metric_ci_lower": lower,
                    "metric_ci_upper": upper,
                    "record_use_mean": util,
                    "record_use_ci_lower": util_lower,
                    "record_use_ci_upper": util_upper,
                    "certified_sensitivity_over_c": float(
                        method_frame.certified_sensitivity.mean()
                    ),
                    "replicates": len(method_frame),
                    "validation_all": bool(method_frame.validation_pass.all()),
                }
            )
        unit = "unitscope-lambda-sweep"
        for competitor in sorted(set(pivot.columns) - {unit}):
            estimate, lower, upper = _interval(
                (pivot[unit] - pivot[competitor]).to_numpy()
            )
            effects.append(
                {
                    "family": family,
                    "dataset": dataset,
                    "dirichlet_alpha": alpha,
                    "contrast": f"UnitScope minus {DISPLAY[competitor]}",
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "paired_seeds": len(seeds),
                }
            )
    summary = pd.DataFrame(summary_rows)
    effect_frame = pd.DataFrame(effects)
    summary.to_csv(output / "method_summary.csv", index=False)
    effect_frame.to_csv(output / "paired_effects.csv", index=False)
    frame.to_csv(output / "all_successful_runs.csv", index=False)
    trajectory.to_csv(output / "trajectory_long.csv", index=False)

    log = (
        frame.groupby(["family", "dataset", "dirichlet_alpha", "method"], dropna=False)
        .agg(
            successful_runs=("run_id", "count"),
            unique_seeds=("seed", "nunique"),
            validation_passed=("validation_pass", "sum"),
            wall_seconds=("wall_time_seconds", "sum"),
        )
        .reset_index()
    )
    log.to_csv(output / "run_log_summary.csv", index=False)

    latex = []
    for row in summary.itertuples(index=False):
        alpha = (
            ""
            if pd.isna(row.dirichlet_alpha)
            else f" $\\alpha={row.dirichlet_alpha:g}$"
        )
        sensitivity = f"{row.certified_sensitivity_over_c:g}$C$"
        latex.append(
            f"{row.dataset}{alpha} & {DISPLAY[row.method]} & {sensitivity} & "
            f"{100 * row.record_use_mean:.1f}\\% & {row.metric_mean:.4f} "
            f"[{row.metric_ci_lower:.4f}, {row.metric_ci_upper:.4f}] \\\\"
        )
    (output / "latex_rows.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )

    credible_unit = summary[
        (summary.family == "credible-accuracy-epsilon8-v1")
        & (summary.method == "unitscope-lambda-sweep")
    ].set_index("dataset")
    credible_pass = bool(
        credible_unit.loc["femnist", "metric_mean"] >= 0.20
        and credible_unit.loc["synthea", "metric_mean"] >= 0.28
    )
    noniid_unit = summary[
        (summary.family == "user-noniid-dirichlet-v1")
        & (summary.method == "unitscope-lambda-sweep")
    ].set_index("dirichlet_alpha")
    noniid_drop = float(
        noniid_unit.loc[0.5, "metric_mean"] - noniid_unit.loc[0.1, "metric_mean"]
    )
    noniid_pass = noniid_drop <= 0.05
    sent140 = summary[summary.family == "sent140-account-task-v1"]
    sent140_pass = bool(
        set(sent140.dataset) == {"sent140"}
        and sent140.validation_all.all()
        and (
            sent140[sent140.method == "unitscope-lambda-sweep"].record_use_mean
            >= 1.0 - 1e-12
        ).all()
    )
    report = {
        "all_successful_runs_reported": int(len(frame)),
        "nan_metrics": int(frame.metric_value.isna().sum()),
        "credible_accuracy_success_criterion": credible_pass,
        "noniid_unit_drop_alpha_0p5_to_0p1": noniid_drop,
        "noniid_robustness_success_criterion": noniid_pass,
        "sent140_integrity_success_criterion": sent140_pass,
        "interpretation_rule": "Unfavorable comparative outcomes are retained as regime boundaries.",
    }
    (output / "analysis_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_trajectories(trajectory, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
