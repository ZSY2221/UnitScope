from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stats import paired_ratio_bootstrap, paired_seed_bootstrap


def _load_jsonl(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not rows:
        raise ValueError("no learning rows found")
    return pd.DataFrame(rows)


def _mean_interval(values: np.ndarray, seed: int = 20260715) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(20_000, len(values)))
    draws = values[indices].mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(lower), float(upper)


def assemble_frontier(main: pd.DataFrame, sweep: pd.DataFrame) -> pd.DataFrame:
    endpoints = main[
        main["method"].isin(
            ["stable-local-fallback", "unitscope-equal", "handle-only-drop-residual"]
        )
    ].copy()
    endpoints["frontier_source"] = endpoints["method"].map(
        {
            "stable-local-fallback": "lambda=0 endpoint",
            "unitscope-equal": "equal-allocation ablation",
            "handle-only-drop-residual": "lambda=1 endpoint",
        }
    )
    sweep = sweep.copy()
    sweep["frontier_source"] = "fixed descriptive grid"
    frame = pd.concat([endpoints, sweep], ignore_index=True)
    frame["lambda"] = pd.to_numeric(frame["lambda"], errors="raise")
    duplicates = frame.duplicated(["dataset", "seed", "lambda"], keep=False)
    if duplicates.any():
        cells = (
            frame.loc[duplicates, ["dataset", "seed", "lambda"]]
            .head()
            .to_dict("records")
        )
        raise ValueError(f"duplicate frontier cells: {cells}")
    return frame


def summarize(frontier: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, lambda_value), group in frontier.groupby(["dataset", "lambda"]):
        values = group["metric_value"].to_numpy(dtype=np.float64)
        lower, upper = _mean_interval(values)
        rows.append(
            {
                "dataset": dataset,
                "lambda": float(lambda_value),
                "frontier_source": group["frontier_source"].iloc[0],
                "seeds": int(group["seed"].nunique()),
                "metric_name": group["metric_name"].iloc[0],
                "metric_mean": float(values.mean()),
                "metric_std": float(values.std(ddof=1)),
                "metric_ci_lower": lower,
                "metric_ci_upper": upper,
                "effective_record_utilization_mean": float(
                    pd.to_numeric(group["effective_record_utilization"]).mean()
                ),
                "partial_handle_record_fraction_mean": float(
                    pd.to_numeric(group["partial_handle_record_fraction"]).mean()
                ),
                "residual_record_fraction_mean": float(
                    pd.to_numeric(group["residual_record_fraction"]).mean()
                ),
                "gradient_distortion_l2_mean": float(
                    pd.to_numeric(group["gradient_distortion_l2"]).mean()
                ),
                "certified_sensitivity_mean": float(
                    pd.to_numeric(group["certified_sensitivity"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "lambda"])


def paired_effects(frontier: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in frontier.groupby("dataset"):
        references = main[
            (main["dataset"] == dataset)
            & main["method"].isin(
                [
                    "stable-local-fallback",
                    "handle-only-drop-residual",
                    "naive-recalibrated-2c",
                    "full-identity-reference",
                ]
            )
        ].pivot(index="seed", columns="method", values="metric_value")
        for lambda_value, point in group.groupby("lambda"):
            point_values = point.set_index("seed")["metric_value"].rename("frontier")
            paired = references.join(point_values, how="inner").dropna()
            fallback = paired["stable-local-fallback"].to_numpy(dtype=np.float64)
            handle = paired["handle-only-drop-residual"].to_numpy(dtype=np.float64)
            naive = paired["naive-recalibrated-2c"].to_numpy(dtype=np.float64)
            oracle = paired["full-identity-reference"].to_numpy(dtype=np.float64)
            unit = paired["frontier"].to_numpy(dtype=np.float64)
            effects = {
                "gain_over_fallback": unit - fallback,
                "difference_from_handle_only": unit - handle,
                "gain_over_naive_2c": unit - naive,
            }
            row: dict[str, object] = {
                "dataset": dataset,
                "lambda": float(lambda_value),
                "paired_seeds": len(paired),
            }
            for name, values in effects.items():
                interval = paired_seed_bootstrap(values)
                row[f"{name}_mean"] = interval.estimate
                row[f"{name}_ci_lower"] = interval.lower
                row[f"{name}_ci_upper"] = interval.upper
                row[f"{name}_positive_seed_fraction"] = float(np.mean(values > 0))
            handle_gain = handle - fallback
            oracle_gap = oracle - fallback
            handle_retention = paired_ratio_bootstrap(unit - fallback, handle_gain)
            oracle_recovery = paired_ratio_bootstrap(unit - fallback, oracle_gap)
            row["handle_gain_retention"] = handle_retention.estimate
            row["handle_gain_retention_ci_lower"] = handle_retention.lower
            row["handle_gain_retention_ci_upper"] = handle_retention.upper
            row["oracle_gap_recovery"] = oracle_recovery.estimate
            row["oracle_gap_recovery_ci_lower"] = oracle_recovery.lower
            row["oracle_gap_recovery_ci_upper"] = oracle_recovery.upper
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "lambda"])


def plot_frontier(summary: pd.DataFrame, output: Path) -> None:
    datasets = list(summary["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(11.8, 4.3), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        group = summary[summary["dataset"] == dataset].sort_values("lambda")
        x = group["lambda"].to_numpy(dtype=np.float64)
        y = group["metric_mean"].to_numpy(dtype=np.float64)
        yerr = np.vstack(
            [
                y - group["metric_ci_lower"].to_numpy(dtype=np.float64),
                group["metric_ci_upper"].to_numpy(dtype=np.float64) - y,
            ]
        )
        axis.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=3,
            color="#1f77b4",
            label="Task utility",
        )
        axis.set_xlabel("Identity allocation lambda")
        axis.set_ylabel(
            "Writer macro accuracy" if dataset == "femnist" else "Patient AUPRC"
        )
        axis.set_title("FEMNIST" if dataset == "femnist" else "Synthea")
        axis.grid(alpha=0.25)
        utilization_axis = axis.twinx()
        utilization_axis.plot(
            x,
            group["effective_record_utilization_mean"],
            marker="s",
            linestyle="--",
            color="#d95f02",
            label="Record utilization",
        )
        utilization_axis.set_ylim(0, 1.08)
        utilization_axis.set_ylabel("Effective record utilization")
        utilization_axis.axvline(0.9, color="#777777", linestyle=":", linewidth=1)
        handles_a, labels_a = axis.get_legend_handles_labels()
        handles_b, labels_b = utilization_axis.get_legend_handles_labels()
        axis.legend(
            handles_a + handles_b, labels_a + labels_b, loc="lower right", fontsize=8
        )
    fig.suptitle("UnitScope identity-allocation frontier at fixed person-level privacy")
    fig.tight_layout()
    fig.savefig(output / "lambda_frontier.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "lambda_frontier.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, effects: pd.DataFrame, output: Path) -> None:
    lines = [
        "# UnitScope lambda frontier",
        "",
        "All points use the same person-level privacy target, model, schedule and ten paired seeds. "
        "Lambda is fixed by the descriptive grid and is not selected on the protected test set.",
        "",
    ]
    for dataset in summary["dataset"].drop_duplicates():
        point = summary[
            (summary["dataset"] == dataset) & np.isclose(summary["lambda"], 0.9)
        ].iloc[0]
        effect = effects[
            (effects["dataset"] == dataset) & np.isclose(effects["lambda"], 0.9)
        ].iloc[0]
        lines.extend(
            [
                f"## {str(dataset).upper()}",
                "",
                f"At lambda=0.9, utility is {point.metric_mean:.4f} "
                f"[{point.metric_ci_lower:.4f}, {point.metric_ci_upper:.4f}] while effective record "
                f"utilization remains {100 * point.effective_record_utilization_mean:.1f}%.",
                "",
                f"The paired gain over fallback is {effect.gain_over_fallback_mean:+.4f} "
                f"[{effect.gain_over_fallback_ci_lower:+.4f}, "
                f"{effect.gain_over_fallback_ci_upper:+.4f}]. The difference from Handle-Only is "
                f"{effect.difference_from_handle_only_mean:+.4f} "
                f"[{effect.difference_from_handle_only_ci_lower:+.4f}, "
                f"{effect.difference_from_handle_only_ci_upper:+.4f}].",
                "",
                f"This point retains {100 * effect.handle_gain_retention:.1f}% of the Handle-Only "
                f"gain over fallback [{100 * effect.handle_gain_retention_ci_lower:.1f}%, "
                f"{100 * effect.handle_gain_retention_ci_upper:.1f}%] without dropping the residual path.",
                "",
            ]
        )
    (output / "LAMBDA_FRONTIER_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze UnitScope's fixed lambda frontier"
    )
    parser.add_argument("--main-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--sweep-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main_frame = _load_jsonl(args.main_inputs)
    sweep_frame = _load_jsonl(args.sweep_inputs)
    frontier = assemble_frontier(main_frame, sweep_frame)
    summary = summarize(frontier)
    effects = paired_effects(frontier, main_frame)
    args.output.mkdir(parents=True, exist_ok=True)
    frontier.to_csv(args.output / "lambda_frontier_long.csv", index=False)
    summary.to_csv(args.output / "lambda_frontier_summary.csv", index=False)
    effects.to_csv(args.output / "lambda_frontier_paired_effects.csv", index=False)
    plot_frontier(summary, args.output)
    write_report(summary, effects, args.output)
    print(
        json.dumps(
            {
                "frontier_points": len(summary),
                "paired_effects": len(effects),
                "seeds_per_point": sorted(summary["seeds"].unique().tolist()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
