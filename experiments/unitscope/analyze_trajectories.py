from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DISPLAY = {
    "stable-local-fallback": "Fallback",
    "handle-only-drop-residual": "Handle-Only",
    "unitscope-equal": "UnitScope-Equal",
    "unitscope-tuned": "UnitScope-Tuned",
    "unitscope-lambda-sweep": "UnitScope-Sweep",
    "naive-recalibrated-2c": "Naive-2C",
    "full-identity-reference": "Full Identity",
    "uldp-avg-weighted-oracle": "ULDP-FL AVG-w",
    "group-privacy-k2-person-corrected": "Group-2",
    "record-level-dp-sgd-reference": "Record-DP",
    "non-private-reference": "Non-private",
}


def load(directory_paths: list[Path]) -> pd.DataFrame:
    rows = []
    for directory in directory_paths:
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            lambda_value = payload.get("lambda")
            variant = str(payload["method"])
            if lambda_value is not None:
                variant = f"{variant}[lambda={float(lambda_value):g}]"
            for point in payload["trajectory"]:
                metric_name = (
                    "patient_or_user_auprc"
                    if point.get("patient_or_user_auprc") is not None
                    else "user_macro_accuracy"
                )
                rows.append(
                    {
                        "run_set": directory.parent.name,
                        "dataset": payload["dataset"],
                        "seed": int(payload["seed"]),
                        "method": payload["method"],
                        "method_variant": variant,
                        "lambda": lambda_value,
                        "round": int(point["round"]),
                        "epsilon": point.get("epsilon"),
                        "metric_name": metric_name,
                        "metric_value": point[metric_name],
                    }
                )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            [
                "run_set",
                "dataset",
                "method",
                "method_variant",
                "lambda",
                "round",
                "metric_name",
            ],
            dropna=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            epsilon_mean=("epsilon", "mean"),
            metric_mean=("metric_value", "mean"),
            metric_std=("metric_value", "std"),
        )
        .reset_index()
    )


def _label(method: str, lambda_value) -> str:
    base = DISPLAY.get(method, method)
    if method == "unitscope-lambda-sweep" and pd.notna(lambda_value):
        return f"{base} lambda={float(lambda_value):g}"
    return base


def plot(summary: pd.DataFrame, output: Path) -> None:
    for (run_set, dataset), group in summary.groupby(
        ["run_set", "dataset"], dropna=False
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
        for (_, variant), method_frame in group.groupby(
            ["method", "method_variant"], dropna=False
        ):
            method_frame = method_frame.sort_values("round")
            method = str(method_frame.iloc[0]["method"])
            label = _label(method, method_frame.iloc[0]["lambda"])
            axes[0].plot(
                method_frame["round"],
                method_frame["metric_mean"],
                marker="o",
                label=label,
            )
            valid = method_frame[method_frame["epsilon_mean"].notna()]
            if not valid.empty:
                axes[1].plot(
                    valid["epsilon_mean"], valid["metric_mean"], marker="o", label=label
                )
        axes[0].set_xlabel("Training round")
        axes[0].set_ylabel(
            "Writer accuracy" if dataset == "femnist" else "Patient AUPRC"
        )
        axes[0].set_title("Convergence")
        axes[1].set_xlabel("Accumulated epsilon")
        axes[1].set_ylabel(
            "Writer accuracy" if dataset == "femnist" else "Patient AUPRC"
        )
        axes[1].set_title("Privacy utility trajectory")
        for axis in axes:
            axis.grid(alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, max(1, len(labels))),
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        fig.savefig(output / f"{run_set}_{dataset}_trajectory.png", dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize UnitScope convergence and privacy trajectories"
    )
    parser.add_argument("--trajectory-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = load(args.trajectory_dirs)
    if frame.empty:
        raise ValueError("no trajectory files found")
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "trajectory_long.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(args.output / "trajectory_summary.csv", index=False)
    plot(summary, args.output)
    print(
        json.dumps(
            {"rows": len(frame), "summary_rows": len(summary)}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
