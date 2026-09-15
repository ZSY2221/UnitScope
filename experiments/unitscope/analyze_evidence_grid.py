from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from .stats import paired_ratio_bootstrap, paired_seed_bootstrap


def _load(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not rows:
        raise ValueError("no evidence-grid rows found")
    return pd.DataFrame(rows)


def _bootstrap_mean(values: np.ndarray, seed: int = 20260715) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(20_000, len(values)))
    draws = values[indices].mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(lower), float(upper)


def assemble(
    grid: pd.DataFrame,
    main: pd.DataFrame,
    central_sweep: pd.DataFrame,
) -> pd.DataFrame:
    central_handle = main[
        (main["dataset"] == "femnist")
        & (main["method"] == "handle-only-drop-residual")
        & (main["seed"] < 5)
    ].copy()
    central_unit = central_sweep[
        (central_sweep["dataset"] == "femnist")
        & (central_sweep["method"] == "unitscope-lambda-sweep")
        & (central_sweep["seed"] < 5)
    ].copy()
    frame = pd.concat([grid, central_handle, central_unit], ignore_index=True)
    frame["q_user_cell"] = pd.to_numeric(frame["q_user_requested"], errors="raise")
    frame["q_record_cell"] = pd.to_numeric(
        frame["q_record_given_handle_requested"], errors="raise"
    )
    expected = {
        (q_user, q_record, seed, method)
        for q_user in (0.25, 0.5, 0.75)
        for q_record in (0.25, 0.5, 0.75)
        for seed in range(5)
        for method in ("handle-only-drop-residual", "unitscope-lambda-sweep")
    }
    observed = {
        (
            float(row.q_user_cell),
            float(row.q_record_cell),
            int(row.seed),
            str(row.method),
        )
        for row in frame.itertuples(index=False)
    }
    if observed != expected:
        raise ValueError(
            f"incomplete evidence grid: missing={len(expected - observed)}, "
            f"unexpected={len(observed - expected)}"
        )
    return frame


def analyze(frame: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    fallback = main[
        (main["dataset"] == "femnist")
        & (main["method"] == "stable-local-fallback")
        & (main["seed"] < 5)
    ].set_index("seed")["metric_value"]
    rows = []
    for (q_user, q_record), group in frame.groupby(["q_user_cell", "q_record_cell"]):
        pivot = group.pivot(index="seed", columns="method", values="metric_value").join(
            fallback.rename("fallback"), how="inner"
        )
        unit = pivot["unitscope-lambda-sweep"].to_numpy(dtype=np.float64)
        handle = pivot["handle-only-drop-residual"].to_numpy(dtype=np.float64)
        fallback_values = pivot["fallback"].to_numpy(dtype=np.float64)
        unit_lower, unit_upper = _bootstrap_mean(unit)
        handle_lower, handle_upper = _bootstrap_mean(handle)
        gain_fallback = paired_seed_bootstrap(unit - fallback_values)
        difference_handle = paired_seed_bootstrap(unit - handle)
        handle_gap = handle - fallback_values
        row: dict[str, object] = {
            "q_user": float(q_user),
            "q_record_given_handle": float(q_record),
            "paired_seeds": len(pivot),
            "unit_metric_mean": float(unit.mean()),
            "unit_metric_ci_lower": unit_lower,
            "unit_metric_ci_upper": unit_upper,
            "handle_metric_mean": float(handle.mean()),
            "handle_metric_ci_lower": handle_lower,
            "handle_metric_ci_upper": handle_upper,
            "gain_over_fallback": gain_fallback.estimate,
            "gain_over_fallback_ci_lower": gain_fallback.lower,
            "gain_over_fallback_ci_upper": gain_fallback.upper,
            "difference_from_handle_only": difference_handle.estimate,
            "difference_from_handle_only_ci_lower": difference_handle.lower,
            "difference_from_handle_only_ci_upper": difference_handle.upper,
            "unit_record_utilization": float(
                group[group["method"] == "unitscope-lambda-sweep"][
                    "effective_record_utilization"
                ].mean()
            ),
            "handle_record_utilization": float(
                group[group["method"] == "handle-only-drop-residual"][
                    "effective_record_utilization"
                ].mean()
            ),
            "realized_q_user": float(group["q_user"].mean()),
            "realized_q_record_given_handle": float(
                group["q_record_given_handle"].mean()
            ),
            "descriptive_positive_region": bool(gain_fallback.lower > 0),
        }
        if float(handle_gap.mean()) > 0.005:
            try:
                retention = paired_ratio_bootstrap(unit - fallback_values, handle_gap)
                row["handle_gain_retention"] = retention.estimate
                row["handle_gain_retention_ci_lower"] = retention.lower
                row["handle_gain_retention_ci_upper"] = retention.upper
            except ValueError:
                row["handle_gain_retention"] = np.nan
                row["handle_gain_retention_ci_lower"] = np.nan
                row["handle_gain_retention_ci_upper"] = np.nan
        else:
            row["handle_gain_retention"] = np.nan
            row["handle_gain_retention_ci_lower"] = np.nan
            row["handle_gain_retention_ci_upper"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["q_user", "q_record_given_handle"])


def _matrix(
    summary: pd.DataFrame, column: str
) -> tuple[np.ndarray, list[float], list[float]]:
    q_users = sorted(summary["q_user"].unique().tolist())
    q_records = sorted(summary["q_record_given_handle"].unique().tolist())
    pivot = summary.pivot(
        index="q_user", columns="q_record_given_handle", values=column
    )
    return pivot.loc[q_users, q_records].to_numpy(dtype=np.float64), q_users, q_records


def plot(summary: pd.DataFrame, output: Path) -> None:
    panels = [
        ("gain_over_fallback", "UnitScope minus Fallback", "RdBu_r", True),
        ("difference_from_handle_only", "UnitScope minus Handle-Only", "RdBu_r", True),
        (
            "handle_record_utilization",
            "Handle-Only record utilization",
            "viridis",
            False,
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    for axis, (column, title, cmap, centered) in zip(axes, panels):
        matrix, q_users, q_records = _matrix(summary, column)
        if centered:
            bound = max(abs(float(matrix.min())), abs(float(matrix.max())), 1e-6)
            image = axis.imshow(
                matrix, cmap=cmap, norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
            )
        else:
            image = axis.imshow(matrix, cmap=cmap, vmin=0, vmax=1)
        for row in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row, column_index]
                label = f"{value:+.3f}" if centered else f"{100 * value:.1f}%"
                axis.text(
                    column_index, row, label, ha="center", va="center", fontsize=9
                )
        axis.set_xticks(range(len(q_records)), [f"{value:.2f}" for value in q_records])
        axis.set_yticks(range(len(q_users)), [f"{value:.2f}" for value in q_users])
        axis.set_xlabel("Within-handle record coverage")
        axis.set_ylabel("Handle-user coverage")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Evidence-value grid at lambda=0.9 and fixed person-level privacy")
    fig.tight_layout()
    fig.savefig(output / "evidence_value_grid.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "evidence_value_grid.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, output: Path) -> None:
    positive = summary[summary["descriptive_positive_region"]]
    lines = [
        "# UnitScope evidence-value grid",
        "",
        "The grid fixes lambda=0.9 and varies two distinct evidence dimensions. All cells use five "
        "paired seeds, the same 1,000 FEMNIST writers and the same person-level privacy target. The "
        "grid is descriptive and retains unfavorable cells.",
        "",
        f"The paired 95% interval for UnitScope minus Fallback is strictly positive in "
        f"{len(positive)} of {len(summary)} cells.",
        "",
        "| Handle users | Within-handle records | UnitScope | vs Fallback | vs Handle-Only | "
        "Handle utilization |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.q_user:.2f} | {row.q_record_given_handle:.2f} | {row.unit_metric_mean:.4f} | "
            f"{row.gain_over_fallback:+.4f} | {row.difference_from_handle_only:+.4f} | "
            f"{100 * row.handle_record_utilization:.1f}% |"
        )
    lines.extend(
        [
            "",
            "Sparse evidence favors the lambda=0 fallback. Denser evidence approaches Handle-Only "
            "while retaining residual records.",
            "",
        ]
    )
    (output / "EVIDENCE_GRID_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the fixed UnitScope evidence-value grid"
    )
    parser.add_argument("--grid-inputs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--main-inputs", "--main-input", dest="main_inputs", type=Path, nargs="+", required=True
    )
    parser.add_argument("--central-sweep-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grid = _load(args.grid_inputs)
    main_frame = _load(args.main_inputs)
    central_sweep = _load([args.central_sweep_input])
    assembled = assemble(grid, main_frame, central_sweep)
    summary = analyze(assembled, main_frame)
    args.output.mkdir(parents=True, exist_ok=True)
    assembled.to_csv(args.output / "evidence_grid_long.csv", index=False)
    summary.to_csv(args.output / "evidence_grid_summary.csv", index=False)
    plot(summary, args.output)
    write_report(summary, args.output)
    print(
        json.dumps(
            {
                "cells": len(summary),
                "positive_cells": int(summary["descriptive_positive_region"].sum()),
                "seeds_per_cell": sorted(summary["paired_seeds"].unique().tolist()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
