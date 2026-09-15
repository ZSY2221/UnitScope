"""Sweep H, B, local-unit semantics, and epsilon on bounded vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

from .accounting import calibrate_noise_multiplier, conservative_gaussian_rdp
from .model import CertificateType, CertifiedSensitivity, MethodId, PublicConfig
from .stats import paired_seed_bootstrap
from .vector_experiment import run_configuration


METHODS = (
    MethodId.STABLE_LOCAL_FALLBACK.value,
    MethodId.UNITSCOPE_TUNED.value,
    MethodId.HANDLE_ONLY.value,
)

METHOD_LABELS = {
    MethodId.STABLE_LOCAL_FALLBACK.value: "Stable local fallback",
    MethodId.UNITSCOPE_TUNED.value: "UnitScope",
    MethodId.HANDLE_ONLY.value: "Handle only",
}

COLORS = {
    MethodId.STABLE_LOCAL_FALLBACK.value: "#7A7A7A",
    MethodId.UNITSCOPE_TUNED.value: "#3B6FB6",
    MethodId.HANDLE_ONLY.value: "#C4A646",
}

MARKERS = {
    MethodId.STABLE_LOCAL_FALLBACK.value: "s",
    MethodId.UNITSCOPE_TUNED.value: "o",
    MethodId.HANDLE_ONLY.value: "^",
}

LINESTYLES = {
    MethodId.STABLE_LOCAL_FALLBACK.value: "--",
    MethodId.UNITSCOPE_TUNED.value: "-",
    MethodId.HANDLE_ONLY.value: "-.",
}


def _stable_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cells() -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []

    # Fragmentation scan.  Four records per silo ensure that H=4 is exercised
    # by both one-silo and two-silo users at 75% record coverage.
    for H in (1, 2, 4):
        cells.append(
            {
                "stage": "handle_multiplicity",
                "H": H,
                "B": 6,
                "lambda": 0.75,
                "target_epsilon": 8.0,
                "mixed_local_slots": False,
                "q_user": 0.75,
                "q_record": 0.75,
                "overlap": 0.75,
                "correlation": 0.5,
                "records_per_silo": 4,
                "handles_per_user": H,
            }
        )

    # B and mixed-unit scan.  The topology reaches at most two institution-local
    # slots per person.  B > 2 therefore measures the cost of a conservative
    # public upper bound rather than a change in the realized population.
    for B in (2, 4, 6, 8):
        for mixed in (False, True):
            cells.append(
                {
                    "stage": "fallback_bound",
                    "H": 2,
                    "B": B,
                    "lambda": 0.75,
                    "target_epsilon": 8.0,
                    "mixed_local_slots": mixed,
                    "q_user": 0.75,
                    "q_record": 0.50,
                    "overlap": 0.75,
                    "correlation": 0.5,
                    "records_per_silo": 2,
                    "handles_per_user": 2,
                }
            )

    # One-release privacy scan.  All methods in the main comparison have a
    # certified sensitivity C and therefore share the calibrated multiplier.
    for epsilon in (1.0, 2.0, 4.0, 8.0, 16.0):
        cells.append(
            {
                "stage": "privacy_level",
                "H": 2,
                "B": 4,
                "lambda": 0.75,
                "target_epsilon": epsilon,
                "mixed_local_slots": False,
                "q_user": 0.75,
                "q_record": 0.50,
                "overlap": 0.75,
                "correlation": 0.5,
                "records_per_silo": 2,
                "handles_per_user": 2,
            }
        )
    return cells


def _mean_interval(values: Iterable[float], *, seed: int) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        raise ValueError("a mean interval needs at least two finite seeds")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(20_000, len(array)))
    draws = array[indices].mean(axis=1)
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return float(array.mean()), float(lower), float(upper)


def _run_cell(
    cell: dict[str, object],
    *,
    output: Path,
    seeds: list[int],
    n_users: int,
    dimension: int,
    n_silos: int,
    delta: float,
) -> Path:
    specification = {
        **cell,
        "seeds": seeds,
        "n_users": n_users,
        "dimension": dimension,
        "n_silos": n_silos,
        "delta": delta,
        "query": "one-release fixed weighted sum of bounded vectors",
    }
    cell_hash = _stable_hash(specification)
    destination = output / "runs" / f"{cell_hash}.csv"
    if destination.exists():
        return destination

    multiplier = calibrate_noise_multiplier(
        float(cell["target_epsilon"]), rounds=1, delta=delta
    )
    config = PublicConfig(
        task_domain=f"contract-sweep-{cell['stage']}",
        epoch="frozen-contract-sweep-v1",
        clip_norm=1.0,
        max_handle_groups=int(cell["H"]),
        max_fallback_slots=int(cell["B"]),
        lambda_=float(cell["lambda"]),
        public_normalizer=float(n_users),
        noise_multiplier=multiplier,
        rounds=1,
        delta=delta,
        n_silos=n_silos,
    )
    certified = CertifiedSensitivity(1.0, CertificateType.PLAN_C, "contract sweep")
    achieved = conservative_gaussian_rdp(
        certified, multiplier, rounds=1, delta=delta
    ).epsilon
    frames = []
    for seed in seeds:
        frame = run_configuration(
            config,
            n_users=n_users,
            dimension=dimension,
            overlap_probability=float(cell["overlap"]),
            gradient_correlation=float(cell["correlation"]),
            vector_norm=0.6,
            q_user=float(cell["q_user"]),
            q_record_given_handle=float(cell["q_record"]),
            handles_per_user=int(cell["handles_per_user"]),
            records_per_silo=int(cell["records_per_silo"]),
            seed=seed,
            mixed_local_slots=bool(cell["mixed_local_slots"]),
        )
        frame["stage"] = str(cell["stage"])
        frame["cell_hash"] = cell_hash
        frame["target_epsilon"] = float(cell["target_epsilon"])
        frame["policy_lambda"] = float(cell["lambda"])
        frame["achieved_epsilon"] = float(achieved)
        frame["noise_multiplier"] = float(multiplier)
        frame["records_per_silo"] = int(cell["records_per_silo"])
        frame["handles_per_user"] = int(cell["handles_per_user"])
        frame["c_partial"] = float(cell["lambda"]) / int(cell["H"])
        frame["c_local_atom"] = (1.0 - float(cell["lambda"])) / int(cell["B"])
        frame["c_local_mixed"] = (1.0 - float(cell["lambda"])) / (2 * int(cell["B"]))
        frames.append(frame)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(destination, index=False)
    return destination


def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = raw[raw["method"].isin(METHODS)].copy()
    group_columns = [
        "stage",
        "H",
        "B",
        "mixed_local_slots",
        "target_epsilon",
        "policy_lambda",
        "method",
    ]
    rows: list[dict[str, object]] = []
    for index, (keys, group) in enumerate(
        selected.groupby(group_columns, sort=True, dropna=False)
    ):
        row = dict(zip(group_columns, keys))
        for metric in ("deterministic_distortion", "noisy_aggregate_mse"):
            mean, lower, upper = _mean_interval(group[metric], seed=20260717 + index)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
        row.update(
            {
                "seeds": int(group["seed"].nunique()),
                "noise_multiplier": float(group["noise_multiplier"].iloc[0]),
                "achieved_epsilon": float(group["achieved_epsilon"].iloc[0]),
                "realized_overlap_mean": float(group["realized_overlap"].mean()),
                "realized_handle_user_fraction_mean": float(
                    group["realized_q_user"].mean()
                ),
                "realized_record_coverage_mean": float(
                    group["realized_q_record_given_handle"].mean()
                ),
                "partial_handle_units_mean": float(
                    group["partial_handle_units"].mean()
                ),
                "local_atom_units_mean": float(group["local_atom_units"].mean()),
                "local_mixed_units_mean": float(group["local_mixed_units"].mean()),
                "c_partial": float(group["c_partial"].iloc[0]),
                "c_local_atom": float(group["c_local_atom"].iloc[0]),
                "c_local_mixed": float(group["c_local_mixed"].iloc[0]),
            }
        )
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)

    paired_rows: list[dict[str, object]] = []
    cell_columns = [
        "stage",
        "H",
        "B",
        "mixed_local_slots",
        "target_epsilon",
        "policy_lambda",
    ]
    for index, (keys, group) in enumerate(
        selected.groupby(cell_columns, sort=True, dropna=False)
    ):
        pivot_d = group.pivot(
            index="seed", columns="method", values="deterministic_distortion"
        )
        pivot_n = group.pivot(
            index="seed", columns="method", values="noisy_aggregate_mse"
        )
        if not set(METHODS).issubset(pivot_d.columns) or not set(METHODS).issubset(
            pivot_n.columns
        ):
            continue
        row = dict(zip(cell_columns, keys))
        for label, pivot in (("deterministic", pivot_d), ("noisy", pivot_n)):
            # Positive values mean that UnitScope has smaller MSE.
            for comparator in (
                MethodId.STABLE_LOCAL_FALLBACK.value,
                MethodId.HANDLE_ONLY.value,
            ):
                differences = (
                    pivot[comparator] - pivot[MethodId.UNITSCOPE_TUNED.value]
                ).to_numpy(dtype=np.float64)
                interval = paired_seed_bootstrap(
                    differences, n_resamples=20_000, seed=20260817 + index
                )
                short = (
                    "fallback"
                    if comparator == MethodId.STABLE_LOCAL_FALLBACK.value
                    else "handle"
                )
                row[f"{label}_improvement_vs_{short}_mean"] = interval.estimate
                row[f"{label}_improvement_vs_{short}_ci_lower"] = interval.lower
                row[f"{label}_improvement_vs_{short}_ci_upper"] = interval.upper
        row["paired_seeds"] = int(len(pivot_d))
        paired_rows.append(row)
    paired = pd.DataFrame(paired_rows).sort_values(cell_columns).reset_index(drop=True)
    return summary, paired


def _paper_style() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 7.5,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.7,
            "axes.grid": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "grid.color": "#E2E5E8",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.55,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.handletextpad": 0.45,
            "legend.columnspacing": 0.9,
            "legend.borderaxespad": 0.25,
            "legend.labelspacing": 0.25,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.4,
            "lines.markeredgewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "savefig.bbox": "tight",
        }
    )


def _line(
    axis,
    frame: pd.DataFrame,
    x_column: str,
    method: str,
    metric: str,
    *,
    label: str | None = None,
    color: str | None = None,
    linestyle: str = "-",
) -> None:
    frame = frame.sort_values(x_column)
    x = frame[x_column].to_numpy(dtype=np.float64)
    y = frame[f"{metric}_mean"].to_numpy(dtype=np.float64)
    lower = frame[f"{metric}_ci_lower"].to_numpy(dtype=np.float64)
    upper = frame[f"{metric}_ci_upper"].to_numpy(dtype=np.float64)
    chosen_color = color or COLORS[method]
    proposed = method == MethodId.UNITSCOPE_TUNED.value
    axis.plot(
        x,
        y,
        marker=MARKERS.get(method, "o"),
        markersize=3.8 if proposed else 3.2,
        linewidth=1.7 if proposed else 1.2,
        linestyle=linestyle if linestyle != "-" else LINESTYLES.get(method, "-"),
        color=chosen_color,
        markerfacecolor="white",
        markeredgecolor=chosen_color,
        markeredgewidth=0.6,
        label=label or METHOD_LABELS[method],
        zorder=5 if proposed else 3,
    )
    axis.fill_between(x, lower, upper, color=chosen_color, alpha=0.12, linewidth=0)


def plot(summary: pd.DataFrame, output: Path) -> None:
    _paper_style()
    # A three-panel, double-column figure should read as a shallow strip.
    # Keeping each axis nearly square made the composite dominate the page.
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.75), constrained_layout=True)

    handle = summary[summary["stage"] == "handle_multiplicity"]
    for method in METHODS:
        _line(
            axes[0],
            handle[handle["method"] == method],
            "H",
            method,
            "deterministic_distortion",
        )
    axes[0].set_yscale("log")
    axes[0].set_xticks([1, 2, 4])
    axes[0].set_xlabel("Handle bound $H$")
    axes[0].set_ylabel("Deterministic MSE")
    axes[0].grid(axis="y", which="major", color="#E2E5E8", linewidth=0.45, alpha=0.55)

    fallback = summary[
        (summary["stage"] == "fallback_bound")
        & (summary["method"] == MethodId.UNITSCOPE_TUNED.value)
    ]
    atom = fallback[~fallback["mixed_local_slots"].astype(bool)]
    mixed = fallback[fallback["mixed_local_slots"].astype(bool)]
    _line(
        axes[1],
        atom,
        "B",
        MethodId.UNITSCOPE_TUNED.value,
        "deterministic_distortion",
        label="Pure local units",
        color="#3B6FB6",
    )
    _line(
        axes[1],
        mixed,
        "B",
        MethodId.UNITSCOPE_TUNED.value,
        "deterministic_distortion",
        label="Mixed local units",
        color="#D17C2F",
        linestyle="--",
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks([2, 4, 6, 8])
    axes[1].set_xlabel("Fallback bound $B$")
    axes[1].grid(axis="y", which="major", color="#E2E5E8", linewidth=0.45, alpha=0.55)
    axes[1].legend(frameon=False, loc="best")

    privacy = summary[summary["stage"] == "privacy_level"]
    for method in METHODS:
        _line(
            axes[2],
            privacy[privacy["method"] == method],
            "target_epsilon",
            method,
            "noisy_aggregate_mse",
        )
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log")
    axes[2].set_xticks([1, 2, 4, 8, 16], labels=["1", "2", "4", "8", "16"])
    axes[2].set_xlabel(r"One-release $\epsilon$ ($\delta=10^{-6}$)")
    axes[2].grid(axis="y", which="major", color="#E2E5E8", linewidth=0.45, alpha=0.55)
    axes[2].legend(frameon=False, loc="upper right", fontsize=6.1)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    raw_panel_bboxes = [
        Bbox.union([axis.get_tightbbox(renderer)])
        .transformed(fig.dpi_scale_trans.inverted())
        .padded(0.025)
        for axis in axes
    ]
    # Export every panel on the same physical canvas.  Cropping each axis to
    # its own tight box made panels with wider tick labels appear at different
    # sizes when LaTeX scaled them to a common width.
    panel_width = max(bbox.width for bbox in raw_panel_bboxes)
    panel_height = max(bbox.height for bbox in raw_panel_bboxes)
    panel_bboxes = [
        Bbox.from_bounds(
            bbox.x0 - (panel_width - bbox.width) / 2,
            bbox.y0 - (panel_height - bbox.height) / 2,
            panel_width,
            panel_height,
        )
        for bbox in raw_panel_bboxes
    ]
    layout_engine = fig.get_layout_engine()
    fig.set_layout_engine("none")
    try:
        for suffix_name, axis, bbox_inches in zip(("a", "b", "c"), axes, panel_bboxes):
            visibility = [(other, other.get_visible()) for other in fig.axes]
            try:
                for other, _ in visibility:
                    other.set_visible(other is axis)
                for file_suffix in ("pdf", "png"):
                    fig.savefig(
                        output
                        / f"contract_parameter_sweep_{suffix_name}.{file_suffix}",
                        dpi=300,
                        bbox_inches=bbox_inches,
                    )
            finally:
                for other, was_visible in visibility:
                    other.set_visible(was_visible)
    finally:
        fig.set_layout_engine(layout_engine)
    for suffix in ("pdf", "png"):
        fig.savefig(
            output / f"contract_parameter_sweep.{suffix}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.025,
        )
    plt.close(fig)


def write_readme(
    output: Path,
    *,
    args: argparse.Namespace,
    cells: list[dict[str, object]],
    raw: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    privacy = summary[
        (summary["stage"] == "privacy_level")
        & (summary["method"] == MethodId.UNITSCOPE_TUNED.value)
    ].sort_values("target_epsilon")
    h_rows = summary[
        (summary["stage"] == "handle_multiplicity")
        & (summary["method"] == MethodId.UNITSCOPE_TUNED.value)
    ].sort_values("H")
    b_rows = summary[
        (summary["stage"] == "fallback_bound")
        & (summary["method"] == MethodId.UNITSCOPE_TUNED.value)
    ].sort_values(["mixed_local_slots", "B"])
    lines = [
        "# UnitScope contract-parameter sweep v1",
        "",
        "## Scope",
        "",
        "This is a mechanism-level bounded-vector experiment, not an end-to-end federated-learning "
        "accuracy experiment. It isolates H, B, mixed-local-unit charging, and calibrated Gaussian "
        "noise while holding the synthetic population generator and paired seeds fixed.",
        "",
        "## Frozen design",
        "",
        f"- People: {args.n_users}",
        f"- Vector dimension: {args.dimension}",
        f"- Institutions: {args.n_silos}",
        f"- Paired seeds: {args.seeds}",
        f"- Neighboring-person delta: {args.delta:g}",
        "- Clip norm C: 1",
        "- Query: one fixed weighted sum of bounded vectors",
        "- Confidence intervals: marginal 95% seed-bootstrap intervals; paired differences are in paired_effects.csv",
        "- No test-set or outcome-dependent parameter selection is performed",
        "",
        "## Outputs",
        "",
        "- raw_results.csv contains every method, cell, and seed.",
        "- summary.csv contains marginal seed-bootstrap intervals.",
        "- paired_effects.csv contains paired UnitScope improvements; positive means lower MSE than the comparator.",
        "- contract_parameter_sweep.pdf and .png are publication-sized three-panel figures.",
        "- manifest.json freezes the environment and cell specifications.",
        "",
        "## Observed ranges",
        "",
        f"- H scan UnitScope deterministic MSE: {h_rows['deterministic_distortion_mean'].min():.6g} to "
        f"{h_rows['deterministic_distortion_mean'].max():.6g}.",
        f"- B/mixed scan UnitScope deterministic MSE: {b_rows['deterministic_distortion_mean'].min():.6g} to "
        f"{b_rows['deterministic_distortion_mean'].max():.6g}.",
        f"- Epsilon scan UnitScope noisy MSE: {privacy['noisy_aggregate_mse_mean'].min():.6g} to "
        f"{privacy['noisy_aggregate_mse_mean'].max():.6g}.",
        "",
        "## Interpretation limits",
        "",
        "- H and B are public admissible-domain bounds. A larger value reserves capacity even when a particular "
        "sample realizes fewer paths.",
        "- The mixed-unit branch assigns LocalKind.MIXED and therefore uses the theorem's factor-two radius. "
        "The generated slots remain user-local so that the experiment isolates charging; it is not an entity-resolution contamination model.",
        "- Privacy levels are conservative one-release RDP conversions without sampling amplification. They are not full multi-round training budgets.",
        "- Deterministic distortion is squared distance from the complete-identity clipped aggregate. No value in this directory is model accuracy.",
        "- Marginal confidence bands should not be read as paired significance tests. Use paired_effects.csv for paired comparisons.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the UnitScope contract-parameter mechanism sweep"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-users", type=int, default=1000)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--n-silos", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.force and (args.output / "runs").exists():
        import shutil

        shutil.rmtree(args.output / "runs")

    cells = _cells()
    paths = []
    for index, cell in enumerate(cells, start=1):
        path = _run_cell(
            cell,
            output=args.output,
            seeds=args.seeds,
            n_users=args.n_users,
            dimension=args.dimension,
            n_silos=args.n_silos,
            delta=args.delta,
        )
        paths.append(path)
        print(
            json.dumps({"cell": index, "total": len(cells), "path": str(path)}),
            flush=True,
        )

    raw = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    raw.to_csv(args.output / "raw_results.csv", index=False)
    summary, paired = summarize(raw)
    summary.to_csv(args.output / "summary.csv", index=False)
    paired.to_csv(args.output / "paired_effects.csv", index=False)
    plot(summary, args.output)

    manifest = {
        "experiment": "contract_parameter_sweep_v1",
        "warning": "bounded-vector mechanism study; not trained-model accuracy",
        "arguments": {
            "n_users": args.n_users,
            "dimension": args.dimension,
            "n_silos": args.n_silos,
            "seeds": args.seeds,
            "delta": args.delta,
        },
        "cells": cells,
        "cell_count": len(cells),
        "raw_rows": len(raw),
        "raw_sha256": hashlib.sha256(
            (args.output / "raw_results.csv").read_bytes()
        ).hexdigest(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": mpl.__version__,
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_readme(args.output, args=args, cells=cells, raw=raw, summary=summary)
    print(
        json.dumps({"rows": len(raw), "output": str(args.output)}, indent=2), flush=True
    )


if __name__ == "__main__":
    main()
