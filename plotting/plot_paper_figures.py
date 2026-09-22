"""Build the paper figures from experiment result files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
PAPER_ROOT = HERE.parents[1]
RESULTS_ROOT = PAPER_ROOT / "results"
FALLBACK_RESULTS_ROOT: Path | None = None
RESULTS = RESULTS_ROOT / "mechanism_studies"
FIGURES = PAPER_ROOT / "figures"
USED_RESULT_ROOTS: set[Path] = set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the UnitScope paper figures")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--fallback-results-root", type=Path)
    parser.add_argument("--figures-root", type=Path, default=FIGURES)
    return parser.parse_args()


def configure_paths(
    results_root: Path,
    figures_root: Path,
    fallback_results_root: Path | None = None,
) -> None:
    global RESULTS_ROOT, FALLBACK_RESULTS_ROOT, RESULTS, FIGURES
    RESULTS_ROOT = results_root.resolve()
    FALLBACK_RESULTS_ROOT = (
        fallback_results_root.resolve() if fallback_results_root is not None else None
    )
    RESULTS = RESULTS_ROOT / "mechanism_studies"
    FIGURES = figures_root.resolve()
    FIGURES.mkdir(parents=True, exist_ok=True)
    USED_RESULT_ROOTS.clear()


def result_path(*parts: str) -> Path:
    relative = Path(*parts)
    candidates = [RESULTS_ROOT / relative]
    aliases = {
        "main": "main",
        "reference_methods": "references",
        "learning_extensions": "extensions",
        "movielens_task": "extensions",
    }
    if relative.parts and relative.parts[0] in aliases:
        candidates.append(
            RESULTS_ROOT
            / aliases[relative.parts[0]]
            / "analysis"
            / Path(*relative.parts[1:])
        )
    if FALLBACK_RESULTS_ROOT is not None:
        candidates.append(FALLBACK_RESULTS_ROOT / relative)
    for candidate in candidates:
        if candidate.is_file():
            USED_RESULT_ROOTS.add(candidate.parent)
            return candidate
    searched = "\n".join(f"  {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"result file not found; searched:\n{searched}")


COLORS = {
    "blue": "#3B6FB6",
    "sky": "#AFC7E3",
    "green": "#3A8D7C",
    "orange": "#D17C2F",
    "red": "#B24C4C",
    "purple": "#8B6FA8",
    "gray": "#7A7A7A",
    "gold": "#C4A646",
    "lightgray": "#ECEFF1",
    "ink": "#252A2E",
}

PLOT_COLORS = [
    COLORS["blue"],
    COLORS["orange"],
    COLORS["green"],
    COLORS["purple"],
    COLORS["red"],
    COLORS["gray"],
    COLORS["gold"],
]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 2)), (0, (1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
PVLDB_DIVERGING = LinearSegmentedColormap.from_list(
    "pvl_db_diverging", [COLORS["red"], "#F7F7F7", COLORS["blue"]]
)


def paper_style() -> None:
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
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
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
            "savefig.pad_inches": 0.025,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300)
    plt.close(fig)


def save_panels(
    fig: plt.Figure,
    panels: list[tuple[str, tuple[plt.Axes, ...]]],
    *,
    common_canvas: bool = True,
    fixed_bboxes: dict[str, Bbox] | None = None,
) -> None:
    """Save each panel on a common canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panel_bboxes = [
        fixed_bboxes.get(stem)
        if fixed_bboxes is not None and stem in fixed_bboxes
        else Bbox.union([axis.get_tightbbox(renderer) for axis in panel_axes])
        .transformed(fig.dpi_scale_trans.inverted())
        .padded(0.025)
        for stem, panel_axes in panels
    ]
    if common_canvas:
        # Keep side-by-side paper panels equal after tight cropping.
        common_width = max(box.width for box in panel_bboxes)
        common_height = max(box.height for box in panel_bboxes)
        panel_bboxes = [
            Bbox.from_bounds(
                0.5 * (box.x0 + box.x1 - common_width),
                0.5 * (box.y0 + box.y1 - common_height),
                common_width,
                common_height,
            )
            for box in panel_bboxes
        ]
    layout_engine = fig.get_layout_engine()
    fig.set_layout_engine("none")
    try:
        for (stem, panel_axes), bbox_inches in zip(panels, panel_bboxes):
            target_ids = {id(axis) for axis in panel_axes}
            visibility = [(axis, axis.get_visible()) for axis in fig.axes]
            try:
                for axis, _ in visibility:
                    axis.set_visible(id(axis) in target_ids)
                fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches=bbox_inches)
                fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches=bbox_inches)
            finally:
                for axis, was_visible in visibility:
                    axis.set_visible(was_visible)
    finally:
        fig.set_layout_engine(layout_engine)


def plot_method(ax, x, y, index, label, *, proposed=False, markevery=None):
    """Apply the shared PVLDB color, line, and marker grammar."""
    color = PLOT_COLORS[index % len(PLOT_COLORS)]
    return ax.plot(
        x,
        y,
        color=color,
        linestyle=LINESTYLES[index % len(LINESTYLES)],
        linewidth=1.7 if proposed else 1.2,
        marker=MARKERS[index % len(MARKERS)],
        markersize=3.8 if proposed else 3.2,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=0.6,
        markevery=markevery,
        label=label,
        zorder=5 if proposed else 3,
    )[0]


def light_horizontal_grid(ax) -> None:
    ax.grid(True, axis="y", which="major", color="#E2E5E8", linewidth=0.45, alpha=0.55)


def rounded(ax, xy, width, height, text, face, edge, *, fontsize=8, weight="normal"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=face,
        edgecolor=edge,
        linewidth=0.7,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        weight=weight,
    )
    return box


def arrow(ax, start, end, color=COLORS["gray"], lw=0.9):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def outer_frame(ax: plt.Axes) -> None:
    """Give diagram-style panels the same four-sided frame as numeric plots."""
    ax.add_patch(
        Rectangle(
            (0.004, 0.004),
            0.992,
            0.992,
            transform=ax.transAxes,
            fill=False,
            edgecolor=COLORS["gray"],
            linewidth=0.7,
            clip_on=False,
            zorder=20,
        )
    )


def partial_handle_scenario() -> None:
    """Page-one motivating example: correct matching, incomplete coverage."""
    fig, axes = plt.subplots(2, 1, figsize=(3.33, 2.45), constrained_layout=True)
    specs = [
        (
            COLORS["red"],
            r"A+B handle gets $C$",
            r"D residual gets $C$",
            r"one person receives $2C$",
        ),
        (
            COLORS["green"],
            r"A+B handle shares $\lambda C$",
            r"D residual shares $(1-\lambda)C$",
            r"compiled plan stays within $C$",
        ),
    ]
    for ax, (verdict_color, left_budget, right_budget, verdict) in zip(axes, specs):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        outer_frame(ax)
        rounded(
            ax,
            (0.02, 0.45),
            0.12,
            0.26,
            "A",
            "#DCEEF8",
            COLORS["blue"],
            fontsize=7.3,
            weight="bold",
        )
        rounded(
            ax,
            (0.18, 0.45),
            0.12,
            0.26,
            "B",
            "#DCEEF8",
            COLORS["blue"],
            fontsize=7.3,
            weight="bold",
        )
        rounded(
            ax,
            (0.86, 0.45),
            0.12,
            0.26,
            "D",
            "#EEF1F3",
            COLORS["gray"],
            fontsize=7.3,
            weight="bold",
        )
        ax.plot([0.08, 0.24], [0.39, 0.39], color=COLORS["blue"], lw=1.7)
        ax.text(
            0.16,
            0.27,
            left_budget,
            ha="center",
            va="top",
            fontsize=5.8,
            color=COLORS["blue"],
        )
        ax.text(
            0.92,
            0.27,
            right_budget,
            ha="center",
            va="top",
            fontsize=5.6,
            color=COLORS["gray"],
        )
        rounded(
            ax,
            (0.37, 0.43),
            0.41,
            0.30,
            verdict,
            "#FCECE7" if verdict_color == COLORS["red"] else "#E7F5EE",
            verdict_color,
            fontsize=6.2,
            weight="bold",
        )
    save_panels(
        fig,
        [
            ("fig0_partial_handle_scenario_a", (axes[0],)),
            ("fig0_partial_handle_scenario_b", (axes[1],)),
        ],
    )
    save(fig, "fig0_partial_handle_scenario")


def overview() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 2.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    outer_frame(ax)

    ax.text(0.105, 0.98, "Identity evidence", ha="center", va="top", weight="bold")
    y_positions = [0.73, 0.51, 0.29, 0.07]
    labels = [
        "Complete identity",
        "Partial handle",
        "Stable local boundary",
        "Inferred partition",
    ]
    fills = ["#DDF3EA", "#DCEEF8", "#EEF1F3", "#FCE9DF"]
    edges = [COLORS["green"], COLORS["blue"], COLORS["gray"], COLORS["red"]]
    for y, label, fill, edge in zip(y_positions, labels, fills, edges):
        rounded(ax, (0.008, y), 0.195, 0.16, label, fill, edge, fontsize=7.3)

    rounded(
        ax,
        (0.265, 0.08),
        0.345,
        0.80,
        "",
        "#F7FAFC",
        COLORS["blue"],
    )
    ax.text(
        0.4375,
        0.83,
        "UnitScope privacy-unit compiler",
        ha="center",
        va="center",
        weight="bold",
        fontsize=7.8,
    )
    rounded(
        ax,
        (0.300, 0.49),
        0.275,
        0.20,
        "Governed-evidence checks\n"
        + "quotas + capacity + stable routing\n"
        + r"certified envelope $C$",
        "#EAF4FB",
        COLORS["blue"],
        fontsize=6.35,
    )
    rounded(
        ax,
        (0.300, 0.20),
        0.275,
        0.20,
        "Data-dependent boundary check\n" + r"global WUR certificate $\Delta$",
        "#F4ECFA",
        COLORS["purple"],
        fontsize=6.8,
    )
    ax.text(
        0.4375,
        0.105,
        "two validation routes, one plan",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=COLORS["gray"],
    )

    ax.text(
        0.705, 0.94, "Compiled plan", ha="center", va="top", weight="bold", fontsize=7.7
    )
    rounded(
        ax,
        (0.645, 0.38),
        0.12,
        0.35,
        "units + radii\n" + r"bound $C$ or $\Delta$",
        "#EDE8F7",
        COLORS["purple"],
        fontsize=6.9,
        weight="bold",
    )
    ax.text(
        0.89, 0.94, "Existing ULDP", ha="center", va="top", weight="bold", fontsize=7.7
    )
    rounded(
        ax,
        (0.805, 0.30),
        0.17,
        0.43,
        "user clipping\n+ Gaussian noise\n+ accounting",
        "#E7F5EE",
        COLORS["green"],
        fontsize=7.1,
    )
    for y in [0.81, 0.59, 0.37]:
        arrow(ax, (0.203, y), (0.265, 0.59))
    arrow(ax, (0.203, 0.15), (0.265, 0.30), COLORS["red"])
    arrow(ax, (0.61, 0.50), (0.645, 0.555), COLORS["purple"])
    arrow(ax, (0.765, 0.555), (0.805, 0.515), COLORS["green"])
    save(fig, "fig1_unitscope_overview")


def boundary_illusion() -> None:
    pilot_rows = []
    selected = {
        "febrl1": (0.72, 2),
        "febrl2": (0.72, 8),
        "febrl3": (0.76, 8),
    }
    for dataset, (threshold, cap) in selected.items():
        source = pd.read_csv(
            result_path(
                "linkage_boundary",
                "pilot",
                dataset,
                "hardcap_empirical_recourse_summary.csv",
            )
        )
        row = source[
            np.isclose(source.threshold, threshold) & np.isclose(source.K, cap)
        ].iloc[0]
        pilot_rows.append(
            {
                "dataset": dataset.upper(),
                "f1": float(row.pair_f1),
                "wur": float(row.max_d_w_empirical),
            }
        )
    febrl = pd.DataFrame(pilot_rows)
    per = pd.read_csv(
        result_path(
            "linkage_boundary", "per", "per_febrl_all", "per_febrl_threshold_scan.csv"
        )
    )
    ncvr = pd.read_csv(
        result_path("mechanism_studies", "ncvr_boundary", "ncvr_combined.csv")
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.15, 2.2), constrained_layout=True)
    ax = axes[0]
    ax.axhspan(0, 1, color="#EDF5F2", alpha=0.55, zorder=0)
    ax.axhline(
        1,
        color=COLORS["green"],
        linewidth=1.0,
        linestyle="--",
        label=r"$\mathsf{Plan}\langle C\rangle$ limit",
    )
    ax.scatter(
        per["pair_f1"],
        per["max_witnessed_wur"],
        s=24,
        marker="o",
        facecolor="white",
        edgecolor=COLORS["blue"],
        linewidth=0.75,
        label="PER 2025 (PESM with TOP)",
        zorder=3,
    )
    ax.scatter(
        febrl["f1"],
        febrl["wur"],
        s=30,
        marker="D",
        color=COLORS["red"],
        edgecolor="white",
        linewidth=0.7,
        label="Link-then-DP pilot",
        zorder=4,
    )
    for _, row in febrl.iterrows():
        offsets = {"FEBRL1": (-38, 4), "FEBRL2": (4, 4), "FEBRL3": (-6, 4)}
        ax.annotate(
            row.dataset,
            (row.f1, row.wur),
            xytext=offsets[row.dataset],
            textcoords="offset points",
            ha="right" if row.dataset == "FEBRL3" else "left",
            fontsize=6.7,
        )
    ax.set_xlabel("Pairwise F1")
    ax.set_ylabel(r"Maximum witnessed WUR ($C$ units)")
    ax.set_xlim(0.78, 0.998)
    ax.set_ylim(0, 15.3)
    light_horizontal_grid(ax)
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    ax.axhspan(0, 1, color="#EDF5F2", alpha=0.55, zorder=0)
    ax.axhline(1, color=COLORS["green"], linewidth=1.0, linestyle="--")
    markers = {0.7: "o", 0.8: "s", 0.9: "^"}
    colors = {"county": COLORS["orange"], "statewide": COLORS["blue"]}
    for (scope, threshold), group in ncvr.groupby(["scope", "threshold"]):
        ax.scatter(
            group["pair_f1"],
            group["deletion_wur_max_over_c"],
            s=26,
            marker=markers[threshold],
            color=colors[scope],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
            label=f"{scope}, t={threshold:.1f}",
        )
    focus = ncvr[
        (ncvr.scope == "statewide") & (ncvr.people == 250000) & (ncvr.threshold == 0.8)
    ].iloc[0]
    ax.annotate(
        "250K people\nprecision 0.992",
        (focus.pair_f1, focus.deletion_wur_max_over_c),
        xytext=(-54, 16),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": COLORS["gray"], "lw": 0.8},
        fontsize=7.0,
    )
    ax.set_xlabel("Pairwise F1")
    ax.set_ylabel(r"Maximum witnessed WUR ($C$ units)")
    ax.set_xlim(0.64, 0.89)
    ax.set_ylim(0, max(8.7, ncvr.deletion_wur_max_over_c.max() + 0.7))
    light_horizontal_grid(ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        ncol=2,
        loc="upper left",
        frameon=False,
        handletextpad=0.2,
        columnspacing=0.6,
    )
    save_panels(
        fig,
        [
            ("fig2_boundary_illusion_a", (axes[0],)),
            ("fig2_boundary_illusion_b", (axes[1],)),
        ],
    )
    save(fig, "fig2_boundary_illusion")


def utility_comparison() -> None:
    methods = pd.read_csv(result_path("main", "method_summary.csv"))
    order = [
        ("stable-local-fallback", "Stable-Local", COLORS["gray"], "s"),
        (
            "unitscope-lambda-sweep",
            r"UnitScope-Fixed ($\lambda=.9$)",
            COLORS["blue"],
            "o",
        ),
        ("handle-only-drop-residual", "Handle-Only", COLORS["orange"], "^"),
        ("naive-recalibrated-2c", r"Naive-2C ($2C$)", COLORS["red"], "D"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(5.75, 2.15), constrained_layout=True)
    for ax, dataset in zip(axes, ["femnist", "synthea"]):
        subset = methods[methods.dataset == dataset]
        for method, label, color, marker in order:
            row = subset[subset.method == method].iloc[0]
            x = 100 * float(row.effective_record_utilization_mean)
            y = float(row.metric_mean)
            ax.errorbar(
                x,
                y,
                yerr=np.array([[y - row.metric_ci_lower], [row.metric_ci_upper - y]]),
                fmt=marker,
                color=color,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.75,
                markersize=4.4 if method == "unitscope-lambda-sweep" else 3.8,
                ecolor=color,
                capsize=2.0,
                elinewidth=0.8,
                linewidth=0.8,
                label=label,
                zorder=3,
            )
        ax.set_xlabel("Records used (%)")
        ax.set_ylabel(
            "Writer macro accuracy" if dataset == "femnist" else "Patient AUPRC"
        )
        ax.set_xlim(20, 104)
        ax.set_xticks([25, 40, 60, 80, 100])
        light_horizontal_grid(ax)
        ax.legend(loc="lower left", ncol=1, fontsize=6.1)
    save_panels(
        fig,
        [
            ("fig3_utility_comparison_a", (axes[0],)),
            ("fig3_utility_comparison_b", (axes[1],)),
        ],
    )
    save(fig, "fig3_utility_comparison")


def lambda_frontier() -> None:
    effects = pd.read_csv(
        result_path(
            "mechanism_studies", "lambda_frontier", "lambda_frontier_paired_effects.csv"
        )
    )
    fig, ax = plt.subplots(figsize=(3.33, 1.9), constrained_layout=True)
    for index, (dataset, label) in enumerate(
        (("femnist", "FEMNIST"), ("synthea", "Synthea"))
    ):
        group = effects[effects.dataset == dataset].sort_values("lambda")
        x = group["lambda"].to_numpy(dtype=float)
        mean = 100 * group.gain_over_fallback_mean.to_numpy(dtype=float)
        lower = 100 * group.gain_over_fallback_ci_lower.to_numpy(dtype=float)
        upper = 100 * group.gain_over_fallback_ci_upper.to_numpy(dtype=float)
        ax.fill_between(
            x, lower, upper, color=PLOT_COLORS[index], alpha=0.10, linewidth=0
        )
        plot_method(ax, x, mean, index, label, proposed=index == 0)
    ax.axhline(0, color=COLORS["gray"], linewidth=0.8)
    ax.axvline(0.9, color=COLORS["gray"], linewidth=0.8, linestyle=":")
    ax.text(
        0.885,
        ax.get_ylim()[1],
        r"replication $\lambda=.9$",
        ha="right",
        va="top",
        fontsize=6.0,
        color=COLORS["gray"],
    )
    ax.set_xlabel(r"Partial-handle allocation $\lambda$")
    ax.set_ylabel("Gain over Stable-Local (pp)")
    light_horizontal_grid(ax)
    ax.legend(loc="upper left")
    save(fig, "fig4_lambda_frontier")


def lambda_frontier_edbt() -> None:
    """Render the two-panel allocation frontier used in the paper."""
    frame = pd.read_csv(
        result_path(
            "mechanism_studies", "lambda_frontier", "lambda_frontier_summary.csv"
        )
    )
    methods = pd.read_csv(result_path("main", "method_summary.csv"))
    requested = {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}
    frame = frame[frame["lambda"].round(10).isin(requested)].copy()
    if len(frame) != 14:
        raise ValueError(f"expected 14 frozen frontier points, found {len(frame)}")

    fig, axes = plt.subplots(1, 2, figsize=(6.65, 1.78), sharex=True)
    colors = {"femnist": "#3569A8", "synthea": "#D17C2F"}
    titles = {"femnist": "FEMNIST", "synthea": "Synthea"}
    ylabels = {"femnist": "Writer macro accuracy", "synthea": "Patient AUPRC"}
    for ax, dataset in zip(axes, ("femnist", "synthea")):
        group = frame[frame.dataset == dataset].sort_values("lambda")
        x = group["lambda"].to_numpy(dtype=float)
        y = group.metric_mean.to_numpy(dtype=float)
        lower = group.metric_ci_lower.to_numpy(dtype=float)
        upper = group.metric_ci_upper.to_numpy(dtype=float)
        color = colors[dataset]
        ax.fill_between(x, lower, upper, color=color, alpha=0.14, linewidth=0)
        ax.plot(
            x, y, color=color, marker="o", markerfacecolor="white", markeredgewidth=0.75
        )
        selected = group[np.isclose(group["lambda"], 0.9)].iloc[0]
        ax.scatter(
            [0.9], [selected.metric_mean], s=27, color="#1E8A70", marker="D", zorder=5
        )
        ax.axvline(0.9, color="#777777", linestyle=":", linewidth=0.75)
        ax.set_title(titles[dataset], loc="left", fontsize=7.5, fontweight="bold")
        ax.set_xlabel(r"Partial-handle allocation $\lambda$")
        ax.set_ylabel(ylabels[dataset])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 0.9, 1.0])
        light_horizontal_grid(ax)

        utilization = (
            group.effective_record_utilization_mean.to_numpy(dtype=float) * 100
        )
        endpoint = methods[
            (methods.dataset == dataset)
            & (methods.method == "handle-only-drop-residual")
        ].iloc[0]
        utilization[-1] = float(endpoint.effective_record_utilization_mean) * 100
        util_ax = ax.twinx()
        util_ax.plot(
            x,
            utilization,
            color="#6C757D",
            linestyle="--",
            marker="s",
            markerfacecolor="white",
            markeredgewidth=0.55,
            linewidth=0.9,
            markersize=2.7,
        )
        util_ax.set_ylim(-2, 106)
        util_ax.set_yticks([0, 50, 100])
        util_ax.set_ylabel("Records used (%)")
        util_ax.annotate(
            rf"$\lambda=1$: {utilization[-1]:.1f}%",
            (1.0, utilization[-1]),
            xytext=(-10, -13),
            textcoords="offset points",
            ha="right",
            fontsize=5.8,
            color="#4D555C",
        )
        ax.annotate(
            r"$\lambda=.9$",
            (0.9, float(selected.metric_mean)),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=5.8,
            color="#1E8A70",
        )
        ax.annotate(
            r"$\equiv$ Stable--Local",
            (0.0, float(y[0])),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=5.5,
            color="#4D555C",
        )
        ax.annotate(
            r"$\equiv$ Handle--Only",
            (1.0, float(y[-1])),
            xytext=(-10, 5),
            textcoords="offset points",
            ha="right",
            fontsize=5.5,
            color="#4D555C",
        )
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_lambda_frontier_edbt")


def evidence_regime() -> None:
    evidence = pd.read_csv(
        result_path("mechanism_studies", "evidence_grid", "evidence_grid_summary.csv")
    )
    fig, axes = plt.subplots(1, 2, figsize=(5.55, 2.05), constrained_layout=True)
    panel_axes = []
    for ax, column, label in (
        (axes[0], "gain_over_fallback", "vs. Stable-Local (pp)"),
        (axes[1], "difference_from_handle_only", "vs. Handle-Only (pp)"),
    ):
        table = (
            evidence.pivot(
                index="q_record_given_handle", columns="q_user", values=column
            )
            * 100
        )
        vals = table.to_numpy(dtype=float)
        maximum = max(abs(vals.min()), abs(vals.max()), 1.0)
        norm = TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum)
        im = ax.imshow(
            vals, origin="lower", cmap=PVLDB_DIVERGING, norm=norm, aspect="auto"
        )
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                color = "white" if abs(norm(vals[i, j]) - 0.5) > 0.31 else COLORS["ink"]
                ax.text(
                    j,
                    i,
                    f"{vals[i, j]:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color=color,
                )
        ax.set_xticks(range(len(table.columns)), [f"{x:.2f}" for x in table.columns])
        ax.set_yticks(range(len(table.index)), [f"{x:.2f}" for x in table.index])
        ax.set_xlabel("Handle-user coverage")
        ax.set_ylabel("Within-handle coverage")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
        cbar.ax.set_ylabel(label, rotation=270, labelpad=11)
        panel_axes.append((ax, cbar.ax))
    save_panels(
        fig,
        [
            ("fig5_evidence_regime_a", panel_axes[0]),
            ("fig5_evidence_regime_b", panel_axes[1]),
        ],
    )
    save(fig, "fig5_evidence_regime")


def evidence_coverage_edbt() -> None:
    """Render the zero-centered evidence grid used in the paper."""
    frame = pd.read_csv(
        result_path("mechanism_studies", "evidence_grid", "evidence_grid_summary.csv")
    )
    users = sorted(frame.q_user.unique())
    handles = sorted(frame.q_record_given_handle.unique())
    if len(frame) != len(users) * len(handles):
        raise ValueError("the frozen evidence grid is incomplete")

    paper_style = dict(mpl.rcParamsDefault)
    paper_style.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    with mpl.rc_context(paper_style):
        fig, axes = plt.subplots(
            1, 2, figsize=(6.65, 2.05), constrained_layout=True
        )
        panels = [
            ("gain_over_fallback", r"$\Delta$ vs Stable--Local (pp)", -4.7, 10.7),
            (
                "difference_from_handle_only",
                r"$\Delta$ vs Handle--Only (pp)",
                -1.2,
                0.2,
            ),
        ]
        for ax, (column, title, vmin, vmax) in zip(axes, panels):
            pivot = frame.pivot(
                index="q_record_given_handle", columns="q_user", values=column
            )
            values = pivot.to_numpy(dtype=float) * 100
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            edges_x = [index - 0.5 for index in range(len(users) + 1)]
            edges_y = [index - 0.5 for index in range(len(handles) + 1)]
            image = ax.pcolormesh(
                edges_x,
                edges_y,
                values,
                cmap=mpl.colormaps["RdBu_r"],
                norm=norm,
                shading="flat",
            )
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_xticks(range(len(users)), [f"{value:.2g}" for value in users])
            ax.set_yticks(range(len(handles)), [f"{value:.2g}" for value in handles])
            ax.set_xlabel(r"$q_{\mathrm{user}}$")
            ax.set_ylabel(r"$q_{\mathrm{record\mid handle}}$")
            ax.set_title(title, fontsize=7.3, fontweight="bold", pad=4)
            ax.set_xticks([index - 0.5 for index in range(1, len(users))], minor=True)
            ax.set_yticks(
                [index - 0.5 for index in range(1, len(handles))], minor=True
            )
            ax.grid(which="minor", color="white", linewidth=1.25)
            ax.tick_params(which="minor", bottom=False, left=False)
            for row in range(values.shape[0]):
                for column_index in range(values.shape[1]):
                    value = values[row, column_index]
                    color = (
                        "white" if abs(norm(value) - 0.5) > 0.22 else "#25313B"
                    )
                    ax.text(
                        column_index,
                        row,
                        f"{value:+.1f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color=color,
                    )
            colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
            colorbar.solids.set_rasterized(False)
            colorbar.ax.tick_params(labelsize=5.6, length=2)
        save(fig, "fig_evidence_coverage_edbt")


def gradient_geometry() -> None:
    data = pd.read_csv(
        result_path(
            "mechanism_studies", "gradient_geometry", "gradient_geometry_summary.csv"
        )
    )
    fixed = data[data.method == "unitscope-tuned"].copy()
    fallback = data[data.method == "stable-local-fallback"].set_index(
        "gradient_correlation"
    )["noisy_mse_mean"]
    fixed["reduction"] = fixed.apply(
        lambda row: 100
        * (1 - row.noisy_mse_mean / fallback.loc[row.gradient_correlation]),
        axis=1,
    )
    # Reserve explicit room for the vertical ylabel and the external legend.
    # The previous fixed Bbox cropped the top of the ylabel in the exported PDF.
    fig, ax = plt.subplots(figsize=(3.33, 2.18))
    fig.subplots_adjust(left=0.22, right=0.99, bottom=0.22, top=0.78)
    for index, (correlation, group) in enumerate(fixed.groupby("gradient_correlation")):
        group = group.sort_values("lambda")
        plot_method(
            ax,
            group["lambda"],
            group.reduction,
            index,
            rf"cosine ${correlation:.1f}$",
            proposed=np.isclose(correlation, 0.0),
        )
    ax.set_xlabel(r"Partial-handle allocation $\lambda$")
    ax.set_ylabel("Noisy-MSE reduction vs. fallback (%)", labelpad=4)
    light_horizontal_grid(ax)
    legend = ax.legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0.0,
        handlelength=1.6,
        handletextpad=0.35,
        columnspacing=0.75,
        labelspacing=0.2,
    )
    fig.savefig(
        FIGURES / "fig5_gradient_geometry.pdf",
        bbox_inches="tight",
        bbox_extra_artists=(legend,),
        pad_inches=0.08,
    )
    fig.savefig(
        FIGURES / "fig5_gradient_geometry.png",
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=(legend,),
        pad_inches=0.08,
    )
    plt.close(fig)


def systems_scaling(
    labels: tuple[str, str, str] = ("a", "b", "c"),
    filename: str = "fig6_system_scaling",
) -> None:
    compiler = pd.read_csv(
        result_path("mechanism_studies", "compiler_benchmark", "compiler_scaling.csv")
    )
    mpc = pd.read_csv(
        result_path("mechanism_studies", "mpc_backend", "mpc_backend_summary.csv")
    )
    lifecycle = pd.read_csv(
        result_path(
            "mechanism_studies", "dynamic_lifecycle", "dynamic_lifecycle_summary.csv"
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3), constrained_layout=True)
    ax = axes[0]
    plot_method(
        ax, compiler.records, compiler.compile_seconds, 0, "Compile", proposed=True
    )
    plot_method(ax, compiler.records, compiler.evidence_seconds, 1, "Evidence audit")
    plot_method(
        ax,
        compiler.records,
        compiler.adjacent_recompile_and_audit_seconds,
        3,
        "Neighbor audit",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Records")
    ax.set_ylabel("Wall time (s)")
    light_horizontal_grid(ax)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    for index, dim in enumerate((128, 1024)):
        g = mpc[mpc.dimension == dim]
        plot_method(
            ax,
            g.n_units,
            g.wan_projected_seconds,
            index,
            f"d={dim}",
            proposed=index == 0,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Plan units")
    ax.set_ylabel("Projected WAN time (s)")
    light_horizontal_grid(ax)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[2]
    show = lifecycle[lifecycle.wur_max.notna()].copy()
    label_map = {
        "preallocated-new-record": "preallocated\nrecord",
        "single-user-deletion": "person\ndeletion",
        "credential-revocation": "credential\nrevocation",
        "institution-exit-same-epoch": "silo exit\nsame epoch",
    }
    show["label"] = show.scenario.map(label_map)
    show = show.sort_values("wur_max")
    colors = [COLORS["green"] if v <= 1 else COLORS["red"] for v in show.wur_max]
    ax.bar(range(len(show)), show.wur_max, color=colors, width=0.68)
    ax.axhline(1, color=COLORS["green"], ls="--", lw=1.0)
    ax.set_xticks(range(len(show)), show.label, rotation=35, ha="right")
    ax.set_ylabel(r"Maximum WUR ($C$ units)")
    ax.set_yscale("symlog", linthresh=1)
    light_horizontal_grid(ax)
    save(fig, filename)


def combined_evidence_and_systems() -> None:
    """Build the six-panel experiment atlas at its final two-column size."""
    frontier = pd.read_csv(
        result_path(
            "mechanism_studies", "lambda_frontier", "lambda_frontier_paired_effects.csv"
        )
    )
    evidence = pd.read_csv(
        result_path("mechanism_studies", "evidence_grid", "evidence_grid_summary.csv")
    )
    geometry_all = pd.read_csv(
        result_path(
            "mechanism_studies", "gradient_geometry", "gradient_geometry_summary.csv"
        )
    )
    geometry = geometry_all[
        geometry_all.method.isin(["unitscope-tuned", "unitscope-equal"])
    ].copy()
    fallback = geometry_all.query("method == 'stable-local-fallback'").set_index(
        "gradient_correlation"
    )["noisy_mse_mean"]
    geometry["reduction"] = geometry.apply(
        lambda r: 100 * (1 - r.noisy_mse_mean / fallback.loc[r.gradient_correlation]),
        axis=1,
    )
    compiler = pd.read_csv(
        result_path("mechanism_studies", "compiler_benchmark", "compiler_scaling.csv")
    )
    mpc = pd.read_csv(
        result_path("mechanism_studies", "mpc_backend", "mpc_backend_summary.csv")
    )
    lifecycle = pd.read_csv(
        result_path(
            "mechanism_studies", "dynamic_lifecycle", "dynamic_lifecycle_summary.csv"
        )
    )

    # Keep the six panels compact at two-column size.
    fig = plt.figure(figsize=(6.9, 3.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 6, height_ratios=(1.0, 1.0))
    axes = [
        fig.add_subplot(grid[0, :2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:]),
        fig.add_subplot(grid[1, :2]),
        fig.add_subplot(grid[1, 2:4]),
        fig.add_subplot(grid[1, 4:]),
    ]

    ax = axes[0]
    for index, (dataset, label) in enumerate(
        (("femnist", "FEMNIST"), ("synthea", "Synthea"))
    ):
        group = frontier[frontier.dataset == dataset].sort_values("lambda")
        x = group["lambda"].to_numpy(dtype=float)
        mean = 100 * group.gain_over_fallback_mean.to_numpy(dtype=float)
        lower = 100 * group.gain_over_fallback_ci_lower.to_numpy(dtype=float)
        upper = 100 * group.gain_over_fallback_ci_upper.to_numpy(dtype=float)
        ax.fill_between(
            x, lower, upper, color=PLOT_COLORS[index], alpha=0.10, linewidth=0
        )
        plot_method(ax, x, mean, index, label, proposed=index == 0)
    ax.axhline(0, color=COLORS["gray"], linewidth=0.8)
    ax.axvline(0.9, color=COLORS["gray"], linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"Allocation $\lambda$")
    ax.set_ylabel("Gain over Stable-Local (pp)")
    light_horizontal_grid(ax)
    ax.legend(loc="upper left")

    ax = axes[1]
    table = (
        evidence.pivot(
            index="q_record_given_handle", columns="q_user", values="gain_over_fallback"
        )
        * 100
    )
    vals = table.to_numpy()
    norm = TwoSlopeNorm(vmin=min(-5, vals.min()), vcenter=0, vmax=max(11, vals.max()))
    im = ax.imshow(vals, origin="lower", cmap=PVLDB_DIVERGING, norm=norm, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            color = "white" if abs(norm(vals[i, j]) - 0.5) > 0.31 else COLORS["ink"]
            ax.text(
                j,
                i,
                f"{vals[i, j]:+.1f}",
                ha="center",
                va="center",
                fontsize=6.6,
                color=color,
            )
    ax.set_xticks(range(len(table.columns)), [f"{x:.2f}" for x in table.columns])
    ax.set_yticks(range(len(table.index)), [f"{x:.2f}" for x in table.index])
    ax.set_xlabel("Handle-user coverage")
    ax.set_ylabel("Within-handle coverage")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.ax.tick_params(labelsize=6.2)
    evidence_cbar = cbar.ax

    ax = axes[2]
    table = geometry.pivot_table(
        index="gradient_correlation",
        columns="lambda",
        values="reduction",
        aggfunc="first",
    )
    vals = table.to_numpy()
    norm = TwoSlopeNorm(vmin=min(-2, vals.min()), vcenter=0, vmax=max(22, vals.max()))
    im = ax.imshow(vals, origin="lower", cmap=PVLDB_DIVERGING, norm=norm, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            value = vals[i, j]
            color = "white" if abs(norm(value) - 0.5) > 0.32 else COLORS["ink"]
            ax.text(
                j,
                i,
                f"{value:+.0f}",
                ha="center",
                va="center",
                fontsize=6.1,
                color=color,
            )
    ax.set_xticks(range(len(table.columns)), [f"{x:.2g}" for x in table.columns])
    ax.set_yticks(range(len(table.index)), [f"{x:.1f}" for x in table.index])
    ax.set_xlabel(r"Allocation $\lambda$")
    ax.set_ylabel("Gradient cosine")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.ax.tick_params(labelsize=6.2)
    geometry_cbar = cbar.ax

    ax = axes[3]
    plot_method(
        ax, compiler.records, compiler.compile_seconds, 0, "Compile", proposed=True
    )
    plot_method(ax, compiler.records, compiler.evidence_seconds, 1, "Evidence audit")
    plot_method(
        ax,
        compiler.records,
        compiler.adjacent_recompile_and_audit_seconds,
        3,
        "Neighbor audit",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Records")
    ax.set_ylabel("Wall time (s)")
    light_horizontal_grid(ax)
    ax.legend(loc="upper left", ncol=1)

    ax = axes[4]
    for index, dim in enumerate((128, 1024)):
        group = mpc[mpc.dimension == dim]
        plot_method(
            ax,
            group.n_units,
            group.wan_projected_seconds,
            index,
            f"d={dim}",
            proposed=index == 0,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Plan units")
    ax.set_ylabel("WAN time (s)")
    light_horizontal_grid(ax)
    ax.legend(loc="upper left")

    ax = axes[5]
    show = lifecycle[lifecycle.wur_max.notna()].copy()
    label_map = {
        "preallocated-new-record": "new\nrecord",
        "single-user-deletion": "person\ndeletion",
        "credential-revocation": "credential\nrevocation",
        "institution-exit-same-epoch": "silo exit\nsame epoch",
    }
    show["label"] = show.scenario.map(label_map)
    show = show.sort_values("wur_max")
    bar_colors = [
        COLORS["green"] if value <= 1 else COLORS["red"] for value in show.wur_max
    ]
    ax.bar(range(len(show)), show.wur_max, color=bar_colors, width=0.66)
    ax.axhline(1, color=COLORS["green"], ls="--", lw=1.0)
    ax.set_xticks(range(len(show)), show.label, rotation=24, ha="right")
    ax.set_ylabel(r"Maximum WUR ($C$ units)")
    ax.set_yscale("symlog", linthresh=1)
    light_horizontal_grid(ax)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for axis, position in zip(
        axes[3:],
        (
            (0.06997339409722222, 0.17542375993405024, 0.2284219293956022, 0.32494775198252823),
            (0.3768251907100465, 0.17542375993405024, 0.22842192939560246, 0.32494775198252823),
            (0.7254444567546218, 0.17542375993405024, 0.22842192939560213, 0.32494775198252823),
        ),
    ):
        axis.set_position(position)
    fixed_bboxes = {
        "fig5_experiment_atlas_d": Bbox.from_bounds(
            0.016670000000000018,
            0.09668013396881739,
            2.067257732100489,
            1.431772812592053,
        ),
        "fig5_experiment_atlas_e": Bbox.from_bounds(
            2.164898080222238,
            0.09668013396881739,
            2.0471593263681753,
            1.429434401780918,
        ),
        "fig5_experiment_atlas_f": Bbox.from_bounds(
            4.520839765929806,
            0.016670000000000018,
            2.085838298506739,
            1.5094445357497355,
        ),
    }
    save_panels(
        fig,
        [
            ("fig5_experiment_atlas_a", (axes[0],)),
            ("fig5_experiment_atlas_b", (axes[1], evidence_cbar)),
            ("fig5_experiment_atlas_c", (axes[2], geometry_cbar)),
            ("fig5_experiment_atlas_d", (axes[3],)),
            ("fig5_experiment_atlas_e", (axes[4],)),
            ("fig5_experiment_atlas_f", (axes[5],)),
        ],
        common_canvas=False,
        fixed_bboxes=fixed_bboxes,
    )
    save(fig, "fig5_experiment_atlas")


TRAJECTORY_DISPLAY = {
    "stable-local-fallback": "Stable-Local",
    "unitscope-lambda-sweep": "UnitScope-Fixed",
    "handle-only-drop-residual": "Handle-Only",
    "naive-recalibrated-2c": "Naive-2C",
    "full-identity-reference": "Complete-ID Oracle",
}
TRAJECTORY_COLORS = {
    "stable-local-fallback": "#6B7280",
    "unitscope-lambda-sweep": "#176B87",
    "handle-only-drop-residual": "#D97706",
    "naive-recalibrated-2c": "#8B5CF6",
    "full-identity-reference": "#2A9D8F",
}
TRAJECTORY_MARKERS = {
    "stable-local-fallback": "s",
    "unitscope-lambda-sweep": "o",
    "handle-only-drop-residual": "v",
    "naive-recalibrated-2c": "P",
    "full-identity-reference": "D",
}


def four_dataset_convergence() -> None:
    """Render the four frozen learning trajectories used in the paper."""
    primary = pd.read_csv(result_path("reference_methods", "main_trajectory_summary.csv"))
    primary_methods = [
        "stable-local-fallback",
        "unitscope-lambda-sweep",
        "handle-only-drop-residual",
        "naive-recalibrated-2c",
    ]
    for dataset, suffix, ylabel in (
        ("femnist", "a", "Writer macro accuracy"),
        ("synthea", "b", "Patient AUPRC"),
    ):
        fig, axis = plt.subplots(figsize=(4.15, 2.75))
        subset = primary[primary.dataset == dataset]
        for method in primary_methods:
            method_frame = subset[subset.method == method].sort_values("round")
            axis.plot(
                method_frame["round"],
                method_frame.metric_mean,
                color=TRAJECTORY_COLORS[method],
                marker=TRAJECTORY_MARKERS[method],
                markersize=3.8,
                linewidth=1.35,
                label=TRAJECTORY_DISPLAY[method],
            )
            axis.fill_between(
                method_frame["round"],
                method_frame.ci_lower,
                method_frame.ci_upper,
                color=TRAJECTORY_COLORS[method],
                alpha=0.11,
                linewidth=0,
            )
        axis.set_xlabel("Training round")
        axis.set_ylabel(ylabel)
        light_horizontal_grid(axis)
        axis.legend(frameon=False, fontsize=7, ncol=2, loc="best")
        fig.tight_layout(pad=0.45)
        save(fig, f"fig_supp_four_dataset_convergence_{suffix}")

    external = pd.read_csv(result_path("learning_extensions", "trajectory_long.csv"))
    external = external[external.family == "sent140-account-task-v1"]

    sent140 = external[external.dataset == "sent140"]
    fig, axis = plt.subplots(figsize=(4.15, 2.75))
    for method in TRAJECTORY_DISPLAY:
        method_frame = sent140[sent140.method == method]
        summary = (
            method_frame.groupby("round")
            .metric_value.agg(["mean", "std", "count"])
            .reset_index()
        )
        interval = 1.96 * summary["std"].fillna(0.0) / np.sqrt(summary["count"])
        axis.plot(
            summary["round"],
            summary["mean"],
            color=TRAJECTORY_COLORS[method],
            marker=TRAJECTORY_MARKERS[method],
            markersize=3.8,
            linewidth=1.35,
            label=TRAJECTORY_DISPLAY[method],
        )
        axis.fill_between(
            summary["round"],
            summary["mean"] - interval,
            summary["mean"] + interval,
            color=TRAJECTORY_COLORS[method],
            alpha=0.1,
            linewidth=0,
        )
    axis.set_xlabel("Training round")
    axis.set_ylabel("Account macro accuracy")
    light_horizontal_grid(axis)
    axis.legend(frameon=False, fontsize=7, ncol=2, loc="best")
    fig.tight_layout(pad=0.45)
    save(fig, "fig_supp_four_dataset_convergence_c")

    movielens = pd.read_csv(result_path("movielens_task", "trajectory_long.csv"))
    movielens = movielens[movielens.dataset == "movielens1m"]
    fig, axis = plt.subplots(figsize=(4.15, 2.75))
    for method in TRAJECTORY_DISPLAY:
        method_frame = movielens[movielens.method == method]
        summary = (
            method_frame.groupby("round")
            .metric_value.agg(
                mean="mean",
                lower=lambda values: np.percentile(values, 2.5),
                upper=lambda values: np.percentile(values, 97.5),
            )
            .reset_index()
        )
        axis.plot(
            summary["round"],
            summary["mean"],
            linewidth=1.35,
            color=TRAJECTORY_COLORS[method],
            marker=TRAJECTORY_MARKERS[method],
            markersize=3.8,
            label=TRAJECTORY_DISPLAY[method],
        )
        axis.fill_between(
            summary["round"],
            summary.lower,
            summary.upper,
            color=TRAJECTORY_COLORS[method],
            alpha=0.10,
            linewidth=0,
        )
    axis.set_xlabel("Training round")
    axis.set_ylabel("Account macro accuracy")
    light_horizontal_grid(axis)
    axis.legend(loc="best", fontsize=7, frameon=False, ncol=2)
    fig.tight_layout(pad=0.45)
    save(fig, "fig_supp_four_dataset_convergence_d")


def noniid_convergence() -> None:
    """Render the two archived non-IID FEMNIST trajectories."""
    trajectory = pd.read_csv(result_path("learning_extensions", "trajectory_long.csv"))
    trajectory = trajectory[trajectory.family == "user-noniid-dirichlet-v1"]
    expected_alphas = {0.1: "a", 0.5: "b"}
    if set(trajectory.dirichlet_alpha.dropna().unique()) != set(expected_alphas):
        raise ValueError("the frozen non-IID trajectory grid is incomplete")

    default_style = dict(mpl.rcParamsDefault)
    default_style["backend"] = "Agg"
    default_style["figure.hooks"] = []
    with mpl.rc_context(default_style):
        for alpha, suffix in expected_alphas.items():
            family_frame = trajectory[np.isclose(trajectory.dirichlet_alpha, alpha)]
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
                    color=TRAJECTORY_COLORS[method],
                    marker=TRAJECTORY_MARKERS[method],
                    markersize=3.0,
                    linewidth=1.15,
                    label=TRAJECTORY_DISPLAY[method],
                )
                axis.fill_between(
                    summary["round"],
                    summary["mean"] - sem95,
                    summary["mean"] + sem95,
                    color=TRAJECTORY_COLORS[method],
                    alpha=0.1,
                    linewidth=0,
                )
            axis.set_xlabel("Training round")
            axis.set_ylabel("Writer macro accuracy")
            axis.grid(axis="y", color="#D1D5DB", linewidth=0.5, alpha=0.7)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color("#111827")
            axis.legend(frameon=False, fontsize=6.1, ncol=2, loc="best")
            fig.tight_layout(pad=0.45)
            fig.savefig(FIGURES / f"fig_supp_noniid_{suffix}.png", dpi=300)
            fig.savefig(FIGURES / f"fig_supp_noniid_{suffix}.pdf")
            plt.close(fig)


CONTRACT_METHOD_LABELS = {
    "stable-local-fallback": "Stable local fallback",
    "unitscope-tuned": "UnitScope",
    "handle-only-drop-residual": "Handle only",
}
CONTRACT_COLORS = {
    "stable-local-fallback": "#7A7A7A",
    "unitscope-tuned": "#3B6FB6",
    "handle-only-drop-residual": "#C4A646",
}
CONTRACT_MARKERS = {
    "stable-local-fallback": "s",
    "unitscope-tuned": "o",
    "handle-only-drop-residual": "^",
}
CONTRACT_LINESTYLES = {
    "stable-local-fallback": "--",
    "unitscope-tuned": "-",
    "handle-only-drop-residual": "-.",
}


def _contract_line(
    axis: plt.Axes,
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
    chosen_color = color or CONTRACT_COLORS[method]
    proposed = method == "unitscope-tuned"
    axis.plot(
        x,
        y,
        marker=CONTRACT_MARKERS.get(method, "o"),
        markersize=3.8 if proposed else 3.2,
        linewidth=1.7 if proposed else 1.2,
        linestyle=linestyle if linestyle != "-" else CONTRACT_LINESTYLES.get(method, "-"),
        color=chosen_color,
        markerfacecolor="white",
        markeredgecolor=chosen_color,
        markeredgewidth=0.6,
        label=label or CONTRACT_METHOD_LABELS[method],
        zorder=5 if proposed else 3,
    )
    axis.fill_between(x, lower, upper, color=chosen_color, alpha=0.12, linewidth=0)


def _render_contract_parameter_sweep() -> None:
    """Render the three contract stress-test panels used in the paper."""
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "Liberation Sans",
                "DejaVu Sans",
            ],
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
            "savefig.pad_inches": 0.1,
        }
    )
    frame = pd.read_csv(result_path("contract_parameter_sweep", "summary.csv"))
    methods = (
        "stable-local-fallback",
        "unitscope-tuned",
        "handle-only-drop-residual",
    )

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.75), constrained_layout=True)
    axis = axes[0]
    subset = frame[frame.stage == "handle_multiplicity"]
    for method in methods:
        _contract_line(
            axis,
            subset[subset.method == method],
            "H",
            method,
            "deterministic_distortion",
        )
    axis.set_xlabel(r"Handle bound $H$")
    axis.set_xticks([1, 2, 4])
    light_horizontal_grid(axis)

    axis = axes[1]
    subset = frame[
        (frame.stage == "fallback_bound") & (frame.method == "unitscope-tuned")
    ]
    for mixed, label, color, linestyle in (
        (False, "Pure local units", COLORS["blue"], "-"),
        (True, "Mixed local units", COLORS["orange"], "--"),
    ):
        _contract_line(
            axis,
            subset[subset.mixed_local_slots == mixed],
            "B",
            "unitscope-tuned",
            "deterministic_distortion",
            label=label,
            color=color,
            linestyle=linestyle,
        )
    axis.set_xlabel(r"Fallback bound $B$")
    axis.set_xticks([2, 4, 6, 8])
    light_horizontal_grid(axis)
    axis.legend(loc="best")

    formatter = FuncFormatter(lambda value, _: f"{value * 1e4:g}")
    for axis in axes[:2]:
        axis.yaxis.set_major_formatter(formatter)
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.yaxis.get_offset_text().set_visible(False)
        axis.set_ylabel(r"Deterministic MSE ($10^{-4}$)")

    axis = axes[2]
    subset = frame[frame.stage == "privacy_level"]
    for method in methods:
        _contract_line(
            axis,
            subset[subset.method == method],
            "target_epsilon",
            method,
            "noisy_aggregate_mse",
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    axis.set_xlabel(r"One-release $\epsilon$ ($\delta=10^{-6}$)")
    axis.set_ylabel("Noisy aggregate MSE")
    light_horizontal_grid(axis)
    axis.legend(loc="upper right", fontsize=6.1)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight_bboxes = [
        axis.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        for axis in axes
    ]
    axes_bboxes = [
        axis.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
        for axis in axes
    ]
    pad = 0.025
    left = max(axis.x0 - tight.x0 for axis, tight in zip(axes_bboxes, tight_bboxes)) + pad
    right = max(tight.x1 - axis.x1 for axis, tight in zip(axes_bboxes, tight_bboxes)) + pad
    bottom = max(axis.y0 - tight.y0 for axis, tight in zip(axes_bboxes, tight_bboxes)) + pad
    top = max(tight.y1 - axis.y1 for axis, tight in zip(axes_bboxes, tight_bboxes)) + pad
    panel_bboxes = [
        Bbox.from_extents(
            axis.x0 - left,
            axis.y0 - bottom,
            axis.x1 + right,
            axis.y1 + top,
        )
        for axis in axes_bboxes
    ]
    layout_engine = fig.get_layout_engine()
    fig.set_layout_engine("none")
    try:
        for suffix, axis, bbox_inches in zip(("a", "b", "c"), axes, panel_bboxes):
            visibility = [(other, other.get_visible()) for other in fig.axes]
            try:
                for other, _ in visibility:
                    other.set_visible(other is axis)
                fig.savefig(
                    FIGURES / f"fig5_contract_parameter_sweep_{suffix}.pdf",
                    bbox_inches=bbox_inches,
                )
                fig.savefig(
                    FIGURES / f"fig5_contract_parameter_sweep_{suffix}.png",
                    dpi=300,
                    bbox_inches=bbox_inches,
                )
            finally:
                for other, was_visible in visibility:
                    other.set_visible(was_visible)
    finally:
        fig.set_layout_engine(layout_engine)
    plt.close(fig)


def contract_parameter_sweep() -> None:
    """Render contract panels without inheriting style from earlier figures."""
    default_style = dict(mpl.rcParamsDefault)
    default_style["backend"] = "Agg"
    default_style["figure.hooks"] = []
    with mpl.rc_context(default_style):
        _render_contract_parameter_sweep()


def main() -> None:
    args = parse_args()
    configure_paths(args.results_root, args.figures_root, args.fallback_results_root)
    paper_style()
    partial_handle_scenario()
    overview()
    boundary_illusion()
    utility_comparison()
    lambda_frontier()
    lambda_frontier_edbt()
    evidence_regime()
    evidence_coverage_edbt()
    gradient_geometry()
    four_dataset_convergence()
    noniid_convergence()
    contract_parameter_sweep()
    combined_evidence_and_systems()
    systems_scaling()
    systems_scaling(("c", "d", "e"), "fig6_system_scaling_combined")
    print(f"Wrote figures to {FIGURES}")
    print("Read result files from:")
    for directory in sorted(USED_RESULT_ROOTS):
        print(f"  {directory}")


if __name__ == "__main__":
    main()
