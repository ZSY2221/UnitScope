from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class CoreEffects:
    delta_fallback: float
    interior_gain: float
    safe_competitor_gain: float
    recovery: float | None
    recovery_status: str


@dataclass(frozen=True)
class ClusterEffectInterval:
    estimate: float
    lower: float
    upper: float
    seeds: int
    minimum_clusters_per_seed: int


def core_effects(
    unit_scope: float,
    fallback: float,
    handle_only: float,
    naive_recalibrated_2c: float,
    full_identity: float,
    *,
    full_identity_gap_interval: tuple[float, float] | None = None,
    minimum_gap: float = 0.005,
) -> CoreEffects:
    delta = unit_scope - fallback
    interior = unit_scope - max(fallback, handle_only)
    safe = unit_scope - max(fallback, handle_only, naive_recalibrated_2c)
    denominator = full_identity - fallback
    if denominator <= 0:
        recovery, status = None, "N/A: full-identity reference does not exceed fallback"
    elif denominator < minimum_gap:
        recovery, status = (
            None,
            "N/A: denominator below preregistered practical threshold",
        )
    elif (
        full_identity_gap_interval is not None
        and full_identity_gap_interval[0] <= 0 <= full_identity_gap_interval[1]
    ):
        recovery, status = None, "N/A: denominator confidence interval contains zero"
    else:
        recovery, status = delta / denominator, "reported"
    return CoreEffects(delta, interior, safe, recovery, status)


def paired_sign_flip_test(differences: Sequence[float]) -> float:
    """Exact two-sided paired randomization test for a small preregistered family."""

    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("at least one finite paired difference is required")
    observed = abs(float(values.mean()))
    if len(values) <= 20:
        exceed = 0
        total = 2 ** len(values)
        for signs in product((-1.0, 1.0), repeat=len(values)):
            statistic = abs(float(np.mean(values * np.asarray(signs))))
            exceed += statistic >= observed - 1e-15
        return exceed / total
    rng = np.random.default_rng(20260714)
    signs = rng.choice((-1.0, 1.0), size=(200_000, len(values)))
    statistics = np.abs((signs * values[None, :]).mean(axis=1))
    return float((1 + np.sum(statistics >= observed - 1e-15)) / (1 + len(statistics)))


def paired_seed_bootstrap(
    differences: Sequence[float], *, n_resamples: int = 20_000, seed: int = 20260714
) -> Interval:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("at least two finite paired seeds are required")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    samples = values[indices].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return Interval(float(values.mean()), float(lower), float(upper))


def paired_ratio_bootstrap(
    numerator: Sequence[float],
    denominator: Sequence[float],
    *,
    n_resamples: int = 20_000,
    seed: int = 20260714,
) -> Interval:
    numerator_values = np.asarray(numerator, dtype=np.float64)
    denominator_values = np.asarray(denominator, dtype=np.float64)
    finite = np.isfinite(numerator_values) & np.isfinite(denominator_values)
    numerator_values = numerator_values[finite]
    denominator_values = denominator_values[finite]
    if len(numerator_values) < 2:
        raise ValueError("paired ratio bootstrap needs at least two finite seed pairs")
    denominator_mean = float(denominator_values.mean())
    if denominator_mean <= 0:
        raise ValueError("ratio denominator mean must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(numerator_values), size=(n_resamples, len(numerator_values))
    )
    numerator_boot = numerator_values[indices].mean(axis=1)
    denominator_boot = denominator_values[indices].mean(axis=1)
    valid = denominator_boot > 0
    if valid.mean() < 0.975:
        raise ValueError("bootstrap denominator is not stably positive")
    ratios = numerator_boot[valid] / denominator_boot[valid]
    lower, upper = np.quantile(ratios, [0.025, 0.975])
    return Interval(
        float(numerator_values.mean() / denominator_mean), float(lower), float(upper)
    )


def clustered_metric_bootstrap(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    cluster_ids: Sequence[str],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_resamples: int = 20_000,
    seed: int = 20260714,
) -> Interval:
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    clusters = np.asarray(cluster_ids)
    if not (len(truth) == len(prediction) == len(clusters)):
        raise ValueError("truth, prediction, and cluster IDs must have equal length")
    unique = np.unique(clusters)
    if len(unique) < 2:
        raise ValueError("clustered bootstrap needs at least two writers/users")
    members = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sampled_clusters = rng.choice(unique, size=len(unique), replace=True)
        sampled_indices = np.concatenate(
            [members[cluster] for cluster in sampled_clusters]
        )
        estimates[index] = metric(truth[sampled_indices], prediction[sampled_indices])
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return Interval(float(metric(truth, prediction)), float(lower), float(upper))


def holm_adjust(p_values: Mapping[str, float]) -> Mapping[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def hierarchical_cluster_effect_bootstrap(
    runs: Mapping[str, Mapping[int, Sequence[Mapping[str, object]]]],
    unit_method: str,
    competitor_methods: Sequence[str],
    *,
    metric: str,
    n_resamples: int = 20_000,
    seed: int = 20260714,
) -> ClusterEffectInterval:
    """Pair by seed and user, then bootstrap both levels."""

    required_methods = (unit_method, *competitor_methods)
    if any(method not in runs for method in required_methods):
        raise ValueError("cluster runs are missing a required method")
    common_seeds = sorted(
        set.intersection(*(set(runs[method]) for method in required_methods))
    )
    if len(common_seeds) < 2:
        raise ValueError("hierarchical bootstrap needs at least two paired seeds")
    prepared = {}
    minimum_clusters = None
    for run_seed in common_seeds:
        by_method = {}
        common_clusters = None
        for method in required_methods:
            mapping = {str(row["cluster_id"]): row for row in runs[method][run_seed]}
            by_method[method] = mapping
            common_clusters = (
                set(mapping)
                if common_clusters is None
                else common_clusters & set(mapping)
            )
        cluster_ids = sorted(common_clusters or ())
        if len(cluster_ids) < 2:
            raise ValueError(f"seed {run_seed} has fewer than two paired clusters")
        minimum_clusters = (
            len(cluster_ids)
            if minimum_clusters is None
            else min(minimum_clusters, len(cluster_ids))
        )
        prepared[run_seed] = (cluster_ids, by_method)

    def evaluate(method: str, cluster_ids: Sequence[str], by_method) -> float:
        rows = [by_method[method][cluster_id] for cluster_id in cluster_ids]
        if metric == "accuracy":
            return float(np.mean([float(row["accuracy"]) for row in rows]))
        if metric == "auprc":
            from sklearn.metrics import average_precision_score

            labels = [int(row["binary_label"]) for row in rows]
            probabilities = [float(row["positive_probability"]) for row in rows]
            if len(set(labels)) < 2:
                return float("nan")
            return float(average_precision_score(labels, probabilities))
        raise ValueError("metric must be accuracy or auprc")

    def effect(run_seed: int, sampled_ids: Sequence[str]) -> float:
        _, by_method = prepared[run_seed]
        unit_value = evaluate(unit_method, sampled_ids, by_method)
        competitor_value = max(
            evaluate(method, sampled_ids, by_method) for method in competitor_methods
        )
        return unit_value - competitor_value

    point_effects = []
    for run_seed in common_seeds:
        cluster_ids, _ = prepared[run_seed]
        point_effects.append(effect(run_seed, cluster_ids))
    estimate = float(np.nanmean(point_effects))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_resamples, dtype=np.float64)
    seed_values = np.asarray(common_seeds)
    for index in range(n_resamples):
        sampled_seeds = rng.choice(seed_values, size=len(seed_values), replace=True)
        seed_effects = []
        for run_seed in sampled_seeds:
            cluster_ids, _ = prepared[int(run_seed)]
            sampled_ids = rng.choice(
                cluster_ids, size=len(cluster_ids), replace=True
            ).tolist()
            seed_effects.append(effect(int(run_seed), sampled_ids))
        bootstrap[index] = np.nanmean(seed_effects)
    finite = bootstrap[np.isfinite(bootstrap)]
    if len(finite) < 0.95 * n_resamples:
        raise ValueError(
            "too many hierarchical bootstrap samples have undefined metrics"
        )
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return ClusterEffectInterval(
        estimate,
        float(lower),
        float(upper),
        len(common_seeds),
        int(minimum_clusters or 0),
    )


def hierarchical_cluster_recovery_bootstrap(
    runs: Mapping[str, Mapping[int, Sequence[Mapping[str, object]]]],
    unit_method: str,
    fallback_method: str,
    full_identity_method: str,
    *,
    metric: str,
    n_resamples: int = 20_000,
    seed: int = 20260714,
) -> ClusterEffectInterval:
    """Recompute the ratio of paired mean gaps inside every two-level draw."""

    required_methods = (unit_method, fallback_method, full_identity_method)
    if any(method not in runs for method in required_methods):
        raise ValueError("cluster runs are missing a recovery method")
    common_seeds = sorted(
        set.intersection(*(set(runs[method]) for method in required_methods))
    )
    if len(common_seeds) < 2:
        raise ValueError("hierarchical recovery needs at least two paired seeds")
    prepared = {}
    minimum_clusters = None
    for run_seed in common_seeds:
        by_method = {}
        common_clusters = None
        for method in required_methods:
            mapping = {str(row["cluster_id"]): row for row in runs[method][run_seed]}
            by_method[method] = mapping
            common_clusters = (
                set(mapping)
                if common_clusters is None
                else common_clusters & set(mapping)
            )
        cluster_ids = sorted(common_clusters or ())
        if len(cluster_ids) < 2:
            raise ValueError(f"seed {run_seed} has fewer than two paired clusters")
        minimum_clusters = (
            len(cluster_ids)
            if minimum_clusters is None
            else min(minimum_clusters, len(cluster_ids))
        )
        prepared[run_seed] = (cluster_ids, by_method)

    def evaluate(method: str, cluster_ids: Sequence[str], by_method) -> float:
        rows = [by_method[method][cluster_id] for cluster_id in cluster_ids]
        if metric == "accuracy":
            return float(np.mean([float(row["accuracy"]) for row in rows]))
        if metric == "auprc":
            from sklearn.metrics import average_precision_score

            labels = [int(row["binary_label"]) for row in rows]
            probabilities = [float(row["positive_probability"]) for row in rows]
            if len(set(labels)) < 2:
                return float("nan")
            return float(average_precision_score(labels, probabilities))
        raise ValueError("metric must be accuracy or auprc")

    def gaps(run_seed: int, sampled_ids: Sequence[str]) -> tuple[float, float]:
        _, by_method = prepared[run_seed]
        fallback = evaluate(fallback_method, sampled_ids, by_method)
        return (
            evaluate(unit_method, sampled_ids, by_method) - fallback,
            evaluate(full_identity_method, sampled_ids, by_method) - fallback,
        )

    point_gaps = [gaps(run_seed, prepared[run_seed][0]) for run_seed in common_seeds]
    point_numerator = float(np.nanmean([value[0] for value in point_gaps]))
    point_denominator = float(np.nanmean([value[1] for value in point_gaps]))
    if point_denominator <= 0:
        raise ValueError("hierarchical recovery denominator is not positive")
    estimate = point_numerator / point_denominator

    rng = np.random.default_rng(seed)
    bootstrap = np.full(n_resamples, np.nan, dtype=np.float64)
    seed_values = np.asarray(common_seeds)
    for index in range(n_resamples):
        sampled_seeds = rng.choice(seed_values, size=len(seed_values), replace=True)
        sampled_gaps = []
        for run_seed in sampled_seeds:
            cluster_ids, _ = prepared[int(run_seed)]
            sampled_ids = rng.choice(
                cluster_ids, size=len(cluster_ids), replace=True
            ).tolist()
            sampled_gaps.append(gaps(int(run_seed), sampled_ids))
        numerator = float(np.nanmean([value[0] for value in sampled_gaps]))
        denominator = float(np.nanmean([value[1] for value in sampled_gaps]))
        if denominator > 0:
            bootstrap[index] = numerator / denominator
    finite = bootstrap[np.isfinite(bootstrap)]
    if len(finite) < 0.975 * n_resamples:
        raise ValueError("hierarchical recovery denominator is not stably positive")
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return ClusterEffectInterval(
        float(estimate),
        float(lower),
        float(upper),
        len(common_seeds),
        int(minimum_clusters or 0),
    )
