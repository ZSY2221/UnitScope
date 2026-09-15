from __future__ import annotations

import pytest

from experiments.unitscope.stats import (
    core_effects,
    hierarchical_cluster_effect_bootstrap,
    hierarchical_cluster_recovery_bootstrap,
    paired_ratio_bootstrap,
    paired_sign_flip_test,
)


def test_five_positive_pairs_cannot_reach_two_sided_five_percent() -> None:
    assert paired_sign_flip_test([1, 1, 1, 1, 1]) == 0.0625


def test_ten_positive_pairs_support_preregistered_exact_test() -> None:
    assert paired_sign_flip_test([1] * 10) == 2 / (2**10)


def test_recovery_is_not_clipped_and_can_be_na() -> None:
    reported = core_effects(
        0.9, 0.5, 0.6, 0.55, 0.8, full_identity_gap_interval=(0.2, 0.4)
    )
    assert reported.recovery is not None and reported.recovery > 1
    unavailable = core_effects(
        0.6, 0.5, 0.55, 0.52, 0.503, full_identity_gap_interval=(-0.01, 0.02)
    )
    assert unavailable.recovery is None


def test_paired_ratio_bootstrap_uses_ratio_of_paired_means() -> None:
    interval = paired_ratio_bootstrap(
        [0.1, 0.2, 0.3], [0.2, 0.4, 0.6], n_resamples=1000, seed=1
    )
    assert interval.estimate == 0.5
    assert interval.lower <= 0.5 <= interval.upper


def test_hierarchical_cluster_bootstrap_pairs_seed_and_writer() -> None:
    runs = {"unit": {}, "fallback": {}, "handle": {}}
    for seed in range(3):
        runs["unit"][seed] = [
            {
                "cluster_id": f"u{index}",
                "accuracy": 1.0,
                "binary_label": None,
                "positive_probability": None,
            }
            for index in range(5)
        ]
        runs["fallback"][seed] = [
            {
                "cluster_id": f"u{index}",
                "accuracy": 0.5,
                "binary_label": None,
                "positive_probability": None,
            }
            for index in range(5)
        ]
        runs["handle"][seed] = [
            {
                "cluster_id": f"u{index}",
                "accuracy": 0.7,
                "binary_label": None,
                "positive_probability": None,
            }
            for index in range(5)
        ]
    interval = hierarchical_cluster_effect_bootstrap(
        runs,
        "unit",
        ("fallback", "handle"),
        metric="accuracy",
        n_resamples=100,
        seed=1,
    )
    assert interval.estimate == pytest.approx(0.3)
    assert interval.minimum_clusters_per_seed == 5


def test_hierarchical_recovery_recomputes_ratio_inside_draws() -> None:
    runs = {"unit": {}, "fallback": {}, "full": {}}
    for seed in range(3):
        runs["unit"][seed] = [
            {
                "cluster_id": f"u{index}",
                "accuracy": 0.7,
                "binary_label": None,
                "positive_probability": None,
            }
            for index in range(6)
        ]
        runs["fallback"][seed] = [
            {
                "cluster_id": f"u{index}",
                "accuracy": 0.5,
                "binary_label": None,
                "positive_probability": None,
            }
            for index in range(6)
        ]
        runs["full"][seed] = [
            {
                "cluster_id": f"u{index}",
                "accuracy": 0.9,
                "binary_label": None,
                "positive_probability": None,
            }
            for index in range(6)
        ]
    interval = hierarchical_cluster_recovery_bootstrap(
        runs,
        "unit",
        "fallback",
        "full",
        metric="accuracy",
        n_resamples=100,
        seed=2,
    )
    assert interval.estimate == pytest.approx(0.5)
    assert interval.lower == pytest.approx(0.5)
    assert interval.upper == pytest.approx(0.5)
