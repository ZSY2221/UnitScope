from __future__ import annotations

import pandas as pd

import pytest

from experiments.unitscope.identity_wur import (
    evaluate_cached_pair_scores,
    linkage_metrics,
)


def test_headline_pair_f1_includes_true_pairs_missed_by_candidates() -> None:
    predicted = {"a": "ab", "b": "ab", "c": "c"}
    truth = {"a": "abc", "b": "abc", "c": "abc"}
    metrics = linkage_metrics(predicted, truth, {("a", "b")})
    assert metrics.pair_precision == 1.0
    assert metrics.pair_recall == pytest.approx(1 / 3)
    assert metrics.pair_f1 == pytest.approx(0.5)
    assert metrics.candidate_recall == pytest.approx(1 / 3)
    assert metrics.conditional_candidate_f1 == 1.0


def test_bridge_deletion_creates_large_wur_despite_high_pair_quality() -> None:
    records = pd.DataFrame(
        {
            "record_id": ["a1", "a2", "bridge", "b1", "b2"],
            "true_entity_id": ["A", "A", "X", "B", "B"],
        }
    )
    pairs = pd.DataFrame(
        {
            "left_id": ["a1", "a2", "bridge", "bridge", "b1"],
            "right_id": ["a2", "bridge", "b1", "b2", "b2"],
            "score": [0.99, 0.95, 0.95, 0.95, 0.99],
        }
    )
    metrics, witnesses = evaluate_cached_pair_scores(records, pairs, threshold=0.9)
    bridge = next(
        witness
        for witness in witnesses
        if witness.removed_records == 1 and witness.changed_new_components == 2
    )
    assert bridge.witness_wur_over_c == 3.0
    assert bridge.surviving_records_reassigned == 4
    assert 0 <= metrics.pair_f1 <= 1
