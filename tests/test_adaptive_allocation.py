from __future__ import annotations

import numpy as np
import pytest

from experiments.unitscope.adaptive_allocation import (
    ALLOCATION_TABLE,
    composed_epsilon,
    release_and_select,
    select_lambda,
)


def test_final_allocation_table() -> None:
    assert ALLOCATION_TABLE == (
        (0.0, 0.0, 0.0),
        (0.0, 0.5, 0.5),
        (0.9, 0.9, 0.9),
    )
    values = (0.25, 0.5, 0.75)
    observed = tuple(tuple(select_lambda(u, r) for r in values) for u in values)
    assert observed == ALLOCATION_TABLE


def test_privacy_budget_composes_to_four() -> None:
    assert composed_epsilon() == pytest.approx(4.0)


def test_private_release_is_bounded_and_selects_a_table_value() -> None:
    result = release_and_select(0.75, 0.5, rng=np.random.default_rng(980000001))
    assert 0.0 <= result.released_user_coverage <= 1.0
    assert 0.0 <= result.released_record_coverage <= 1.0
    assert result.selected_lambda in {0.0, 0.5, 0.9}
