"""Coverage-adaptive UnitScope allocation with private coverage releases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LOW_MIDDLE_THRESHOLD = 0.33
MIDDLE_HIGH_THRESHOLD = 0.66
GATE_EPSILON_EACH = 0.05
GATE_DELTA_EACH = 1e-12
LEARNING_EPSILON = 3.9
TOTAL_EPSILON = 4.0
PUBLIC_DENOMINATOR = 1600

ALLOCATION_TABLE = (
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.9, 0.9, 0.9),
)


def coverage_bin(value: float) -> int:
    if not 0.0 <= value <= 1.0:
        raise ValueError("coverage must be in [0, 1]")
    if value < LOW_MIDDLE_THRESHOLD:
        return 0
    if value < MIDDLE_HIGH_THRESHOLD:
        return 1
    return 2


def select_lambda(
    released_user_coverage: float, released_record_coverage: float
) -> float:
    return ALLOCATION_TABLE[coverage_bin(released_user_coverage)][
        coverage_bin(released_record_coverage)
    ]


@dataclass(frozen=True)
class PrivateCoverageRelease:
    released_user_coverage: float
    released_record_coverage: float
    selected_lambda: float
    epsilon_each: float = GATE_EPSILON_EACH
    delta_each: float = GATE_DELTA_EACH


def release_and_select(
    user_coverage: float,
    record_coverage: float,
    *,
    rng: np.random.Generator,
    public_denominator: int = PUBLIC_DENOMINATOR,
) -> PrivateCoverageRelease:
    if public_denominator <= 0:
        raise ValueError("public_denominator must be positive")
    coverage_bin(user_coverage)
    coverage_bin(record_coverage)
    scale = 1.0 / (public_denominator * GATE_EPSILON_EACH)
    released = np.clip(
        np.asarray((user_coverage, record_coverage)) + rng.laplace(0.0, scale, size=2),
        0.0,
        1.0,
    )
    released_user, released_record = map(float, released)
    return PrivateCoverageRelease(
        released_user_coverage=released_user,
        released_record_coverage=released_record,
        selected_lambda=select_lambda(released_user, released_record),
    )


def composed_epsilon() -> float:
    return LEARNING_EPSILON + 2 * GATE_EPSILON_EACH
