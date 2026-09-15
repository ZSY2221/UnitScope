from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audit import require_certified_sensitivity
from .model import CertifiedSensitivity


DEFAULT_ORDERS = np.asarray(
    [1.25, 1.5, 1.75, 2, 3, 4, 5, 8, 10, 16, 20, 32, 64, 128, 256], dtype=np.float64
)


@dataclass(frozen=True)
class PrivacyReport:
    epsilon: float
    delta: float
    optimal_order: float
    rounds: int
    sensitivity: float
    noise_std: float
    sampling_amplification_used: bool = False


def conservative_gaussian_rdp(
    sensitivity: CertifiedSensitivity,
    noise_std: float,
    rounds: int,
    delta: float,
    orders: np.ndarray = DEFAULT_ORDERS,
) -> PrivacyReport:
    certified = require_certified_sensitivity(sensitivity)
    if noise_std <= 0 or rounds <= 0 or not 0 < delta < 1:
        raise ValueError(
            "noise_std and rounds must be positive; delta must lie in (0, 1)"
        )
    rdp = rounds * orders * certified.value**2 / (2.0 * noise_std**2)
    eps = rdp + np.log(1.0 / delta) / (orders - 1.0)
    index = int(np.argmin(eps))
    return PrivacyReport(
        epsilon=float(eps[index]),
        delta=delta,
        optimal_order=float(orders[index]),
        rounds=rounds,
        sensitivity=certified.value,
        noise_std=noise_std,
        sampling_amplification_used=False,
    )


def calibrate_noise_multiplier(
    target_epsilon: float, rounds: int, delta: float
) -> float:
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive")
    low, high = 1e-4, 1.0
    unit = CertifiedSensitivity(
        1.0, certificate_type=_plan_c(), basis="unit calibration"
    )
    while conservative_gaussian_rdp(unit, high, rounds, delta).epsilon > target_epsilon:
        high *= 2.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if (
            conservative_gaussian_rdp(unit, middle, rounds, delta).epsilon
            > target_epsilon
        ):
            low = middle
        else:
            high = middle
    return high


def _plan_c():
    # Local import avoids exporting a second certificate constructor.
    from .model import CertificateType

    return CertificateType.PLAN_C
