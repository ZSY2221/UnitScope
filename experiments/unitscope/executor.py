from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .model import (
    CertificateType,
    ContributionPlan,
    PublicConfig,
    PublicRecord,
    UnitKind,
)


@dataclass(frozen=True)
class ExecutionDiagnostics:
    records: int
    active_units: int
    dropped_records: int
    clipped_units: int
    clip_fraction: float
    preclip_norm_mean: float
    preclip_norm_max: float
    signal_norm: float
    noise_norm: float
    aggregate_snr: float


@dataclass(frozen=True)
class QueryRelease:
    value: np.ndarray
    deterministic_value: np.ndarray
    noise: np.ndarray
    diagnostics: ExecutionDiagnostics


def _validate_vectors(
    records: Sequence[PublicRecord], record_vectors: np.ndarray, plan: ContributionPlan
) -> Mapping[str, int]:
    if not plan.validation.valid:
        raise ValueError(f"refusing to execute invalid plan: {plan.validation.errors}")
    array = np.asarray(record_vectors)
    if array.ndim != 2 or array.shape[0] != len(records):
        raise ValueError(
            "record_vectors must have shape [number of records, vector dimension]"
        )
    record_index = {record.record_id: index for index, record in enumerate(records)}
    if len(record_index) != len(records):
        raise ValueError("record IDs must be unique")
    if set(plan.unit_for_record()) != set(record_index):
        raise ValueError("plan assignment must match the active record set exactly")
    return record_index


def reduce_by_plan(
    records: Sequence[PublicRecord],
    record_vectors: np.ndarray,
    plan: ContributionPlan,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the fixed weighted-sum reduction to each plan unit."""

    vectors = np.asarray(record_vectors, dtype=np.float64)
    record_index = _validate_vectors(records, vectors, plan)
    weights = {record.record_id: record.query_weight for record in records}
    unit_vectors = np.zeros((len(plan.units), vectors.shape[1]), dtype=np.float64)
    radii = np.zeros(len(plan.units), dtype=np.float64)
    for unit_index, unit in enumerate(plan.units):
        if unit.record_ids:
            indices = [record_index[record_id] for record_id in unit.record_ids]
            unit_weights = np.asarray(
                [weights[record_id] for record_id in unit.record_ids], dtype=np.float64
            )
            unit_vectors[unit_index] = (vectors[indices] * unit_weights[:, None]).sum(
                axis=0
            )
        radii[unit_index] = unit.radius
    return unit_vectors, radii


def clip_plan_units(
    unit_vectors: np.ndarray, radii: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = np.asarray(unit_vectors, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    if vectors.ndim != 2 or radii.shape != (vectors.shape[0],):
        raise ValueError("unit vectors and radii have incompatible shapes")
    norms = np.linalg.norm(vectors, axis=1)
    factors = np.ones_like(norms)
    positive = norms > 0
    factors[positive] = np.minimum(1.0, radii[positive] / norms[positive])
    factors[radii == 0] = 0.0
    return vectors * factors[:, None], norms, factors


def unpartitioned_weighted_sum(
    records: Sequence[PublicRecord], record_vectors: np.ndarray
) -> np.ndarray:
    vectors = np.asarray(record_vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != len(records):
        raise ValueError(
            "record_vectors must have shape [number of records, vector dimension]"
        )
    weights = np.asarray([record.query_weight for record in records], dtype=np.float64)
    return (vectors * weights[:, None]).sum(axis=0)


def execute_release(
    records: Sequence[PublicRecord],
    record_vectors: np.ndarray,
    plan: ContributionPlan,
    config: PublicConfig,
    standard_normal: np.ndarray | None,
    *,
    allow_unsafe_diagnostic: bool = False,
) -> QueryRelease:
    if plan.config_hash != config.config_hash:
        raise ValueError("plan was compiled under a different immutable public config")
    vectors = np.asarray(record_vectors, dtype=np.float64)
    _validate_vectors(records, vectors, plan)

    if plan.certificate_type == CertificateType.NON_PRIVATE:
        deterministic_sum = unpartitioned_weighted_sum(records, vectors)
        noise = np.zeros(vectors.shape[1], dtype=np.float64)
        preclip_norms = np.asarray([np.linalg.norm(row) for row in vectors])
        factors = np.ones_like(preclip_norms)
    else:
        if (
            plan.certificate_type == CertificateType.UNSAFE_DIAGNOSTIC
            and not allow_unsafe_diagnostic
        ):
            raise ValueError(
                "unsafe diagnostic execution requires explicit allow_unsafe_diagnostic=True"
            )
        if plan.certificate_type not in (
            CertificateType.PLAN_C,
            CertificateType.PLAN_DELTA,
            CertificateType.RECORD_C,
            CertificateType.UNSAFE_DIAGNOSTIC,
        ):
            raise ValueError("unsupported plan certificate type")
        unit_vectors, radii = reduce_by_plan(records, vectors, plan)
        clipped, preclip_norms, factors = clip_plan_units(unit_vectors, radii)
        deterministic_sum = clipped.sum(axis=0)
        if standard_normal is None:
            raise ValueError(
                "private and diagnostic releases require an explicit paired standard-normal vector"
            )
        z = np.asarray(standard_normal, dtype=np.float64)
        if z.shape != (vectors.shape[1],):
            raise ValueError("standard_normal must match the query dimension")
        noise = z * config.noise_multiplier * plan.noise_sensitivity

    if plan.certificate_type == CertificateType.NON_PRIVATE:
        diagnostic_norms = preclip_norms
        diagnostic_factors = factors
    else:
        active_mask = np.asarray(
            [unit.radius > 0 and unit.kind != UnitKind.DROPPED for unit in plan.units],
            dtype=bool,
        )
        diagnostic_norms = preclip_norms[active_mask]
        diagnostic_factors = factors[active_mask]

    value = (deterministic_sum + noise) / config.public_normalizer
    deterministic_value = deterministic_sum / config.public_normalizer
    scaled_noise = noise / config.public_normalizer
    signal_norm = float(np.linalg.norm(deterministic_value))
    noise_norm = float(np.linalg.norm(scaled_noise))
    dropped_records = (
        0
        if plan.certificate_type == CertificateType.NON_PRIVATE
        else sum(
            len(unit.record_ids) for unit in plan.units if unit.kind == UnitKind.DROPPED
        )
    )
    diagnostics = ExecutionDiagnostics(
        records=len(records),
        active_units=len(plan.active_units),
        dropped_records=dropped_records,
        clipped_units=int(np.sum(diagnostic_factors < 1.0 - 1e-12)),
        clip_fraction=float(np.mean(diagnostic_factors < 1.0 - 1e-12))
        if len(diagnostic_factors)
        else 0.0,
        preclip_norm_mean=float(np.mean(diagnostic_norms))
        if len(diagnostic_norms)
        else 0.0,
        preclip_norm_max=float(np.max(diagnostic_norms))
        if len(diagnostic_norms)
        else 0.0,
        signal_norm=signal_norm,
        noise_norm=noise_norm,
        aggregate_snr=signal_norm / noise_norm if noise_norm > 0 else float("inf"),
    )
    return QueryRelease(value, deterministic_value, scaled_noise, diagnostics)


def assert_partition_invariant(
    records: Sequence[PublicRecord],
    record_vectors: np.ndarray,
    plans: Sequence[ContributionPlan],
) -> None:
    target = unpartitioned_weighted_sum(records, record_vectors)
    for plan in plans:
        unit_vectors, _ = reduce_by_plan(records, record_vectors, plan)
        if not np.allclose(unit_vectors.sum(axis=0), target, rtol=1e-12, atol=1e-12):
            raise AssertionError(
                f"unclipped query changed under plan {plan.method.value}"
            )
