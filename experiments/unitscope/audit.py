from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np

from .compiler import compile_plan
from .executor import execute_release
from .model import (
    CertifiedSensitivity,
    ContributionPlan,
    EvidenceView,
    MethodId,
    PermanentRegistry,
    PublicConfig,
    PublicRecord,
)


@dataclass(frozen=True)
class ObservedWUR:
    value: float
    changed_old_labels: Tuple[str, ...]
    changed_new_labels: Tuple[str, ...]
    label: str = "observed-not-certified"


@dataclass(frozen=True)
class WURReport:
    observed: ObservedWUR
    actual_query_difference: float | None = None


def _unit_map(plan: ContributionPlan) -> Mapping[str, object]:
    return {unit.label: unit for unit in plan.units}


def audit_plan_transition(
    old_plan: ContributionPlan, new_plan: ContributionPlan
) -> ObservedWUR:
    """Compute the weighted plan edit for one observed adjacency."""

    old_units = _unit_map(old_plan)
    new_units = _unit_map(new_plan)
    changed_old = []
    changed_new = []
    value = 0.0
    for label in sorted(set(old_units) | set(new_units)):
        old = old_units.get(label)
        new = new_units.get(label)
        old_signature = (
            None if old is None else (old.kind, frozenset(old.record_ids), old.radius)
        )
        new_signature = (
            None if new is None else (new.kind, frozenset(new.record_ids), new.radius)
        )
        if old_signature == new_signature:
            continue
        if old is not None:
            changed_old.append(label)
            value += old.radius
        if new is not None:
            changed_new.append(label)
            value += new.radius
    return ObservedWUR(value, tuple(changed_old), tuple(changed_new))


def full_recompile_audit(
    method: MethodId,
    old_records: Sequence[PublicRecord],
    new_records: Sequence[PublicRecord],
    evidence: EvidenceView,
    registry: PermanentRegistry,
    config: PublicConfig,
    old_vectors: np.ndarray | None = None,
    new_vectors: np.ndarray | None = None,
) -> WURReport:
    old_plan = compile_plan(method, old_records, evidence, registry, config)
    new_plan = compile_plan(method, new_records, evidence, registry, config)
    observed = audit_plan_transition(old_plan, new_plan)
    actual = None
    if old_vectors is not None or new_vectors is not None:
        if old_vectors is None or new_vectors is None:
            raise ValueError("old_vectors and new_vectors must be supplied together")
        zero_old = np.zeros(np.asarray(old_vectors).shape[1])
        zero_new = np.zeros(np.asarray(new_vectors).shape[1])
        old_release = execute_release(
            old_records,
            old_vectors,
            old_plan,
            config,
            zero_old,
            allow_unsafe_diagnostic=True,
        )
        new_release = execute_release(
            new_records,
            new_vectors,
            new_plan,
            config,
            zero_new,
            allow_unsafe_diagnostic=True,
        )
        # WUR is expressed before the public normalization, so report the
        # actual query difference in the same units.
        actual = float(
            np.linalg.norm(
                old_release.deterministic_value - new_release.deterministic_value
            )
            * config.public_normalizer
        )
    return WURReport(observed, actual)


def require_certified_sensitivity(value: object) -> CertifiedSensitivity:
    if not isinstance(value, CertifiedSensitivity):
        raise TypeError(
            "the accountant accepts only CertifiedSensitivity, never observed or witness WUR"
        )
    return value
