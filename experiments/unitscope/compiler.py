from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import replace
from typing import List, MutableMapping, Sequence, Set, Tuple

from .model import (
    CertificateType,
    CertifiedSensitivity,
    ContributionPlan,
    EvidenceView,
    GroundTruth,
    LocalKind,
    MethodId,
    PermanentRegistry,
    PlanUnit,
    PlanValidation,
    PublicConfig,
    PublicRecord,
    UnitKind,
    canonical_label,
    ensure_unique_record_ids,
)


def _check_context(
    records: Sequence[PublicRecord],
    evidence: EvidenceView,
    registry: PermanentRegistry,
    config: PublicConfig,
) -> Tuple[str, ...]:
    errors: List[str] = []
    ensure_unique_record_ids(records)
    if (evidence.task_domain, evidence.epoch) != (config.task_domain, config.epoch):
        errors.append("evidence domain/epoch does not match public config")
    if (registry.task_domain, registry.epoch) != (config.task_domain, config.epoch):
        errors.append("registry domain/epoch does not match public config")

    universe = set(registry.record_universe)
    allowed_slots = set(registry.fallback_slots)
    for record in records:
        if record.record_id not in universe:
            errors.append(f"record {record.record_id} is outside the frozen universe")
        if record.fallback_slot not in allowed_slots:
            errors.append(
                f"fallback slot {record.fallback_slot} is not frozen in the registry"
            )

    allowed_handles = set(registry.handle_claim_ids)
    allowed_complete = set(registry.complete_claim_ids)
    active_record_ids = {record.record_id for record in records}
    claimed: Set[str] = set()
    for claim in evidence.complete_claims:
        if claim.claim_id not in allowed_complete:
            errors.append(
                f"complete claim {claim.claim_id} is not in the frozen registry"
            )
        outside = set(claim.record_ids) - universe
        if outside:
            errors.append(
                f"complete claim {claim.claim_id} references records outside the frozen universe"
            )
        if not claim.complete_domain_attestation:
            errors.append(
                f"complete claim {claim.claim_id} lacks a complete-domain attestation"
            )
        for record_id in set(claim.record_ids) & active_record_ids:
            if record_id in claimed:
                errors.append(
                    f"record {record_id} is covered by multiple identity claims"
                )
            claimed.add(record_id)
    for claim in evidence.partial_handles:
        if claim.claim_id not in allowed_handles:
            errors.append(
                f"partial claim {claim.claim_id} is not in the frozen registry"
            )
        outside = set(claim.record_ids) - universe
        if outside:
            errors.append(
                f"partial claim {claim.claim_id} references records outside the frozen universe"
            )
        if claim.quota_slot >= config.H:
            errors.append(f"partial claim {claim.claim_id} exceeds frozen H={config.H}")
        if not claim.valid:
            continue
        for record_id in set(claim.record_ids) & active_record_ids:
            if record_id in claimed:
                errors.append(
                    f"record {record_id} is covered by multiple identity claims"
                )
            claimed.add(record_id)
    return tuple(sorted(set(errors)))


def _method_lambda(method: MethodId, config: PublicConfig) -> float:
    if method == MethodId.UNITSCOPE_EQUAL:
        return config.equal_lambda
    if method in (MethodId.UNITSCOPE_TUNED, MethodId.UNITSCOPE_SWEEP):
        return config.lambda_
    if method == MethodId.STABLE_LOCAL_FALLBACK:
        return 0.0
    if method == MethodId.HANDLE_ONLY:
        return 1.0
    raise ValueError(f"method {method.value} does not use a UnitScope lambda")


def _local_radius(
    record: PublicRecord, total_local_budget: float, config: PublicConfig
) -> float:
    denominator = config.B if record.local_kind == LocalKind.ATOM else 2 * config.B
    return total_local_budget / denominator


def _append_grouped_local_units(
    units: List[PlanUnit],
    records: Sequence[PublicRecord],
    remaining_record_ids: Set[str],
    total_local_budget: float,
    config: PublicConfig,
    dropped: bool = False,
) -> None:
    records_by_slot: MutableMapping[str, List[PublicRecord]] = defaultdict(list)
    for record in records:
        if record.record_id in remaining_record_ids:
            records_by_slot[record.fallback_slot].append(record)
    for slot, members in sorted(records_by_slot.items()):
        local_kinds = {member.local_kind for member in members}
        if len(local_kinds) != 1:
            raise ValueError(
                f"frozen fallback slot {slot} changes local-kind semantics"
            )
        local_kind = next(iter(local_kinds))
        exemplar = members[0]
        radius = 0.0 if dropped else _local_radius(exemplar, total_local_budget, config)
        kind = UnitKind.DROPPED
        if not dropped:
            kind = (
                UnitKind.LOCAL_ATOM
                if local_kind == LocalKind.ATOM
                else UnitKind.LOCAL_MIXED
            )
        units.append(
            PlanUnit(
                label=canonical_label(
                    config.task_domain, config.epoch, "fallback", slot
                ),
                kind=kind,
                record_ids=tuple(sorted(member.record_id for member in members)),
                radius=radius,
            )
        )


def _compile_private_plan(
    method: MethodId,
    records: Sequence[PublicRecord],
    evidence: EvidenceView,
    config: PublicConfig,
) -> Tuple[List[PlanUnit], float, float, CertificateType, CertifiedSensitivity | None]:
    active = {record.record_id for record in records}
    remaining = set(active)
    units: List[PlanUnit] = []

    if method == MethodId.RECORD_LEVEL_DP:
        units = [
            PlanUnit(
                label=canonical_label(
                    config.task_domain,
                    config.epoch,
                    method.value,
                    record.record_id,
                ),
                kind=UnitKind.RECORD,
                record_ids=(record.record_id,),
                radius=config.C,
            )
            for record in sorted(records, key=lambda item: item.record_id)
        ]
        sensitivity = CertifiedSensitivity(
            config.C,
            CertificateType.RECORD_C,
            "one C-clipped contribution per record under record adjacency",
        )
        return (
            units,
            config.C,
            config.C,
            CertificateType.RECORD_C,
            sensitivity,
        )

    use_complete = method not in (MethodId.STABLE_LOCAL_FALLBACK, MethodId.HANDLE_ONLY)
    oracle_weighted_methods = (MethodId.ULDP_AVG_WEIGHTED,)
    group_privacy_methods = (MethodId.GROUP_PRIVACY_2,)
    if method in (
        MethodId.COMPLETE_ONLY,
        MethodId.FULL_IDENTITY,
        *oracle_weighted_methods,
        *group_privacy_methods,
    ):
        use_complete = True
    if method in oracle_weighted_methods:
        units = []
        remaining = set(active)
        record_by_id = {record.record_id: record for record in records}
        for claim in sorted(evidence.complete_claims, key=lambda item: item.claim_id):
            members = sorted(set(claim.record_ids) & active)
            if not members:
                continue
            by_silo: MutableMapping[str, List[str]] = defaultdict(list)
            for record_id in members:
                by_silo[record_by_id[record_id].silo_id].append(record_id)
            total = len(members)
            active_silos = len(by_silo)
            for silo_id, silo_members in sorted(by_silo.items()):
                radius = config.C * len(silo_members) / total
                units.append(
                    PlanUnit(
                        label=canonical_label(
                            config.task_domain,
                            config.epoch,
                            method.value,
                            claim.claim_id,
                            silo_id,
                        ),
                        kind=UnitKind.ORACLE_WEIGHTED,
                        record_ids=tuple(sorted(silo_members)),
                        radius=radius,
                    )
                )
            remaining.difference_update(members)
        if remaining:
            raise ValueError(
                f"{method.value} requires complete claims for every active record"
            )
        sensitivity = CertifiedSensitivity(
            config.C, CertificateType.PLAN_C, method.value
        )
        return units, config.C, config.C, CertificateType.PLAN_C, sensitivity

    if method in group_privacy_methods:
        k = 2
        units = []
        remaining = set(active)
        for claim in sorted(evidence.complete_claims, key=lambda item: item.claim_id):
            members = sorted(set(claim.record_ids) & active)
            if not members:
                continue
            # The public hash order is stable under deletion and prevents an
            # active record from refilling a vacated contribution slot.
            selected = sorted(
                members,
                key=lambda record_id: canonical_label(
                    config.task_domain,
                    config.epoch,
                    method.value,
                    claim.claim_id,
                    record_id,
                ),
            )[:k]
            selected_set = set(selected)
            for record_id in members:
                included = record_id in selected_set
                units.append(
                    PlanUnit(
                        label=canonical_label(
                            config.task_domain,
                            config.epoch,
                            method.value,
                            claim.claim_id,
                            record_id,
                        ),
                        kind=UnitKind.GROUP_RECORD if included else UnitKind.DROPPED,
                        record_ids=(record_id,),
                        radius=config.C if included else 0.0,
                    )
                )
            remaining.difference_update(members)
        if remaining:
            raise ValueError(
                f"{method.value} requires complete claims for every active record"
            )
        sensitivity_value = k * config.C
        sensitivity = CertifiedSensitivity(
            sensitivity_value,
            CertificateType.PLAN_DELTA,
            f"record clipping with a stable per-person cap k={k}",
        )
        return (
            units,
            sensitivity_value,
            sensitivity_value,
            CertificateType.PLAN_DELTA,
            sensitivity,
        )
    if use_complete:
        for claim in sorted(evidence.complete_claims, key=lambda item: item.claim_id):
            members = tuple(sorted(set(claim.record_ids) & active))
            if not members:
                continue
            units.append(
                PlanUnit(
                    label=canonical_label(
                        config.task_domain, config.epoch, "complete", claim.claim_id
                    ),
                    kind=UnitKind.COMPLETE,
                    record_ids=members,
                    radius=config.C,
                )
            )
            remaining.difference_update(members)

    if method == MethodId.FULL_IDENTITY:
        if remaining:
            raise ValueError(
                "full-identity reference requires complete claims for every active record"
            )
        sensitivity = CertifiedSensitivity(
            config.C, CertificateType.PLAN_C, "complete shared identity"
        )
        return units, config.C, config.C, CertificateType.PLAN_C, sensitivity

    if method == MethodId.COMPLETE_ONLY:
        _append_grouped_local_units(units, records, remaining, config.C, config)
        sensitivity = CertifiedSensitivity(
            config.C, CertificateType.PLAN_C, "complete claim or full fallback"
        )
        return units, config.C, config.C, CertificateType.PLAN_C, sensitivity

    if method in (
        MethodId.STABLE_LOCAL_FALLBACK,
        MethodId.HANDLE_ONLY,
        MethodId.UNITSCOPE_EQUAL,
        MethodId.UNITSCOPE_TUNED,
        MethodId.UNITSCOPE_SWEEP,
    ):
        lambda_ = _method_lambda(method, config)
        use_partial = lambda_ > 0
        partial_radius = lambda_ * config.C / config.H
        local_budget = (1.0 - lambda_) * config.C
        dropped = method == MethodId.HANDLE_ONLY or math.isclose(
            lambda_, 1.0, abs_tol=1e-12
        )
        structural_bound = config.C
        noise_sensitivity = config.C
        certificate_type = CertificateType.PLAN_C
        certificate = CertifiedSensitivity(
            config.C, certificate_type, "evidence-budget feasible plan"
        )
    elif method in (MethodId.NAIVE_CLAIMED_C, MethodId.NAIVE_RECALIBRATED_2C):
        use_partial = True
        partial_radius = config.C / config.H
        local_budget = config.C
        dropped = False
        structural_bound = 2.0 * config.C
        if method == MethodId.NAIVE_CLAIMED_C:
            noise_sensitivity = config.C
            certificate_type = CertificateType.UNSAFE_DIAGNOSTIC
            certificate = None
        else:
            noise_sensitivity = 2.0 * config.C
            certificate_type = CertificateType.PLAN_DELTA
            certificate = CertifiedSensitivity(
                2.0 * config.C, certificate_type, "two independent C-capacity paths"
            )
    else:
        raise ValueError(f"unsupported private method {method.value}")

    if use_partial:
        for claim in sorted(evidence.partial_handles, key=lambda item: item.claim_id):
            if not claim.valid:
                continue
            members = tuple(sorted(set(claim.record_ids) & remaining))
            if not members:
                continue
            units.append(
                PlanUnit(
                    label=canonical_label(
                        config.task_domain, config.epoch, "partial", claim.claim_id
                    ),
                    kind=UnitKind.PARTIAL_HANDLE,
                    record_ids=members,
                    radius=partial_radius,
                )
            )
            remaining.difference_update(members)

    _append_grouped_local_units(
        units, records, remaining, local_budget, config, dropped=dropped
    )
    return units, structural_bound, noise_sensitivity, certificate_type, certificate


def compile_plan(
    method: MethodId,
    records: Sequence[PublicRecord],
    evidence: EvidenceView,
    registry: PermanentRegistry,
    config: PublicConfig,
) -> ContributionPlan:
    """Compile evidence into a contribution plan without receiving oracle IDs."""

    context_errors = list(_check_context(records, evidence, registry, config))
    if method == MethodId.UNITSCOPE_EQUAL and not math.isclose(
        config.lambda_, config.equal_lambda, abs_tol=1e-12
    ):
        # Equal uses its derived lambda regardless of the tuned-lambda field, but
        # record the mismatch to keep manifests unambiguous.
        equal_warning = f"UnitScope-Equal overrides lambda={config.lambda_} with H/(H+B)={config.equal_lambda}"
    else:
        equal_warning = ""

    if method == MethodId.NON_PRIVATE:
        units = tuple(
            PlanUnit(
                label=canonical_label(
                    config.task_domain, config.epoch, "nonprivate", record.record_id
                ),
                kind=UnitKind.NON_PRIVATE,
                record_ids=(record.record_id,),
                radius=0.0,
            )
            for record in sorted(records, key=lambda item: item.record_id)
        )
        plan = ContributionPlan(
            method=method,
            units=units,
            config_hash=config.config_hash,
            certificate_type=CertificateType.NON_PRIVATE,
            certified_sensitivity=None,
            structural_sensitivity_bound=None,
            noise_sensitivity=0.0,
            validation=PlanValidation(not context_errors, tuple(context_errors)),
        )
        return plan

    units, structural_bound, noise_sensitivity, certificate_type, certificate = (
        _compile_private_plan(method, records, evidence, config)
    )
    assigned = [record_id for unit in units for record_id in unit.record_ids]
    active_ids = {record.record_id for record in records}
    assignment_counts = Counter(assigned)
    duplicate_ids = sorted(
        record_id for record_id, count in assignment_counts.items() if count > 1
    )
    missing_ids = sorted(active_ids - set(assigned))
    extra_ids = sorted(set(assigned) - active_ids)
    if duplicate_ids:
        context_errors.append(f"records assigned more than once: {duplicate_ids[:5]}")
    if missing_ids:
        context_errors.append(f"active records missing from plan: {missing_ids[:5]}")
    if extra_ids:
        context_errors.append(
            f"inactive records unexpectedly assigned: {extra_ids[:5]}"
        )
    warnings = (equal_warning,) if equal_warning else ()
    return ContributionPlan(
        method=method,
        units=tuple(sorted(units, key=lambda item: item.label)),
        config_hash=config.config_hash,
        certificate_type=certificate_type,
        certified_sensitivity=certificate,
        structural_sensitivity_bound=structural_bound,
        noise_sensitivity=noise_sensitivity,
        validation=PlanValidation(
            not context_errors, tuple(sorted(set(context_errors))), warnings
        ),
    )


def validate_plan_with_oracle(
    plan: ContributionPlan,
    records: Sequence[PublicRecord],
    truth: GroundTruth,
    config: PublicConfig,
) -> PlanValidation:
    """Validate a compiled plan against simulation ground truth."""

    errors = list(plan.validation.errors)
    warnings = list(plan.validation.warnings)
    truth_map = truth.as_dict()
    active_ids = {record.record_id for record in records}
    if active_ids - set(truth_map):
        errors.append("ground truth is missing active records")

    record_by_id = {record.record_id: record for record in records}
    users_to_records: MutableMapping[str, Set[str]] = defaultdict(set)
    for record_id in active_ids:
        if record_id in truth_map:
            users_to_records[truth_map[record_id]].add(record_id)

    partial_count: MutableMapping[str, int] = defaultdict(int)
    local_slots: MutableMapping[str, Set[str]] = defaultdict(set)
    capacity: MutableMapping[str, float] = defaultdict(float)
    complete_users: Set[str] = set()

    for user_id, user_records in users_to_records.items():
        local_slots[user_id] = {
            record_by_id[record_id].fallback_slot for record_id in user_records
        }
        if len(local_slots[user_id]) > config.B:
            errors.append(
                f"oracle user {user_id} reaches {len(local_slots[user_id])} fallback slots > B={config.B}"
            )

    for unit in plan.units:
        unit_users = {
            truth_map[record_id]
            for record_id in unit.record_ids
            if record_id in truth_map
        }
        if (
            unit.kind
            in (
                UnitKind.COMPLETE,
                UnitKind.PARTIAL_HANDLE,
                UnitKind.LOCAL_ATOM,
                UnitKind.ORACLE_WEIGHTED,
                UnitKind.GROUP_RECORD,
            )
            and len(unit_users) > 1
        ):
            errors.append(
                f"unit {unit.label} claims purity but contains {len(unit_users)} oracle users"
            )
        for user_id in unit_users:
            factor = 2.0 if unit.kind == UnitKind.LOCAL_MIXED else 1.0
            if len(unit_users) > 1 and unit.kind in (
                UnitKind.COMPLETE,
                UnitKind.PARTIAL_HANDLE,
            ):
                factor = 2.0
            capacity[user_id] += factor * unit.radius
            if unit.kind == UnitKind.PARTIAL_HANDLE:
                partial_count[user_id] += 1
            if unit.kind == UnitKind.COMPLETE:
                complete_users.add(user_id)
                if set(unit.record_ids) != users_to_records[user_id]:
                    errors.append(
                        f"complete unit {unit.label} does not cover all active records of oracle user {user_id}"
                    )

    for user_id, count in partial_count.items():
        if count > config.H:
            errors.append(
                f"oracle user {user_id} reaches {count} partial handles > H={config.H}"
            )
    for user_id in complete_users:
        noncomplete = [
            unit.label
            for unit in plan.units
            if unit.kind != UnitKind.COMPLETE
            and any(
                truth_map.get(record_id) == user_id for record_id in unit.record_ids
            )
        ]
        if noncomplete:
            errors.append(f"complete user {user_id} also reaches non-complete paths")

    if plan.structural_sensitivity_bound is not None:
        tolerance = 1e-9 * max(1.0, plan.structural_sensitivity_bound)
        for user_id, value in capacity.items():
            if value > plan.structural_sensitivity_bound + tolerance:
                errors.append(
                    f"oracle user {user_id} has path capacity {value:.12g} > "
                    f"bound {plan.structural_sensitivity_bound:.12g}"
                )
    return PlanValidation(
        not errors, tuple(sorted(set(errors))), tuple(sorted(set(warnings)))
    )


def validate_record_level_plan(
    plan: ContributionPlan,
    records: Sequence[PublicRecord],
    config: PublicConfig,
) -> PlanValidation:
    """Validate a record-adjacent plan without making a user-level claim."""

    errors = list(plan.validation.errors)
    warnings = list(plan.validation.warnings)
    active_ids = {record.record_id for record in records}
    assigned = [record_id for unit in plan.units for record_id in unit.record_ids]
    if set(assigned) != active_ids or len(assigned) != len(active_ids):
        errors.append("record-level plan must assign every active record exactly once")
    for unit in plan.units:
        if unit.kind != UnitKind.RECORD or len(unit.record_ids) != 1:
            errors.append(f"unit {unit.label} is not a singleton record unit")
        if unit.radius > config.C + 1e-12:
            errors.append(f"unit {unit.label} exceeds the public record clip norm C")
    if plan.certificate_type != CertificateType.RECORD_C:
        errors.append("record-level plan lacks a RECORD_C certificate")
    if plan.certified_sensitivity is None or not math.isclose(
        plan.certified_sensitivity.value, config.C, rel_tol=0.0, abs_tol=1e-12
    ):
        errors.append("record-level plan must certify sensitivity C")
    if not math.isclose(plan.noise_sensitivity, config.C, rel_tol=0.0, abs_tol=1e-12):
        errors.append("record-level plan noise must be calibrated to C")
    return PlanValidation(
        not errors,
        tuple(sorted(set(errors))),
        tuple(sorted(set(warnings))),
    )


def with_oracle_validation(
    plan: ContributionPlan, validation: PlanValidation
) -> ContributionPlan:
    return replace(plan, validation=validation)
