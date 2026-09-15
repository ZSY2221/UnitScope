from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, MutableMapping, Sequence

import numpy as np

from .model import (
    CompleteIdentityClaim,
    EvidenceView,
    GroundTruth,
    PartialHandleClaim,
    PermanentRegistry,
    PublicConfig,
    PublicRecord,
)


def stable_rank(seed: int, *parts: object) -> int:
    value = "|".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


@dataclass(frozen=True)
class EvidencePolicy:
    handle_user_probability: float
    record_coverage_given_handle: float
    complete_user_probability: float = 0.0
    handles_per_user: int = 1
    issuer: str = "regional-credential-root"

    def __post_init__(self) -> None:
        for value, name in (
            (self.handle_user_probability, "q_user"),
            (self.record_coverage_given_handle, "q_record_given_handle"),
            (self.complete_user_probability, "q_complete"),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.handles_per_user <= 0:
            raise ValueError("handles_per_user must be positive")


@dataclass(frozen=True)
class GeneratedEvidence:
    evidence: EvidenceView
    registry: PermanentRegistry
    realized_handle_user_fraction: float
    realized_record_coverage_given_handle: float
    realized_pooled_record_coverage_given_handle: float
    realized_total_record_coverage: float
    realized_complete_user_fraction: float
    handle_users_with_residual_fraction: float
    handle_users_fully_covered_fraction: float
    mean_residual_records_per_handle_user: float


def _select_exact_fraction(
    values: Sequence[str], fraction: float, seed: int, label: str
) -> set[str]:
    count = int(round(fraction * len(values)))
    ordered = sorted(values, key=lambda value: stable_rank(seed, label, value))
    return set(ordered[:count])


def generate_evidence_view(
    records: Sequence[PublicRecord],
    truth: GroundTruth,
    config: PublicConfig,
    policy: EvidencePolicy,
    seed: int,
) -> GeneratedEvidence:
    """Generate an evidence view from simulation ground truth."""

    if policy.handles_per_user > config.H:
        raise ValueError("the policy cannot issue more than the public H bound")
    truth_map = truth.as_dict()
    by_user: MutableMapping[str, List[PublicRecord]] = {}
    for record in records:
        if record.record_id not in truth_map:
            raise ValueError(f"ground truth is missing record {record.record_id}")
        by_user.setdefault(truth_map[record.record_id], []).append(record)
    users = sorted(by_user)
    complete_users = _select_exact_fraction(
        users, policy.complete_user_probability, seed, "complete-user"
    )
    remaining_users = [user for user in users if user not in complete_users]
    handle_users = _select_exact_fraction(
        remaining_users, policy.handle_user_probability, seed, "handle-user"
    )

    complete_claims: List[CompleteIdentityClaim] = []
    partial_claims: List[PartialHandleClaim] = []
    covered_records = 0
    handle_user_records = 0
    handled_user_coverages: List[float] = []
    handled_user_residual_counts: List[int] = []
    for user in users:
        user_records = sorted(
            by_user[user], key=lambda item: stable_rank(seed, "record", item.record_id)
        )
        if user in complete_users:
            claim_id = f"complete-{stable_rank(seed, config.task_domain, user):016x}"
            complete_claims.append(
                CompleteIdentityClaim(
                    claim_id=claim_id,
                    issuer=policy.issuer,
                    record_ids=tuple(record.record_id for record in user_records),
                    complete_domain_attestation=True,
                )
            )
            continue
        if user not in handle_users:
            continue
        selected_count = int(
            round(policy.record_coverage_given_handle * len(user_records))
        )
        if policy.record_coverage_given_handle > 0 and user_records:
            selected_count = max(1, selected_count)
        selected = user_records[:selected_count]
        if not selected:
            continue
        handle_user_records += len(user_records)
        covered_records += len(selected)
        handled_user_coverages.append(len(selected) / len(user_records))
        handled_user_residual_counts.append(len(user_records) - len(selected))
        n_handles = min(policy.handles_per_user, len(selected))
        shards: List[List[PublicRecord]] = [[] for _ in range(n_handles)]
        for index, record in enumerate(selected):
            shards[index % n_handles].append(record)
        for quota_slot, shard in enumerate(shards):
            claim_id = f"partial-{stable_rank(seed, config.task_domain, user, quota_slot):016x}"
            partial_claims.append(
                PartialHandleClaim(
                    claim_id=claim_id,
                    issuer=policy.issuer,
                    quota_slot=quota_slot,
                    record_ids=tuple(sorted(record.record_id for record in shard)),
                    valid=True,
                )
            )

    evidence = EvidenceView(
        task_domain=config.task_domain,
        epoch=config.epoch,
        partial_handles=tuple(sorted(partial_claims, key=lambda item: item.claim_id)),
        complete_claims=tuple(sorted(complete_claims, key=lambda item: item.claim_id)),
    )
    registry = PermanentRegistry(
        task_domain=config.task_domain,
        epoch=config.epoch,
        record_universe=tuple(sorted(record.record_id for record in records)),
        fallback_slots=tuple(sorted({record.fallback_slot for record in records})),
        handle_claim_ids=tuple(claim.claim_id for claim in evidence.partial_handles),
        complete_claim_ids=tuple(claim.claim_id for claim in evidence.complete_claims),
    )
    return GeneratedEvidence(
        evidence=evidence,
        registry=registry,
        realized_handle_user_fraction=len(handled_user_coverages) / len(users)
        if users
        else 0.0,
        realized_record_coverage_given_handle=float(np.mean(handled_user_coverages))
        if handled_user_coverages
        else 0.0,
        realized_pooled_record_coverage_given_handle=covered_records
        / handle_user_records
        if handle_user_records
        else 0.0,
        realized_total_record_coverage=covered_records / len(records)
        if records
        else 0.0,
        realized_complete_user_fraction=len(complete_users) / len(users)
        if users
        else 0.0,
        handle_users_with_residual_fraction=sum(
            value > 0 for value in handled_user_residual_counts
        )
        / len(handled_user_residual_counts)
        if handled_user_residual_counts
        else 0.0,
        handle_users_fully_covered_fraction=sum(
            value == 0 for value in handled_user_residual_counts
        )
        / len(handled_user_residual_counts)
        if handled_user_residual_counts
        else 0.0,
        mean_residual_records_per_handle_user=float(
            np.mean(handled_user_residual_counts)
        )
        if handled_user_residual_counts
        else 0.0,
    )


def invalidate_claim(evidence: EvidenceView, claim_id: str) -> EvidenceView:
    partial = tuple(
        PartialHandleClaim(
            claim.claim_id, claim.issuer, claim.quota_slot, claim.record_ids, False
        )
        if claim.claim_id == claim_id
        else claim
        for claim in evidence.partial_handles
    )
    return EvidenceView(
        evidence.task_domain, evidence.epoch, partial, evidence.complete_claims
    )
