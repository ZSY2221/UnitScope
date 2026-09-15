from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable, Mapping, Optional, Tuple


class LocalKind(str, Enum):
    ATOM = "local-atom"
    MIXED = "local-mixed"


class UnitKind(str, Enum):
    COMPLETE = "complete"
    PARTIAL_HANDLE = "partial-handle"
    LOCAL_ATOM = "local-atom"
    LOCAL_MIXED = "local-mixed"
    DROPPED = "dropped"
    NON_PRIVATE = "non-private"
    ORACLE_WEIGHTED = "oracle-weighted"
    GROUP_RECORD = "group-record"
    RECORD = "record"


class CertificateType(str, Enum):
    PLAN_C = "plan-c"
    PLAN_DELTA = "plan-delta"
    UNSAFE_DIAGNOSTIC = "unsafe-diagnostic"
    NON_PRIVATE = "non-private"
    RECORD_C = "record-c"


class MethodId(str, Enum):
    STABLE_LOCAL_FALLBACK = "stable-local-fallback"
    HANDLE_ONLY = "handle-only-drop-residual"
    UNITSCOPE_EQUAL = "unitscope-equal"
    UNITSCOPE_TUNED = "unitscope-tuned"
    UNITSCOPE_SWEEP = "unitscope-lambda-sweep"
    NAIVE_CLAIMED_C = "naive-claimed-c"
    NAIVE_RECALIBRATED_2C = "naive-recalibrated-2c"
    COMPLETE_ONLY = "complete-only-gate"
    FULL_IDENTITY = "full-identity-reference"
    ULDP_AVG_WEIGHTED = "uldp-avg-weighted-oracle"
    GROUP_PRIVACY_2 = "group-privacy-k2-person-corrected"
    RECORD_LEVEL_DP = "record-level-dp-sgd-reference"
    NON_PRIVATE = "non-private-reference"


@dataclass(frozen=True)
class PublicConfig:
    task_domain: str
    epoch: str
    clip_norm: float
    max_handle_groups: int
    max_fallback_slots: int
    lambda_: float
    public_normalizer: float
    noise_multiplier: float
    rounds: int
    delta: float
    n_silos: int
    use_sampling_amplification: bool = False
    query_reduction: str = "fixed-weighted-sum"

    def __post_init__(self) -> None:
        if not self.task_domain or not self.epoch:
            raise ValueError("task_domain and epoch must be non-empty")
        if not math.isfinite(self.clip_norm) or self.clip_norm <= 0:
            raise ValueError("clip_norm must be finite and positive")
        if self.max_handle_groups <= 0 or self.max_fallback_slots <= 0:
            raise ValueError("H and B must be positive public bounds")
        if not 0.0 <= self.lambda_ <= 1.0:
            raise ValueError("lambda must lie in [0, 1]")
        if not math.isfinite(self.public_normalizer) or self.public_normalizer <= 0:
            raise ValueError("public_normalizer must be finite and positive")
        if not math.isfinite(self.noise_multiplier) or self.noise_multiplier < 0:
            raise ValueError("noise_multiplier must be finite and non-negative")
        if self.rounds <= 0 or not 0 < self.delta < 1:
            raise ValueError("rounds must be positive and delta must lie in (0, 1)")
        if self.n_silos <= 0:
            raise ValueError("n_silos must be positive")
        if self.use_sampling_amplification:
            raise ValueError(
                "the frozen primary protocol conservatively disables sampling amplification"
            )
        if self.query_reduction != "fixed-weighted-sum":
            raise ValueError(
                "the primary protocol requires fixed-weighted-sum reduction"
            )

    @property
    def C(self) -> float:
        return self.clip_norm

    @property
    def H(self) -> int:
        return self.max_handle_groups

    @property
    def B(self) -> int:
        return self.max_fallback_slots

    @property
    def equal_lambda(self) -> float:
        return self.H / (self.H + self.B)

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PublicRecord:
    record_id: str
    silo_id: str
    fallback_slot: str
    local_kind: LocalKind = LocalKind.ATOM
    query_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.record_id or not self.silo_id or not self.fallback_slot:
            raise ValueError("record_id, silo_id, and fallback_slot are required")
        if not math.isfinite(self.query_weight) or self.query_weight < 0:
            raise ValueError("query_weight must be finite and non-negative")


@dataclass(frozen=True)
class PartialHandleClaim:
    claim_id: str
    issuer: str
    quota_slot: int
    record_ids: Tuple[str, ...]
    valid: bool = True

    def __post_init__(self) -> None:
        if not self.claim_id or not self.issuer:
            raise ValueError("claim_id and issuer are required")
        if self.quota_slot < 0:
            raise ValueError("quota_slot must be non-negative")
        if len(set(self.record_ids)) != len(self.record_ids):
            raise ValueError("a claim cannot repeat a record")


@dataclass(frozen=True)
class CompleteIdentityClaim:
    claim_id: str
    issuer: str
    record_ids: Tuple[str, ...]
    complete_domain_attestation: bool = True

    def __post_init__(self) -> None:
        if not self.claim_id or not self.issuer:
            raise ValueError("claim_id and issuer are required")
        if len(set(self.record_ids)) != len(self.record_ids):
            raise ValueError("a complete claim cannot repeat a record")


@dataclass(frozen=True)
class EvidenceView:
    task_domain: str
    epoch: str
    partial_handles: Tuple[PartialHandleClaim, ...] = ()
    complete_claims: Tuple[CompleteIdentityClaim, ...] = ()


@dataclass(frozen=True)
class PermanentRegistry:
    task_domain: str
    epoch: str
    record_universe: Tuple[str, ...]
    fallback_slots: Tuple[str, ...]
    handle_claim_ids: Tuple[str, ...] = ()
    complete_claim_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for values, name in (
            (self.record_universe, "record_universe"),
            (self.fallback_slots, "fallback_slots"),
            (self.handle_claim_ids, "handle_claim_ids"),
            (self.complete_claim_ids, "complete_claim_ids"),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")


@dataclass(frozen=True)
class GroundTruth:
    """Oracle-only record-to-person relation used by generators and evaluators."""

    assignments: Tuple[Tuple[str, str], ...]

    def __post_init__(self) -> None:
        record_ids = [record_id for record_id, _ in self.assignments]
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("ground truth must assign each record exactly once")

    def as_dict(self) -> Mapping[str, str]:
        return dict(self.assignments)


@dataclass(frozen=True)
class CertifiedSensitivity:
    value: float
    certificate_type: CertificateType
    basis: str

    def __post_init__(self) -> None:
        if self.certificate_type not in (
            CertificateType.PLAN_C,
            CertificateType.PLAN_DELTA,
            CertificateType.RECORD_C,
        ):
            raise ValueError(
                "only certified plan or record types can create CertifiedSensitivity"
            )
        if not math.isfinite(self.value) or self.value < 0:
            raise ValueError("certified sensitivity must be finite and non-negative")


@dataclass(frozen=True)
class PlanUnit:
    label: str
    kind: UnitKind
    record_ids: Tuple[str, ...]
    radius: float

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("unit label is required")
        if len(set(self.record_ids)) != len(self.record_ids):
            raise ValueError("a plan unit cannot repeat a record")
        if not math.isfinite(self.radius) or self.radius < 0:
            raise ValueError("unit radius must be finite and non-negative")


@dataclass(frozen=True)
class PlanValidation:
    valid: bool
    errors: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ContributionPlan:
    method: MethodId
    units: Tuple[PlanUnit, ...]
    config_hash: str
    certificate_type: CertificateType
    certified_sensitivity: Optional[CertifiedSensitivity]
    structural_sensitivity_bound: Optional[float]
    noise_sensitivity: float
    validation: PlanValidation

    def unit_for_record(self) -> Mapping[str, PlanUnit]:
        output = {}
        for unit in self.units:
            for record_id in unit.record_ids:
                if record_id in output:
                    raise ValueError(
                        f"record {record_id} appears in more than one plan unit"
                    )
                output[record_id] = unit
        return output

    @property
    def active_units(self) -> Tuple[PlanUnit, ...]:
        return tuple(
            unit
            for unit in self.units
            if unit.radius > 0 and unit.kind != UnitKind.DROPPED
        )


def canonical_label(*parts: object) -> str:
    encoded = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
    return f"unit:{digest}"


def ensure_unique_record_ids(records: Iterable[PublicRecord]) -> None:
    record_ids = [record.record_id for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("public records must have unique record_id values")
