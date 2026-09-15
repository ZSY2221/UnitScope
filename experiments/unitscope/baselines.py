from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict

from .model import MethodId


class ImplementationProvenance(str, Enum):
    THIS_WORK = "this-work"
    ADAPTED = "adapted"
    ORACLE = "oracle"


class PrivacySemantics(str, Enum):
    USER_CERTIFIED = "user-certified"
    ORACLE_ASSUMPTION = "oracle-assumption"
    RECORD_DP = "record-dp"
    UNSAFE_DIAGNOSTIC = "unsafe-diagnostic"
    NON_PRIVATE = "non-private"


class ComparisonTier(str, Enum):
    PRIMARY = "primary"
    FULL_IDENTITY_REFERENCE = "full-identity-reference"
    SECONDARY = "secondary"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    display_name: str
    provenance: ImplementationProvenance
    privacy_semantics: PrivacySemantics
    comparison_tier: ComparisonTier
    identity_requirement: str
    residual_handling: str
    notes: str = ""


PRIMARY_METHODS: Dict[MethodId, MethodSpec] = {
    MethodId.STABLE_LOCAL_FALLBACK: MethodSpec(
        MethodId.STABLE_LOCAL_FALLBACK.value,
        "Stable-Local Fallback",
        ImplementationProvenance.THIS_WORK,
        PrivacySemantics.USER_CERTIFIED,
        ComparisonTier.PRIMARY,
        "stable institution-local slots",
        "retained under the full local capacity",
    ),
    MethodId.HANDLE_ONLY: MethodSpec(
        MethodId.HANDLE_ONLY.value,
        "Handle-Only Drop-Residual",
        ImplementationProvenance.THIS_WORK,
        PrivacySemantics.USER_CERTIFIED,
        ComparisonTier.PRIMARY,
        "positive partial-handle claims",
        "discarded with zero contribution",
    ),
    MethodId.UNITSCOPE_EQUAL: MethodSpec(
        MethodId.UNITSCOPE_EQUAL.value,
        "UnitScope-Equal",
        ImplementationProvenance.THIS_WORK,
        PrivacySemantics.USER_CERTIFIED,
        ComparisonTier.PRIMARY,
        "positive partial-handle claims and frozen H/B",
        "retained; lambda is H/(H+B)",
    ),
    MethodId.UNITSCOPE_TUNED: MethodSpec(
        MethodId.UNITSCOPE_TUNED.value,
        "UnitScope-Tuned",
        ImplementationProvenance.THIS_WORK,
        PrivacySemantics.USER_CERTIFIED,
        ComparisonTier.PRIMARY,
        "positive partial-handle claims and frozen H/B",
        "retained; lambda selected by a frozen public/independent policy",
    ),
    MethodId.UNITSCOPE_SWEEP: MethodSpec(
        MethodId.UNITSCOPE_SWEEP.value,
        "UnitScope Lambda Sweep",
        ImplementationProvenance.THIS_WORK,
        PrivacySemantics.USER_CERTIFIED,
        ComparisonTier.DIAGNOSTIC,
        "positive partial-handle claims and frozen H/B",
        "retained; lambda is a preregistered descriptive grid value",
        "Never reported as the publicly selected UnitScope-Tuned result.",
    ),
    MethodId.NAIVE_RECALIBRATED_2C: MethodSpec(
        MethodId.NAIVE_RECALIBRATED_2C.value,
        "Naive-Recalibrated-2C",
        ImplementationProvenance.THIS_WORK,
        PrivacySemantics.USER_CERTIFIED,
        ComparisonTier.PRIMARY,
        "positive partial-handle claims",
        "retained with a second full C capacity and 2C-calibrated noise",
    ),
    MethodId.FULL_IDENTITY: MethodSpec(
        MethodId.FULL_IDENTITY.value,
        "Full-Identity Reference",
        ImplementationProvenance.ORACLE,
        PrivacySemantics.ORACLE_ASSUMPTION,
        ComparisonTier.FULL_IDENTITY_REFERENCE,
        "complete shared identity over the task domain",
        "jointly grouped",
    ),
    MethodId.ULDP_AVG_WEIGHTED: MethodSpec(
        MethodId.ULDP_AVG_WEIGHTED.value,
        "ULDP-FL AVG-w Oracle",
        ImplementationProvenance.ADAPTED,
        PrivacySemantics.ORACLE_ASSUMPTION,
        ComparisonTier.FULL_IDENTITY_REFERENCE,
        "complete common user IDs and per-silo record counts",
        "all records retained with record-count-proportional per-user silo weights",
        "Common-trainer adaptation of the PVLDB 2024 enhanced weighting semantics.",
    ),
    MethodId.GROUP_PRIVACY_2: MethodSpec(
        MethodId.GROUP_PRIVACY_2.value,
        "Group-2 Person-Corrected",
        ImplementationProvenance.ADAPTED,
        PrivacySemantics.USER_CERTIFIED,
        ComparisonTier.SECONDARY,
        "complete IDs and a public two-record cap",
        "records beyond the stable cap are discarded",
    ),
    MethodId.RECORD_LEVEL_DP: MethodSpec(
        MethodId.RECORD_LEVEL_DP.value,
        "Record-level DP-SGD Reference",
        ImplementationProvenance.ADAPTED,
        PrivacySemantics.RECORD_DP,
        ComparisonTier.SECONDARY,
        "no cross-silo identity",
        "every selected record is retained and clipped independently",
        "This is record-level DP and is not a user-level privacy baseline.",
    ),
    MethodId.NON_PRIVATE: MethodSpec(
        MethodId.NON_PRIVATE.value,
        "Non-private Reference",
        ImplementationProvenance.ORACLE,
        PrivacySemantics.NON_PRIVATE,
        ComparisonTier.DIAGNOSTIC,
        "oracle evaluation only",
        "all records retained",
    ),
    MethodId.NAIVE_CLAIMED_C: MethodSpec(
        MethodId.NAIVE_CLAIMED_C.value,
        "Naive-Claimed-C",
        ImplementationProvenance.THIS_WORK,
        PrivacySemantics.UNSAFE_DIAGNOSTIC,
        ComparisonTier.DIAGNOSTIC,
        "positive partial-handle claims",
        "retained but true sensitivity is 2C while noise is claimed at C",
    ),
}
