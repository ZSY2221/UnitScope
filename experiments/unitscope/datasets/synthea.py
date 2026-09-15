from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd

from ..evidence import GeneratedEvidence
from ..model import GroundTruth, LocalKind, PublicRecord
from ..model import EvidenceView, PartialHandleClaim, PermanentRegistry, PublicConfig


ENCOUNTER_CLASSES = (
    "ambulatory",
    "emergency",
    "inpatient",
    "outpatient",
    "urgentcare",
    "wellness",
    "other",
)


@dataclass(frozen=True)
class SyntheaTask:
    records: Tuple[PublicRecord, ...]
    truth: GroundTruth
    x: np.ndarray
    y: np.ndarray
    patient_ids: np.ndarray
    encounter_dates: np.ndarray
    organization_ids: np.ndarray
    feature_names: Tuple[str, ...]
    index_date: str
    horizon_days: int


def _opaque(seed: int, *parts: object) -> str:
    value = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _read_required(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} misses required columns {missing}")
    return frame


def build_synthea_future_inpatient_task(
    csv_directory: Path,
    *,
    index_date: str,
    lookback_days: int = 365,
    horizon_days: int = 365,
    n_silos: int = 6,
    seed: int = 20260714,
) -> SyntheaTask:
    """Build the synthetic-patient future-inpatient task."""

    if lookback_days <= 0 or horizon_days <= 0 or n_silos <= 0:
        raise ValueError("lookback, horizon, and n_silos must be positive")
    directory = Path(csv_directory)
    patients = _read_required(directory / "patients.csv", ("Id", "BIRTHDATE", "GENDER"))
    encounters = _read_required(
        directory / "encounters.csv",
        ("Id", "START", "PATIENT", "ORGANIZATION", "ENCOUNTERCLASS"),
    )
    cutoff = pd.Timestamp(index_date, tz="UTC")
    window_start = cutoff - pd.to_timedelta(float(lookback_days), unit="D")
    horizon_end = cutoff + pd.to_timedelta(float(horizon_days), unit="D")
    encounters = encounters.copy()
    encounters["START_TS"] = pd.to_datetime(
        encounters["START"], utc=True, errors="coerce"
    )
    encounters = encounters.dropna(subset=["START_TS", "PATIENT", "ORGANIZATION"])
    future = encounters[
        (encounters["START_TS"] >= cutoff) & (encounters["START_TS"] < horizon_end)
    ]
    future_inpatient = set(
        future.loc[
            future["ENCOUNTERCLASS"].str.lower() == "inpatient", "PATIENT"
        ].astype(str)
    )
    history = encounters[
        (encounters["START_TS"] >= window_start) & (encounters["START_TS"] < cutoff)
    ].copy()
    if history.empty:
        raise ValueError("the frozen lookback window contains no encounters")

    birth_dates = pd.to_datetime(patients["BIRTHDATE"], utc=True, errors="coerce")
    birth_by_patient = dict(zip(patients["Id"].astype(str), birth_dates))
    gender_by_patient = dict(
        zip(patients["Id"].astype(str), patients["GENDER"].astype(str).str.upper())
    )
    feature_names = (
        "age_years",
        "gender_male",
        *(f"class_{name}" for name in ENCOUNTER_CLASSES),
    )
    records: List[PublicRecord] = []
    assignments: List[Tuple[str, str]] = []
    features: List[np.ndarray] = []
    labels: List[int] = []
    patient_ids: List[str] = []
    dates: List[str] = []
    organizations: List[str] = []
    seen_encounters = set()
    for row in history.itertuples(index=False):
        encounter_id = str(getattr(row, "Id"))
        patient_id = str(getattr(row, "PATIENT"))
        organization = str(getattr(row, "ORGANIZATION"))
        if encounter_id in seen_encounters:
            raise ValueError(f"duplicate encounter ID {encounter_id}")
        seen_encounters.add(encounter_id)
        birth = birth_by_patient.get(patient_id)
        if birth is None or pd.isna(birth):
            continue
        start = getattr(row, "START_TS")
        age = max(0.0, (start - birth).days / 365.25)
        encounter_class = str(getattr(row, "ENCOUNTERCLASS")).lower()
        if encounter_class not in ENCOUNTER_CLASSES[:-1]:
            encounter_class = "other"
        class_features = [float(encounter_class == name) for name in ENCOUNTER_CLASSES]
        vector = np.asarray(
            [
                age / 100.0,
                float(gender_by_patient.get(patient_id) == "M"),
                *class_features,
            ],
            dtype=np.float32,
        )
        silo = (
            int.from_bytes(
                hashlib.sha256(organization.encode("utf-8")).digest()[:8], "big"
            )
            % n_silos
        )
        record_id = f"synthea-{_opaque(seed, encounter_id)}"
        fallback_slot = f"local-{_opaque(seed + 1, patient_id, silo)}"
        records.append(
            PublicRecord(record_id, f"silo-{silo}", fallback_slot, LocalKind.ATOM, 1.0)
        )
        assignments.append((record_id, patient_id))
        features.append(vector)
        labels.append(int(patient_id in future_inpatient))
        patient_ids.append(patient_id)
        dates.append(start.isoformat())
        organizations.append(organization)
    if not records:
        raise ValueError("no eligible Synthea records were produced")
    return SyntheaTask(
        records=tuple(records),
        truth=GroundTruth(tuple(assignments)),
        x=np.stack(features),
        y=np.asarray(labels, dtype=np.int64),
        patient_ids=np.asarray(patient_ids),
        encounter_dates=np.asarray(dates),
        organization_ids=np.asarray(organizations),
        feature_names=tuple(feature_names),
        index_date=index_date,
        horizon_days=horizon_days,
    )


def generate_health_card_policy_evidence(
    task: SyntheaTask,
    active_record_ids: Sequence[str],
    config: PublicConfig,
    *,
    issuance_date: str,
    participating_silos: Sequence[str],
    handle_domains: int = 1,
    issuer: str = "regional-health-card-root",
) -> GeneratedEvidence:
    """Issue regional handles for eligible Synthea records."""

    if config.H < 1:
        raise ValueError("health-card policy requires H >= 1")
    if handle_domains < 1 or handle_domains > config.H:
        raise ValueError("health-card handle_domains must lie in [1, H]")
    active = set(active_record_ids)
    record_by_id = {
        record.record_id: record
        for record in task.records
        if record.record_id in active
    }
    truth_map = task.truth.as_dict()
    date_by_id = {
        record.record_id: pd.Timestamp(str(date))
        for record, date in zip(task.records, task.encounter_dates)
        if record.record_id in active
    }
    threshold = pd.Timestamp(issuance_date)
    threshold = (
        threshold.tz_localize("UTC")
        if threshold.tzinfo is None
        else threshold.tz_convert("UTC")
    )
    allowed_silos = set(participating_silos)
    ordered_silos = sorted(allowed_silos)
    silo_domain = {
        silo: index % handle_domains for index, silo in enumerate(ordered_silos)
    }
    records_by_user: MutableMapping[str, List[str]] = {}
    for record_id in active:
        records_by_user.setdefault(truth_map[record_id], []).append(record_id)
    claims = []
    covered_records = 0
    users_with_handle = 0
    total_handle_user_records = 0
    handle_user_coverages: List[float] = []
    residual_counts: List[int] = []
    for user_id, user_records in sorted(records_by_user.items()):
        eligible = sorted(
            record_id
            for record_id in user_records
            if record_by_id[record_id].silo_id in allowed_silos
            and date_by_id[record_id] >= threshold
        )
        if not eligible:
            continue
        users_with_handle += 1
        total_handle_user_records += len(user_records)
        covered_records += len(eligible)
        handle_user_coverages.append(len(eligible) / len(user_records))
        residual_counts.append(len(user_records) - len(eligible))
        records_by_domain: MutableMapping[int, List[str]] = {}
        for record_id in eligible:
            domain = silo_domain[record_by_id[record_id].silo_id]
            records_by_domain.setdefault(domain, []).append(record_id)
        for quota_slot, domain_records in sorted(records_by_domain.items()):
            claim_id = f"health-card-{_opaque(0, config.task_domain, config.epoch, user_id, quota_slot)}"
            claims.append(
                PartialHandleClaim(
                    claim_id,
                    f"{issuer}-domain-{quota_slot}",
                    quota_slot,
                    tuple(sorted(domain_records)),
                    True,
                )
            )
    evidence = EvidenceView(config.task_domain, config.epoch, tuple(claims), ())
    registry = PermanentRegistry(
        config.task_domain,
        config.epoch,
        tuple(sorted(active)),
        tuple(sorted({record.fallback_slot for record in record_by_id.values()})),
        tuple(claim.claim_id for claim in claims),
        (),
    )
    total_users = len(records_by_user)
    return GeneratedEvidence(
        evidence,
        registry,
        users_with_handle / total_users if total_users else 0.0,
        float(np.mean(handle_user_coverages)) if handle_user_coverages else 0.0,
        covered_records / total_handle_user_records
        if total_handle_user_records
        else 0.0,
        covered_records / len(active) if active else 0.0,
        0.0,
        sum(value > 0 for value in residual_counts) / len(residual_counts)
        if residual_counts
        else 0.0,
        sum(value == 0 for value in residual_counts) / len(residual_counts)
        if residual_counts
        else 0.0,
        float(np.mean(residual_counts)) if residual_counts else 0.0,
    )
