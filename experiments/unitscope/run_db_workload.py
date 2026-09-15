from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .compiler import compile_plan, validate_plan_with_oracle, with_oracle_validation
from .datasets import (
    build_synthea_future_inpatient_task,
    generate_health_card_policy_evidence,
)
from .manifest import atomic_write_json, environment_fingerprint
from .model import GroundTruth, MethodId, PublicConfig, PublicRecord, UnitKind


DECLARATION_PAYLOAD_SHA256 = (
    "56664307c8c0208c18c8fd1971f5f80640a15f7a0a86ead019b6c77e2471e9c2"
)
DECLARATION_FILE_SHA256 = (
    "efdf9b5d5ed3e5805aa808d26208c0ad6ef3af14f640861e19cda9301d7c248f"
)
SOURCE_FINGERPRINT = "559757dc849f4361a328f456d2c0a20c6df72419068321c753c6be787161e937"
EPSILONS = (1.0, 4.0, 16.0)
SEEDS = tuple(range(42001, 42011))
ROUTES = ("stable-local", "naive-2c", "unitscope")
BOOTSTRAP_SEED = 20270724
BOOTSTRAP_DRAWS = 20_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_sha256(path: Path) -> str:
    payload = path.read_bytes()
    marker = b"Declaration payload SHA-256:"
    return hashlib.sha256(payload[: payload.index(marker)]).hexdigest()


def _subset_truth(truth: GroundTruth, record_ids: set[str]) -> GroundTruth:
    return GroundTruth(
        tuple(
            (record_id, person)
            for record_id, person in truth.assignments
            if record_id in record_ids
        )
    )


def distinct_person_count(
    record_ids: Sequence[str], truth: GroundTruth, positive_people: set[str]
) -> int:
    truth_map = truth.as_dict()
    return len(
        {
            truth_map[record_id]
            for record_id in record_ids
            if truth_map[record_id] in positive_people
        }
    )


def stable_local_count(
    records: Sequence[PublicRecord], truth: GroundTruth, positive_people: set[str]
) -> int:
    truth_map = truth.as_dict()
    return len(
        {
            record.fallback_slot
            for record in records
            if truth_map[record.record_id] in positive_people
        }
    )


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap(values: np.ndarray, draws: np.ndarray) -> tuple[float, float, float]:
    means = values[draws].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def run(data: Path, output: Path, declaration: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            "database workload output directory must be empty before its run-start record"
        )
    observed = {
        "declaration_payload_sha256": _payload_sha256(declaration),
        "declaration_file_sha256": _sha256(declaration),
    }
    expected = {
        "declaration_payload_sha256": DECLARATION_PAYLOAD_SHA256,
        "declaration_file_sha256": DECLARATION_FILE_SHA256,
    }
    if observed != expected:
        raise RuntimeError(
            f"declaration integrity failed: observed={observed}, expected={expected}"
        )
    atomic_write_json(
        output / "run_start.json",
        {
            "workload": "synthea-person-level-cohort-count-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "declaration": observed,
            "source_fingerprint": SOURCE_FINGERPRINT,
            "epsilons": EPSILONS,
            "seeds": SEEDS,
            "expected_releases": 90,
            "environment": environment_fingerprint(),
        },
    )

    task = build_synthea_future_inpatient_task(
        data,
        index_date="2020-01-01",
        lookback_days=365,
        horizon_days=365,
        n_silos=6,
        seed=20260714,
    )
    config = PublicConfig(
        task_domain="synthea-db-workload-v1",
        epoch="db-workload-v1",
        clip_norm=1.0,
        max_handle_groups=1,
        max_fallback_slots=6,
        lambda_=1.0,
        public_normalizer=1.0,
        noise_multiplier=0.0,
        rounds=1,
        delta=1e-5,
        n_silos=6,
    )
    generated = generate_health_card_policy_evidence(
        task,
        [record.record_id for record in task.records],
        config,
        issuance_date="2019-07-01",
        participating_silos=("silo-0", "silo-1", "silo-2", "silo-3"),
        handle_domains=1,
    )
    truth_map = task.truth.as_dict()
    label_by_person: dict[str, int] = {}
    for record, label in zip(task.records, task.y):
        person = truth_map[record.record_id]
        previous = label_by_person.setdefault(person, int(label))
        if previous != int(label):
            raise RuntimeError(
                "the frozen future-inpatient label is not stable within patient"
            )
    positive_people = {
        person for person, label in label_by_person.items() if label == 1
    }

    valid_claims = [
        claim for claim in generated.evidence.partial_handles if claim.valid
    ]
    claimed_record_ids = [
        record_id for claim in valid_claims for record_id in claim.record_ids
    ]
    if len(claimed_record_ids) != len(set(claimed_record_ids)):
        raise RuntimeError("a query-universe record belongs to multiple valid claims")
    query_ids = set(claimed_record_ids)
    if not query_ids:
        raise RuntimeError("health-card query universe is empty")
    query_records = tuple(
        record for record in task.records if record.record_id in query_ids
    )
    query_truth = _subset_truth(task.truth, query_ids)
    claim_people: dict[str, str] = {}
    for claim in valid_claims:
        people = {truth_map[record_id] for record_id in claim.record_ids}
        if len(people) != 1:
            raise RuntimeError(f"impure health-card claim {claim.claim_id}")
        claim_people[claim.claim_id] = next(iter(people))
    if len(claim_people) != len(set(claim_people.values())):
        raise RuntimeError(
            "one person maps to more than one claim in the one-domain query universe"
        )

    oracle_count = distinct_person_count(tuple(query_ids), task.truth, positive_people)
    handle_count = sum(person in positive_people for person in claim_people.values())
    if handle_count != oracle_count:
        raise RuntimeError(
            "handle de-duplication does not equal oracle COUNT(DISTINCT person)"
        )
    local_count = stable_local_count(query_records, task.truth, positive_people)
    if local_count <= oracle_count:
        raise RuntimeError(
            "the declared Stable-Local positive duplicate bias was not realized"
        )

    unit_plan = compile_plan(
        MethodId.UNITSCOPE_SWEEP,
        query_records,
        generated.evidence,
        generated.registry,
        config,
    )
    unit_validation = validate_plan_with_oracle(
        unit_plan, query_records, query_truth, config
    )
    unit_plan = with_oracle_validation(unit_plan, unit_validation)
    naive_plan = compile_plan(
        MethodId.NAIVE_RECALIBRATED_2C,
        query_records,
        generated.evidence,
        generated.registry,
        config,
    )
    naive_validation = validate_plan_with_oracle(
        naive_plan, query_records, query_truth, config
    )
    naive_plan = with_oracle_validation(naive_plan, naive_validation)
    if not unit_validation.valid or not naive_validation.valid:
        raise RuntimeError(
            f"compiler validation failed: unit={unit_validation.errors}, naive={naive_validation.errors}"
        )
    if (
        unit_plan.certified_sensitivity is None
        or unit_plan.certified_sensitivity.value != 1.0
    ):
        raise RuntimeError("UnitScope did not certify C sensitivity")
    if (
        naive_plan.certified_sensitivity is None
        or naive_plan.certified_sensitivity.value != 2.0
    ):
        raise RuntimeError("Naive did not certify 2C sensitivity")
    unit_for_record = unit_plan.unit_for_record()
    person_to_unit_labels: dict[str, set[str]] = {}
    for record in query_records:
        unit = unit_for_record[record.record_id]
        if unit.kind != UnitKind.PARTIAL_HANDLE:
            raise RuntimeError(
                "a health-card query record did not compile to Partial-Handle"
            )
        person_to_unit_labels.setdefault(truth_map[record.record_id], set()).add(
            unit.label
        )
    if any(len(labels) != 1 for labels in person_to_unit_labels.values()):
        raise RuntimeError(
            "a claimed person maps to more than one UnitScope contribution unit"
        )

    silos_by_person: dict[str, set[str]] = {}
    for record in query_records:
        silos_by_person.setdefault(truth_map[record.record_id], set()).add(
            record.silo_id
        )
    fixture_person = next(
        (person for person, silos in silos_by_person.items() if len(silos) > 1), None
    )
    if fixture_person is None:
        raise RuntimeError(
            "no multi-silo person exists for the explicit de-duplication fixture"
        )
    fixture_ids = [
        record.record_id
        for record in query_records
        if truth_map[record.record_id] == fixture_person
    ]
    fixture_count = distinct_person_count(fixture_ids, task.truth, {fixture_person})
    if fixture_count != 1:
        raise RuntimeError("explicit multi-silo fixture was not counted once")

    route_deterministic = {
        "stable-local": local_count,
        "naive-2c": oracle_count,
        "unitscope": oracle_count,
    }
    route_sensitivity = {"stable-local": 1.0, "naive-2c": 2.0, "unitscope": 1.0}
    rows: list[dict] = []
    for seed in SEEDS:
        standard_laplace = float(np.random.default_rng(seed).laplace(0.0, 1.0))
        for epsilon in EPSILONS:
            for route in ROUTES:
                deterministic = route_deterministic[route]
                sensitivity = route_sensitivity[route]
                noise = sensitivity / epsilon * standard_laplace
                release = deterministic + noise
                error = release - oracle_count
                rows.append(
                    {
                        "seed": seed,
                        "epsilon": epsilon,
                        "route": route,
                        "reference_person_count": oracle_count,
                        "deterministic_count": deterministic,
                        "sensitivity": sensitivity,
                        "standard_laplace": standard_laplace,
                        "noise": noise,
                        "release": release,
                        "error": error,
                        "squared_error": error * error,
                        "absolute_relative_error": abs(error) / oracle_count,
                    }
                )
    if len(rows) != 90:
        raise RuntimeError(f"release completeness failed: {len(rows)} rows")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(SEEDS), size=(BOOTSTRAP_DRAWS, len(SEEDS)))
    summary: list[dict] = []
    for epsilon in EPSILONS:
        for route in ROUTES:
            selected = [
                row
                for row in rows
                if row["epsilon"] == epsilon and row["route"] == route
            ]
            squared = np.asarray([float(row["squared_error"]) for row in selected])
            relative = np.asarray(
                [float(row["absolute_relative_error"]) for row in selected]
            )
            mse, mse_low, mse_high = _bootstrap(squared, draws)
            rel, rel_low, rel_high = _bootstrap(relative, draws)
            summary.append(
                {
                    "epsilon": epsilon,
                    "route": route,
                    "releases": len(selected),
                    "mse": mse,
                    "mse_ci_lower": mse_low,
                    "mse_ci_upper": mse_high,
                    "absolute_relative_error": rel,
                    "absolute_relative_error_ci_lower": rel_low,
                    "absolute_relative_error_ci_upper": rel_high,
                }
            )
    for epsilon in EPSILONS:
        by_route = {row["route"]: row for row in summary if row["epsilon"] == epsilon}
        if not float(by_route["unitscope"]["mse"]) < float(by_route["naive-2c"]["mse"]):
            raise RuntimeError(
                f"predeclared UnitScope MSE expectation failed at epsilon={epsilon}"
            )

    audit = {
        "source_fingerprint": SOURCE_FINGERPRINT,
        "query_semantics": "COUNT(DISTINCT natural_person) with positive future-inpatient label",
        "query_universe_records": len(query_records),
        "query_universe_people": len(person_to_unit_labels),
        "positive_person_reference_count": oracle_count,
        "stable_local_deterministic_count": local_count,
        "stable_local_excess_count": local_count - oracle_count,
        "stable_local_duplicate_ratio": local_count / oracle_count,
        "stable_local_relative_bias": (local_count - oracle_count) / oracle_count,
        "valid_claims": len(valid_claims),
        "claim_purity_pass": True,
        "exactly_one_claim_per_query_person": True,
        "unit_plan_validation_pass": unit_validation.valid,
        "unit_plan_certificate": unit_plan.certified_sensitivity.value,
        "naive_plan_validation_pass": naive_validation.valid,
        "naive_plan_certificate": naive_plan.certified_sensitivity.value,
        "unit_deterministic_answer_matches_oracle": handle_count == oracle_count,
        "naive_deterministic_answer_matches_oracle": handle_count == oracle_count,
        "fixture_person_hash": hashlib.sha256(
            fixture_person.encode("utf-8")
        ).hexdigest()[:24],
        "fixture_silos": sorted(silos_by_person[fixture_person]),
        "fixture_records": len(fixture_ids),
        "fixture_distinct_count": fixture_count,
        "all_seeds_reported": sorted({row["seed"] for row in rows}) == list(SEEDS),
        "release_rows": len(rows),
        "zero_discarded": True,
    }

    analysis = output / "analysis"
    _write_csv(analysis / "releases.csv", rows, tuple(rows[0]))
    _write_csv(analysis / "summary.csv", summary, tuple(summary[0]))
    atomic_write_json(analysis / "query_audit.json", audit)
    output_files = sorted(path for path in analysis.iterdir() if path.is_file())
    hashes = {path.name: _sha256(path) for path in output_files}
    analysis_hash = hashlib.sha256(
        "".join(
            f"{name}\0{digest}\n" for name, digest in sorted(hashes.items())
        ).encode("utf-8")
    ).hexdigest()
    atomic_write_json(
        analysis / "analysis_manifest.json",
        {
            "analysis": "db-workload-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "release_rows": len(rows),
            "zero_discarded": True,
            "output_hashes": hashes,
            "analysis_hash": analysis_hash,
        },
    )
    print(json.dumps({"analysis_hash": analysis_hash, **audit}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pre-declared database-native UnitScope workload"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data, args.output, args.declaration)
