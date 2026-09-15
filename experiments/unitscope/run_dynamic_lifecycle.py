from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .audit import audit_plan_transition
from .compiler import compile_plan, validate_plan_with_oracle, with_oracle_validation
from .evidence import EvidencePolicy, generate_evidence_view, invalidate_claim
from .model import (
    GroundTruth,
    MethodId,
    PartialHandleClaim,
    PermanentRegistry,
    PublicConfig,
    PublicRecord,
    UnitKind,
)
from .synthetic import make_vector_population


def checked(records, truth, evidence, registry, config):
    plan = compile_plan(MethodId.UNITSCOPE_TUNED, records, evidence, registry, config)
    validation = validate_plan_with_oracle(plan, records, truth, config)
    return with_oracle_validation(plan, validation)


def run_seed(seed: int) -> list[dict[str, object]]:
    config = PublicConfig(
        "lifecycle-v1", "epoch-0", 1.0, 1, 6, 0.9, 512.0, 1.0, 50, 1e-5, 6
    )
    population = make_vector_population(
        n_users=512,
        n_silos=6,
        dimension=16,
        overlap_probability=0.6,
        gradient_correlation=0.5,
        norm=0.6,
        seed=seed,
        records_per_silo=1,
    )
    generated = generate_evidence_view(
        population.records,
        population.truth,
        config,
        EvidencePolicy(0.5, 0.5, handles_per_user=1),
        seed + 100,
    )
    records = tuple(population.records)
    truth = population.truth
    evidence = generated.evidence
    registry = generated.registry
    base = checked(records, truth, evidence, registry, config)
    if not base.validation.valid:
        raise RuntimeError("invalid lifecycle base plan")
    truth_map = truth.as_dict()
    first_claim = evidence.partial_handles[0]
    affected_user = truth_map[first_claim.record_ids[0]]
    affected_records = {
        record_id for record_id, user in truth.assignments if user == affected_user
    }
    deleted_records = tuple(
        record for record in records if record.record_id not in affected_records
    )
    deleted_truth = GroundTruth(
        tuple(item for item in truth.assignments if item[0] not in affected_records)
    )
    deleted_plan = checked(deleted_records, deleted_truth, evidence, registry, config)

    revoked_evidence = invalidate_claim(evidence, first_claim.claim_id)
    revoked_plan = checked(records, truth, revoked_evidence, registry, config)

    new_record = PublicRecord(
        record_id=f"future-{seed}",
        silo_id=records[0].silo_id,
        fallback_slot=f"future-slot-{seed}",
    )
    new_truth = GroundTruth(
        truth.assignments + ((new_record.record_id, affected_user),)
    )
    unregistered_plan = checked(
        records + (new_record,), new_truth, evidence, registry, config
    )
    pre_registry = PermanentRegistry(
        registry.task_domain,
        registry.epoch,
        registry.record_universe + (new_record.record_id,),
        registry.fallback_slots + (new_record.fallback_slot,),
        registry.handle_claim_ids,
        registry.complete_claim_ids,
    )
    pre_base = checked(records, truth, evidence, pre_registry, config)
    preallocated_plan = checked(
        records + (new_record,), new_truth, evidence, pre_registry, config
    )

    exiting_silo = sorted({record.silo_id for record in records})[0]
    exit_records = tuple(record for record in records if record.silo_id != exiting_silo)
    exit_ids = {record.record_id for record in exit_records}
    exit_truth = GroundTruth(
        tuple(item for item in truth.assignments if item[0] in exit_ids)
    )
    same_epoch_exit = checked(exit_records, exit_truth, evidence, registry, config)
    next_config = replace(config, epoch="epoch-1", n_silos=5)
    next_partial = tuple(
        replace(
            claim,
            record_ids=tuple(
                record_id for record_id in claim.record_ids if record_id in exit_ids
            ),
        )
        for claim in evidence.partial_handles
        if any(record_id in exit_ids for record_id in claim.record_ids)
    )
    next_complete = tuple(
        replace(
            claim,
            record_ids=tuple(
                record_id for record_id in claim.record_ids if record_id in exit_ids
            ),
        )
        for claim in evidence.complete_claims
        if any(record_id in exit_ids for record_id in claim.record_ids)
    )
    next_evidence = replace(
        evidence,
        epoch="epoch-1",
        partial_handles=next_partial,
        complete_claims=next_complete,
    )
    next_registry = PermanentRegistry(
        config.task_domain,
        "epoch-1",
        tuple(sorted(exit_ids)),
        tuple(sorted({record.fallback_slot for record in exit_records})),
        tuple(claim.claim_id for claim in next_partial),
        tuple(claim.claim_id for claim in next_complete),
    )
    next_epoch_exit = checked(
        exit_records, exit_truth, next_evidence, next_registry, next_config
    )

    reissued = replace(
        evidence,
        partial_handles=evidence.partial_handles
        + (
            PartialHandleClaim(
                f"reissue-{seed}", first_claim.issuer, 0, first_claim.record_ids, True
            ),
        ),
    )
    reissued_plan = checked(records, truth, reissued, registry, config)

    cases = [
        (
            "single-user-deletion",
            deleted_plan,
            audit_plan_transition(base, deleted_plan).value,
            True,
            "same-epoch-safe",
        ),
        (
            "credential-revocation",
            revoked_plan,
            audit_plan_transition(base, revoked_plan).value,
            True,
            "fallback-without-refill",
        ),
        ("unregistered-new-record", unregistered_plan, None, False, "reject"),
        (
            "preallocated-new-record",
            preallocated_plan,
            audit_plan_transition(pre_base, preallocated_plan).value,
            True,
            "preallocated-slot",
        ),
        (
            "institution-exit-same-epoch",
            same_epoch_exit,
            audit_plan_transition(base, same_epoch_exit).value,
            True,
            "requires-new-epoch",
        ),
        ("institution-exit-new-epoch", next_epoch_exit, None, True, "new-epoch-valid"),
        ("credential-reissue-no-refill", reissued_plan, None, False, "reject"),
    ]
    rows = []
    for scenario, plan, wur, expected_valid, decision in cases:
        pass_status = plan.validation.valid == expected_valid
        if scenario == "single-user-deletion" and wur is not None:
            pass_status = pass_status and wur <= config.C + 1e-9
        if scenario == "credential-revocation":
            fallback_records = sum(
                len(unit.record_ids)
                for unit in plan.units
                if unit.kind in (UnitKind.LOCAL_ATOM, UnitKind.LOCAL_MIXED)
            )
            pass_status = (
                pass_status
                and fallback_records > 0
                and plan.noise_sensitivity == config.C
            )
        rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "expected_valid": expected_valid,
                "actual_valid": plan.validation.valid,
                "decision": decision,
                "observed_wur": wur,
                "noise_sensitivity": plan.noise_sensitivity,
                "errors": " | ".join(plan.validation.errors),
                "pass": pass_status,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row for seed in args.seeds for row in run_seed(seed)])
    frame.to_csv(args.output / "dynamic_lifecycle_results.csv", index=False)
    summary = frame.groupby(["scenario", "decision"], as_index=False).agg(
        seeds=("seed", "nunique"),
        pass_rate=("pass", "mean"),
        valid_rate=("actual_valid", "mean"),
        wur_mean=("observed_wur", "mean"),
        wur_max=("observed_wur", "max"),
    )
    summary.to_csv(args.output / "dynamic_lifecycle_summary.csv", index=False)
    manifest = {
        "seeds": args.seeds,
        "rows": len(frame),
        "all_pass": bool(frame["pass"].all()),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest))
    if not manifest["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
