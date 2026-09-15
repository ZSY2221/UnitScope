from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .compiler import compile_plan, validate_plan_with_oracle, with_oracle_validation
from .evidence import EvidencePolicy, generate_evidence_view
from .model import EvidenceView, MethodId, PartialHandleClaim, PublicConfig, UnitKind
from .synthetic import make_vector_population


def compile_checked(records, truth, evidence, registry, config):
    plan = compile_plan(MethodId.UNITSCOPE_TUNED, records, evidence, registry, config)
    validation = validate_plan_with_oracle(plan, records, truth, config)
    return with_oracle_validation(plan, validation)


def run_seed(seed: int) -> list[dict[str, object]]:
    config = PublicConfig(
        task_domain="failure-injection-v1",
        epoch="frozen-v1",
        clip_norm=1.0,
        max_handle_groups=2,
        max_fallback_slots=6,
        lambda_=0.75,
        public_normalizer=256.0,
        noise_multiplier=1.0,
        rounds=50,
        delta=1e-5,
        n_silos=6,
    )
    population = make_vector_population(
        n_users=256,
        n_silos=6,
        dimension=32,
        overlap_probability=0.7,
        gradient_correlation=0.5,
        norm=0.6,
        seed=seed,
        records_per_silo=2,
    )
    generated = generate_evidence_view(
        population.records,
        population.truth,
        config,
        EvidencePolicy(0.75, 0.6, handles_per_user=2),
        seed + 100,
    )
    evidence = generated.evidence
    registry = generated.registry
    base = compile_checked(
        population.records, population.truth, evidence, registry, config
    )
    if not base.validation.valid or not evidence.partial_handles:
        raise RuntimeError("base failure-injection plan is not valid")
    first = evidence.partial_handles[0]
    second = evidence.partial_handles[1] if len(evidence.partial_handles) > 1 else first

    invalidated = EvidenceView(
        evidence.task_domain,
        evidence.epoch,
        tuple(
            replace(claim, valid=False) if claim.claim_id == first.claim_id else claim
            for claim in evidence.partial_handles
        ),
        evidence.complete_claims,
    )
    all_invalidated = EvidenceView(
        evidence.task_domain,
        evidence.epoch,
        tuple(replace(claim, valid=False) for claim in evidence.partial_handles),
        evidence.complete_claims,
    )
    unregistered = EvidenceView(
        evidence.task_domain,
        evidence.epoch,
        evidence.partial_handles
        + (
            PartialHandleClaim(
                "unregistered-claim", first.issuer, 0, first.record_ids, True
            ),
        ),
        evidence.complete_claims,
    )
    quota = EvidenceView(
        evidence.task_domain,
        evidence.epoch,
        tuple(
            replace(claim, quota_slot=config.H)
            if claim.claim_id == first.claim_id
            else claim
            for claim in evidence.partial_handles
        ),
        evidence.complete_claims,
    )
    outside = EvidenceView(
        evidence.task_domain,
        evidence.epoch,
        tuple(
            replace(claim, record_ids=claim.record_ids + ("outside-record",))
            if claim.claim_id == first.claim_id
            else claim
            for claim in evidence.partial_handles
        ),
        evidence.complete_claims,
    )
    overlap_record = first.record_ids[0]
    overlap = EvidenceView(
        evidence.task_domain,
        evidence.epoch,
        tuple(
            replace(
                claim,
                record_ids=tuple(dict.fromkeys(claim.record_ids + (overlap_record,))),
            )
            if claim.claim_id == second.claim_id
            else claim
            for claim in evidence.partial_handles
        ),
        evidence.complete_claims,
    )
    scenarios = [
        ("valid-control", evidence, True, "accept"),
        ("single-invalid-handle", invalidated, True, "fallback"),
        ("institution-nonresponse", all_invalidated, True, "fallback"),
        (
            "cross-domain-replay",
            replace(evidence, task_domain="other-domain"),
            False,
            "reject",
        ),
        ("expired-epoch", replace(evidence, epoch="expired"), False, "reject"),
        ("unregistered-claim", unregistered, False, "reject"),
        ("handle-quota-exceeded", quota, False, "reject"),
        ("record-count-mismatch", outside, False, "reject"),
        ("overlapping-claims", overlap, False, "reject"),
    ]
    rows: list[dict[str, object]] = []
    for name, candidate, expected_valid, expected_route in scenarios:
        plan = compile_checked(
            population.records, population.truth, candidate, registry, config
        )
        partial = sum(
            unit.kind == UnitKind.PARTIAL_HANDLE and unit.radius > 0
            for unit in plan.units
        )
        local = sum(
            unit.kind in (UnitKind.LOCAL_ATOM, UnitKind.LOCAL_MIXED) and unit.radius > 0
            for unit in plan.units
        )
        actual_route = (
            "reject"
            if not plan.validation.valid
            else (
                "fallback"
                if partial < sum(c.valid for c in evidence.partial_handles)
                else "accept"
            )
        )
        pass_status = plan.validation.valid == expected_valid
        if expected_route == "fallback":
            pass_status = (
                pass_status and local > 0 and plan.noise_sensitivity == config.C
            )
        if expected_route == "reject":
            pass_status = pass_status and actual_route == "reject"
        rows.append(
            {
                "seed": seed,
                "scenario": name,
                "expected_valid": expected_valid,
                "actual_valid": plan.validation.valid,
                "expected_route": expected_route,
                "actual_route": actual_route,
                "partial_units": partial,
                "local_units": local,
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
    frame.to_csv(args.output / "failure_injection_results.csv", index=False)
    summary = frame.groupby("scenario", as_index=False).agg(
        seeds=("seed", "nunique"),
        pass_rate=("pass", "mean"),
        valid_rate=("actual_valid", "mean"),
    )
    summary.to_csv(args.output / "failure_injection_summary.csv", index=False)
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
