from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from experiments.unitscope.accounting import conservative_gaussian_rdp
from experiments.unitscope.audit import audit_plan_transition, full_recompile_audit
from experiments.unitscope.compiler import (
    compile_plan,
    validate_plan_with_oracle,
    validate_record_level_plan,
    with_oracle_validation,
)
from experiments.unitscope.evidence import (
    EvidencePolicy,
    generate_evidence_view,
    invalidate_claim,
)
from experiments.unitscope.executor import (
    assert_partition_invariant,
    execute_release,
    reduce_by_plan,
)
from experiments.unitscope.model import (
    CertificateType,
    EvidenceView,
    GroundTruth,
    LocalKind,
    MethodId,
    PartialHandleClaim,
    PermanentRegistry,
    PublicConfig,
    PublicRecord,
    UnitKind,
)
from experiments.unitscope.synthetic import make_vector_population


def config(lambda_: float = 0.4, *, H: int = 1, B: int = 3) -> PublicConfig:
    return PublicConfig(
        task_domain="test-domain",
        epoch="epoch-1",
        clip_norm=1.0,
        max_handle_groups=H,
        max_fallback_slots=B,
        lambda_=lambda_,
        public_normalizer=4.0,
        noise_multiplier=1.25,
        rounds=10,
        delta=1e-6,
        n_silos=B,
    )


def fixture(lambda_: float = 0.4):
    cfg = config(lambda_)
    population = make_vector_population(
        n_users=4,
        n_silos=3,
        dimension=5,
        overlap_probability=1.0,
        gradient_correlation=0.5,
        norm=0.7,
        seed=11,
    )
    generated = generate_evidence_view(
        population.records,
        population.truth,
        cfg,
        EvidencePolicy(1.0, 0.5, handles_per_user=1),
        seed=22,
    )
    return cfg, population, generated


def validated(method: MethodId, cfg, population, generated):
    plan = compile_plan(
        method, population.records, generated.evidence, generated.registry, cfg
    )
    check = validate_plan_with_oracle(plan, population.records, population.truth, cfg)
    assert check.valid, check.errors
    return with_oracle_validation(plan, check)


def test_lambda_zero_is_fallback_vector_for_vector() -> None:
    cfg, population, generated = fixture(0.0)
    unit_scope = validated(MethodId.UNITSCOPE_TUNED, cfg, population, generated)
    fallback = validated(MethodId.STABLE_LOCAL_FALLBACK, cfg, population, generated)
    assert unit_scope.units == fallback.units
    z = np.zeros(population.vectors.shape[1])
    left = execute_release(
        population.records, population.vectors, unit_scope, cfg, z
    ).deterministic_value
    right = execute_release(
        population.records, population.vectors, fallback, cfg, z
    ).deterministic_value
    np.testing.assert_array_equal(left, right)


def test_lambda_one_is_handle_only_vector_for_vector() -> None:
    cfg, population, generated = fixture(1.0)
    unit_scope = validated(MethodId.UNITSCOPE_TUNED, cfg, population, generated)
    handle_only = validated(MethodId.HANDLE_ONLY, cfg, population, generated)
    assert unit_scope.units == handle_only.units
    z = np.zeros(population.vectors.shape[1])
    left = execute_release(
        population.records, population.vectors, unit_scope, cfg, z
    ).deterministic_value
    right = execute_release(
        population.records, population.vectors, handle_only, cfg, z
    ).deterministic_value
    np.testing.assert_array_equal(left, right)


def test_equal_lambda_matches_equal_rule() -> None:
    cfg, population, generated = fixture(0.999)
    equal_plan = validated(MethodId.UNITSCOPE_EQUAL, cfg, population, generated)
    tuned_cfg = replace(cfg, lambda_=cfg.equal_lambda)
    tuned_plan = compile_plan(
        MethodId.UNITSCOPE_TUNED,
        population.records,
        generated.evidence,
        generated.registry,
        tuned_cfg,
    )
    check = validate_plan_with_oracle(
        tuned_plan, population.records, population.truth, tuned_cfg
    )
    assert check.valid
    assert [(unit.kind, unit.record_ids, unit.radius) for unit in equal_plan.units] == [
        (unit.kind, unit.record_ids, unit.radius) for unit in tuned_plan.units
    ]


def test_partition_only_changes_clipping_not_unclipped_query() -> None:
    cfg, population, generated = fixture(0.4)
    plans = [
        validated(method, cfg, population, generated)
        for method in (
            MethodId.STABLE_LOCAL_FALLBACK,
            MethodId.HANDLE_ONLY,
            MethodId.UNITSCOPE_TUNED,
            MethodId.NAIVE_RECALIBRATED_2C,
        )
    ]
    assert_partition_invariant(population.records, population.vectors, plans)


def test_group_sum_is_not_group_mean() -> None:
    records = (
        PublicRecord("r1", "s1", "slot", query_weight=1.0),
        PublicRecord("r2", "s1", "slot", query_weight=1.0),
    )
    truth = GroundTruth((("r1", "u1"), ("r2", "u1")))
    cfg = config(0.0)
    evidence = EvidenceView(cfg.task_domain, cfg.epoch)
    registry = PermanentRegistry(cfg.task_domain, cfg.epoch, ("r1", "r2"), ("slot",))
    plan = compile_plan(
        MethodId.STABLE_LOCAL_FALLBACK, records, evidence, registry, cfg
    )
    plan = with_oracle_validation(
        plan, validate_plan_with_oracle(plan, records, truth, cfg)
    )
    vectors = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    reduced, _ = reduce_by_plan(records, vectors, plan)
    np.testing.assert_array_equal(reduced[0], [4.0, 6.0])
    assert not np.array_equal(reduced[0], vectors.mean(axis=0))


def test_dropped_residual_units_are_not_counted_as_clipped() -> None:
    cfg, population, generated = fixture(1.0)
    handle_only = validated(MethodId.HANDLE_ONLY, cfg, population, generated)
    assert any(unit.kind == UnitKind.DROPPED for unit in handle_only.units)
    zero_vectors = np.zeros_like(population.vectors)
    release = execute_release(
        population.records,
        zero_vectors,
        handle_only,
        cfg,
        np.zeros(population.vectors.shape[1]),
    )
    assert release.diagnostics.dropped_records > 0
    assert release.diagnostics.clipped_units == 0
    assert release.diagnostics.clip_fraction == 0.0


def test_naive_claimed_c_has_2c_bound_but_no_valid_accounting() -> None:
    cfg, population, generated = fixture(0.4)
    plan = validated(MethodId.NAIVE_CLAIMED_C, cfg, population, generated)
    assert plan.structural_sensitivity_bound == pytest.approx(2 * cfg.C)
    assert plan.noise_sensitivity == pytest.approx(cfg.C)
    assert plan.certified_sensitivity is None
    assert plan.certificate_type == CertificateType.UNSAFE_DIAGNOSTIC
    with pytest.raises(ValueError, match="unsafe diagnostic"):
        execute_release(population.records, population.vectors, plan, cfg, np.zeros(5))


def test_naive_recalibrated_uses_twice_the_noise_sensitivity() -> None:
    cfg, population, generated = fixture(0.4)
    plan = validated(MethodId.NAIVE_RECALIBRATED_2C, cfg, population, generated)
    assert plan.certified_sensitivity is not None
    assert plan.certified_sensitivity.value == pytest.approx(2 * cfg.C)
    assert plan.noise_sensitivity == pytest.approx(2 * cfg.C)


def _full_identity_view(cfg, population):
    return generate_evidence_view(
        population.records,
        population.truth,
        cfg,
        EvidencePolicy(0.0, 0.0, complete_user_probability=1.0),
        seed=77,
    )


def test_uldp_oracle_weights_sum_to_one_user_capacity() -> None:
    cfg, population, _ = fixture(0.4)
    full = _full_identity_view(cfg, population)
    plan = compile_plan(
        MethodId.ULDP_AVG_WEIGHTED,
        population.records,
        full.evidence,
        full.registry,
        cfg,
    )
    check = validate_plan_with_oracle(plan, population.records, population.truth, cfg)
    assert check.valid, check.errors
    truth = population.truth.as_dict()
    capacity = {}
    for unit in plan.units:
        users = {truth[record_id] for record_id in unit.record_ids}
        assert len(users) == 1
        user = next(iter(users))
        capacity[user] = capacity.get(user, 0.0) + unit.radius
    assert set(capacity) == set(truth.values())
    assert all(value == pytest.approx(cfg.C) for value in capacity.values())
    assert plan.noise_sensitivity == pytest.approx(cfg.C)


def test_group_privacy_baseline_caps_records_and_calibrates_person_noise(
) -> None:
    cfg, population, _ = fixture(0.4)
    full = _full_identity_view(cfg, population)
    k = 2
    plan = compile_plan(
        MethodId.GROUP_PRIVACY_2,
        population.records,
        full.evidence,
        full.registry,
        cfg,
    )
    check = validate_plan_with_oracle(plan, population.records, population.truth, cfg)
    assert check.valid, check.errors
    truth = population.truth.as_dict()
    selected = {}
    for unit in plan.units:
        if unit.kind != UnitKind.GROUP_RECORD:
            continue
        user = truth[unit.record_ids[0]]
        selected[user] = selected.get(user, 0) + 1
    assert all(count <= k for count in selected.values())
    assert plan.certified_sensitivity is not None
    assert plan.certified_sensitivity.value == pytest.approx(k * cfg.C)
    assert plan.noise_sensitivity == pytest.approx(k * cfg.C)


def test_record_level_reference_has_singleton_units_and_record_certificate() -> None:
    cfg, population, generated = fixture(0.4)
    plan = compile_plan(
        MethodId.RECORD_LEVEL_DP,
        population.records,
        generated.evidence,
        generated.registry,
        cfg,
    )
    check = validate_record_level_plan(plan, population.records, cfg)
    assert check.valid, check.errors
    assert len(plan.units) == len(population.records)
    assert all(unit.kind == UnitKind.RECORD for unit in plan.units)
    assert all(len(unit.record_ids) == 1 for unit in plan.units)
    assert all(unit.radius == pytest.approx(cfg.C) for unit in plan.units)
    assert plan.certificate_type == CertificateType.RECORD_C
    assert plan.certified_sensitivity is not None
    assert plan.certified_sensitivity.value == pytest.approx(cfg.C)
    assert plan.noise_sensitivity == pytest.approx(cfg.C)


def test_record_level_reference_is_not_mislabeled_as_user_level() -> None:
    cfg, population, generated = fixture(0.4)
    plan = compile_plan(
        MethodId.RECORD_LEVEL_DP,
        population.records,
        generated.evidence,
        generated.registry,
        cfg,
    )
    user_check = validate_plan_with_oracle(
        plan, population.records, population.truth, cfg
    )
    assert not user_check.valid
    record_check = validate_record_level_plan(plan, population.records, cfg)
    assert record_check.valid


def test_invalid_handle_fails_closed_without_budget_reissue() -> None:
    cfg, population, generated = fixture(0.4)
    claim = generated.evidence.partial_handles[0]
    invalid = invalidate_claim(generated.evidence, claim.claim_id)
    plan = compile_plan(
        MethodId.UNITSCOPE_TUNED, population.records, invalid, generated.registry, cfg
    )
    invalid_records = set(claim.record_ids)
    assignment = plan.unit_for_record()
    assert all(
        assignment[record_id].kind in (UnitKind.LOCAL_ATOM, UnitKind.LOCAL_MIXED)
        for record_id in invalid_records
    )
    expected_radius = (1 - cfg.lambda_) * cfg.C / cfg.B
    assert all(
        assignment[record_id].radius == pytest.approx(expected_radius)
        for record_id in invalid_records
    )


def test_claim_cannot_reference_outside_frozen_universe() -> None:
    cfg = config(0.4, H=1, B=1)
    records = (PublicRecord("r1", "s1", "slot"),)
    evidence = EvidenceView(
        cfg.task_domain,
        cfg.epoch,
        (PartialHandleClaim("claim", "issuer", 0, ("r1", "outside"), True),),
        (),
    )
    registry = PermanentRegistry(
        cfg.task_domain, cfg.epoch, ("r1",), ("slot",), ("claim",), ()
    )
    plan = compile_plan(MethodId.UNITSCOPE_TUNED, records, evidence, registry, cfg)
    assert not plan.validation.valid
    assert any(
        "outside the frozen universe" in error for error in plan.validation.errors
    )


def test_deletion_has_no_refill_and_transition_is_bounded() -> None:
    cfg, population, generated = fixture(0.4)
    old_plan = validated(MethodId.UNITSCOPE_TUNED, cfg, population, generated)
    truth = population.truth.as_dict()
    removed_user = next(iter(sorted(set(truth.values()))))
    keep_indices = [
        i
        for i, record in enumerate(population.records)
        if truth[record.record_id] != removed_user
    ]
    new_records = tuple(population.records[i] for i in keep_indices)
    new_truth = GroundTruth(
        tuple((record.record_id, truth[record.record_id]) for record in new_records)
    )
    new_plan = compile_plan(
        MethodId.UNITSCOPE_TUNED,
        new_records,
        generated.evidence,
        generated.registry,
        cfg,
    )
    check = validate_plan_with_oracle(new_plan, new_records, new_truth, cfg)
    assert check.valid, check.errors
    observed = audit_plan_transition(old_plan, new_plan)
    assert observed.value <= cfg.C + 1e-12
    for label in set(unit.label for unit in old_plan.units) & set(
        unit.label for unit in new_plan.units
    ):
        old = next(unit for unit in old_plan.units if unit.label == label)
        new = next(unit for unit in new_plan.units if unit.label == label)
        assert set(new.record_ids).issubset(old.record_ids)


@pytest.mark.parametrize(
    "method",
    [
        MethodId.STABLE_LOCAL_FALLBACK,
        MethodId.UNITSCOPE_TUNED,
        MethodId.NAIVE_RECALIBRATED_2C,
    ],
)
def test_actual_adjacent_query_difference_is_bounded_by_observed_wur(
    method: MethodId,
) -> None:
    cfg, population, generated = fixture(0.4)
    truth = population.truth.as_dict()
    removed_user = sorted(set(truth.values()))[0]
    keep = np.asarray(
        [truth[record.record_id] != removed_user for record in population.records],
        dtype=bool,
    )
    new_records = tuple(
        record for record, retained in zip(population.records, keep) if retained
    )
    report = full_recompile_audit(
        method,
        population.records,
        new_records,
        generated.evidence,
        generated.registry,
        cfg,
        population.vectors,
        population.vectors[keep],
    )
    assert report.actual_query_difference is not None
    assert report.actual_query_difference <= report.observed.value + 1e-10


def test_observed_wur_cannot_enter_accountant() -> None:
    cfg, population, generated = fixture(0.4)
    plan = validated(MethodId.UNITSCOPE_TUNED, cfg, population, generated)
    observed = audit_plan_transition(plan, plan)
    with pytest.raises(TypeError, match="CertifiedSensitivity"):
        conservative_gaussian_rdp(observed, 1.0, 10, 1e-6)  # type: ignore[arg-type]


def test_noise_and_public_normalizer_do_not_depend_on_active_groups() -> None:
    cfg, population, generated = fixture(0.4)
    plan = validated(MethodId.UNITSCOPE_TUNED, cfg, population, generated)
    z = np.ones(population.vectors.shape[1])
    release = execute_release(population.records, population.vectors, plan, cfg, z)
    expected = np.full_like(z, cfg.noise_multiplier * cfg.C / cfg.public_normalizer)
    np.testing.assert_allclose(release.noise, expected)


def test_same_seed_reproduces_evidence_and_vectors_bitwise() -> None:
    left_cfg, left_population, left_generated = fixture(0.4)
    right_cfg, right_population, right_generated = fixture(0.4)
    assert left_cfg == right_cfg
    assert left_population.records == right_population.records
    assert left_generated == right_generated
    np.testing.assert_array_equal(left_population.vectors, right_population.vectors)


@pytest.mark.parametrize("requested", [-0.8, 0.0, 0.8])
def test_vector_population_realizes_signed_cross_silo_cosine(requested: float) -> None:
    population = make_vector_population(
        n_users=32,
        n_silos=6,
        dimension=16,
        overlap_probability=1.0,
        gradient_correlation=requested,
        norm=0.7,
        seed=101,
    )
    assert population.realized_pairwise_cosine_mean == pytest.approx(
        requested, abs=1e-12
    )
    assert population.realized_pairwise_cosine_p10 == pytest.approx(
        requested, abs=1e-12
    )
    assert population.realized_pairwise_cosine_p90 == pytest.approx(
        requested, abs=1e-12
    )


def test_vector_population_can_realize_four_handle_fragments_without_extra_silos() -> (
    None
):
    cfg = config(0.4, H=4, B=3)
    population = make_vector_population(
        n_users=4,
        n_silos=3,
        dimension=8,
        overlap_probability=1.0,
        gradient_correlation=0.0,
        norm=0.7,
        seed=202,
        records_per_silo=2,
    )
    generated = generate_evidence_view(
        population.records,
        population.truth,
        cfg,
        EvidencePolicy(1.0, 1.0, handles_per_user=4),
        seed=203,
    )
    truth = population.truth.as_dict()
    counts = {}
    for claim in generated.evidence.partial_handles:
        user = truth[claim.record_ids[0]]
        counts[user] = counts.get(user, 0) + 1
    assert set(counts.values()) == {4}


def test_mixed_unit_transition_pays_both_sides() -> None:
    cfg = config(0.0, B=1)
    old_records = (
        PublicRecord("u-record", "s1", "mixed-slot", LocalKind.MIXED),
        PublicRecord("v-record", "s1", "mixed-slot", LocalKind.MIXED),
    )
    new_records = (old_records[1],)
    evidence = EvidenceView(cfg.task_domain, cfg.epoch)
    registry = PermanentRegistry(
        cfg.task_domain, cfg.epoch, ("u-record", "v-record"), ("mixed-slot",)
    )
    old = compile_plan(
        MethodId.STABLE_LOCAL_FALLBACK, old_records, evidence, registry, cfg
    )
    new = compile_plan(
        MethodId.STABLE_LOCAL_FALLBACK, new_records, evidence, registry, cfg
    )
    observed = audit_plan_transition(old, new)
    assert old.units[0].radius == pytest.approx(cfg.C / 2)
    assert new.units[0].radius == pytest.approx(cfg.C / 2)
    assert observed.value == pytest.approx(cfg.C)


def test_full_identity_and_complete_only_endpoints() -> None:
    cfg, population, generated = fixture(0.4)
    complete = generate_evidence_view(
        population.records,
        population.truth,
        cfg,
        EvidencePolicy(0.0, 0.0, complete_user_probability=1.0),
        seed=33,
    )
    full = validated(MethodId.FULL_IDENTITY, cfg, population, complete)
    unit_scope = validated(MethodId.UNITSCOPE_TUNED, cfg, population, complete)
    assert full.units == unit_scope.units

    empty = generate_evidence_view(
        population.records, population.truth, cfg, EvidencePolicy(0.0, 0.0), seed=44
    )
    complete_only = validated(MethodId.COMPLETE_ONLY, cfg, population, empty)
    fallback = validated(MethodId.STABLE_LOCAL_FALLBACK, cfg, population, empty)
    assert complete_only.units == fallback.units
