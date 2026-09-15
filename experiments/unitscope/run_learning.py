from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np

from .accounting import calibrate_noise_multiplier
from .baselines import PRIMARY_METHODS
from .datasets import (
    build_synthea_future_inpatient_task,
    generate_health_card_policy_evidence,
    load_and_partition_movielens1m,
    balance_movielens_partition,
    load_and_partition_sent140,
    load_tff_federated_emnist,
    partition_femnist_writers,
)
from .evidence import EvidencePolicy, generate_evidence_view
from .manifest import (
    ALLOWED_LAMBDA_SOURCES,
    atomic_write_json,
    environment_fingerprint,
    validate_result_row,
)
from .model import GroundTruth, MethodId, PublicConfig, PublicRecord
from .torch_training import LearningTask, TrainingConfig, TrainingSeeds, train_method


DEFAULT_METHODS = (
    MethodId.STABLE_LOCAL_FALLBACK,
    MethodId.HANDLE_ONLY,
    MethodId.UNITSCOPE_EQUAL,
    MethodId.UNITSCOPE_TUNED,
    MethodId.NAIVE_RECALIBRATED_2C,
    MethodId.FULL_IDENTITY,
    MethodId.NON_PRIVATE,
)

FROZEN_CONTROLLED_SLICES = {
    "controlled-favorable": {
        "H": 1,
        "B": 6,
        "q_user": 0.75,
        "q_record": 0.75,
        "overlap": 0.8,
    },
    "controlled-middle": {
        "H": 1,
        "B": 6,
        "q_user": 0.5,
        "q_record": 0.5,
        "overlap": 0.5,
    },
    "controlled-stress": {
        "H": 4,
        "B": 6,
        "q_user": 0.25,
        "q_record": 0.25,
        "overlap": 0.2,
    },
}


def _rank(seed: int, *parts: object) -> int:
    value = "|".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _subset_truth(truth: GroundTruth, records: Sequence[PublicRecord]) -> GroundTruth:
    truth_map = truth.as_dict()
    return GroundTruth(
        tuple((record.record_id, truth_map[record.record_id]) for record in records)
    )


def _user_disjoint_record_split(
    records: Sequence[PublicRecord], truth: GroundTruth, seed: int
) -> np.ndarray:
    truth_map = truth.as_dict()
    users = sorted(
        set(truth_map.values()), key=lambda user: _rank(seed, "user-split", user)
    )
    n_public_auxiliary = int(round(0.05 * len(users)))
    n_train = int(round(0.65 * len(users)))
    n_validation = int(round(0.1 * len(users)))
    split_by_user = {user: "public_auxiliary" for user in users[:n_public_auxiliary]}
    train_start = n_public_auxiliary
    validation_start = train_start + n_train
    test_start = validation_start + n_validation
    split_by_user.update(
        {user: "train" for user in users[train_start:validation_start]}
    )
    split_by_user.update(
        {user: "validation" for user in users[validation_start:test_start]}
    )
    split_by_user.update({user: "test" for user in users[test_start:]})
    return np.asarray(
        [split_by_user[truth_map[record.record_id]] for record in records]
    )


def prepare_task(args: argparse.Namespace):
    if args.dataset == "femnist":
        writers = load_tff_federated_emnist(
            args.data,
            only_digits=args.only_digits,
            maximum_writers=args.maximum_writers,
            maximum_examples_per_writer=args.maximum_examples_per_writer,
            writer_selection_seed=args.writer_selection_seed
            if args.writer_selection_seed is not None
            else args.split_seed,
            writer_selection_offset=args.writer_selection_offset,
        )
        partition = partition_femnist_writers(
            writers,
            n_silos=args.n_silos,
            train_overlap_probability=args.overlap,
            split_seed=args.split_seed,
            topology_seed=args.topology_seed,
            dirichlet_alpha=getattr(args, "dirichlet_alpha", None),
        )
        return (
            LearningTask(
                partition.records,
                partition.truth,
                partition.x,
                partition.y,
                partition.record_split,
                10 if args.only_digits else 62,
                "femnist",
            ),
            partition,
        )
    if args.dataset == "synthea":
        synthea = build_synthea_future_inpatient_task(
            args.data,
            index_date=args.index_date,
            lookback_days=args.lookback_days,
            horizon_days=args.horizon_days,
            n_silos=args.n_silos,
            seed=args.split_seed,
        )
        split = _user_disjoint_record_split(
            synthea.records, synthea.truth, args.split_seed
        )
        return (
            LearningTask(
                synthea.records,
                synthea.truth,
                synthea.x,
                synthea.y,
                split,
                2,
                "structured",
                stable_user_label=True,
            ),
            synthea,
        )
    if args.dataset == "sent140":
        partition = load_and_partition_sent140(
            args.data,
            n_silos=args.n_silos,
            train_overlap_probability=args.overlap,
            split_seed=args.split_seed,
            topology_seed=args.topology_seed,
            minimum_records_per_account=args.minimum_records_per_account,
            maximum_accounts=args.maximum_accounts,
            maximum_records_per_account=args.maximum_records_per_account,
            account_selection_seed=args.account_selection_seed,
            text_hash_dimension=args.text_hash_dimension,
        )
        return (
            LearningTask(
                partition.records,
                partition.truth,
                partition.x,
                partition.y,
                partition.record_split,
                2,
                "text",
            ),
            partition,
        )
    if args.dataset == "movielens1m":
        partition = load_and_partition_movielens1m(
            args.data,
            n_silos=args.n_silos,
            train_overlap_probability=args.overlap,
            split_seed=args.split_seed,
            topology_seed=args.topology_seed,
            maximum_accounts=args.maximum_accounts,
            maximum_records_per_account=args.maximum_records_per_account,
            account_selection_seed=args.account_selection_seed,
        )
        if args.movielens_task_variant == "balanced-v1":
            partition = balance_movielens_partition(
                partition,
                balance_seed=args.balance_seed,
            )
        return (
            LearningTask(
                partition.records,
                partition.truth,
                partition.x,
                partition.y,
                partition.record_split,
                2,
                "movielens-balanced"
                if args.movielens_task_variant == "balanced-v1"
                else "structured",
            ),
            partition,
        )
    raise ValueError(args.dataset)


def _training_view(task: LearningTask) -> tuple[Tuple[PublicRecord, ...], GroundTruth]:
    indices = np.flatnonzero(task.record_split == "train")
    records = tuple(task.records[index] for index in indices)
    return records, _subset_truth(task.truth, records)


def _realized_train_overlap(
    records: Sequence[PublicRecord], truth: GroundTruth
) -> float:
    truth_map = truth.as_dict()
    silos = {}
    for record in records:
        silos.setdefault(truth_map[record.record_id], set()).add(record.silo_id)
    return (
        sum(len(value) >= 2 for value in silos.values()) / len(silos) if silos else 0.0
    )


def _reported_lambda(method: MethodId, config: PublicConfig) -> float | None:
    if method == MethodId.STABLE_LOCAL_FALLBACK:
        return 0.0
    if method == MethodId.HANDLE_ONLY:
        return 1.0
    if method == MethodId.UNITSCOPE_EQUAL:
        return config.equal_lambda
    if method in (MethodId.UNITSCOPE_TUNED, MethodId.UNITSCOPE_SWEEP):
        return config.lambda_
    return None


def _evaluation_evidence_groups(records, truth, generated) -> Mapping[str, str]:
    """Label evaluation users by evidence coverage for subgroup reporting."""

    truth_map = truth.as_dict()
    records_by_user: dict[str, set[str]] = {}
    for record in records:
        records_by_user.setdefault(truth_map[record.record_id], set()).add(
            record.record_id
        )
    claimed = {
        record_id
        for claim in generated.evidence.partial_handles
        if claim.valid
        for record_id in claim.record_ids
    }
    output = {}
    for user, user_records in records_by_user.items():
        covered = len(user_records & claimed)
        if covered == 0:
            output[user] = "no-handle"
        elif covered == len(user_records):
            output[user] = "handle-complete-observed-window"
        else:
            output[user] = "handle-with-residual"
    return output


def _hash_payload(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _code_tree_hash() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _record_view_hash(
    records: Sequence[PublicRecord], split: Sequence[str] | None = None
) -> str:
    values = []
    for index, record in enumerate(records):
        values.append(
            (
                record.record_id,
                record.silo_id,
                record.fallback_slot,
                record.local_kind.value,
                record.query_weight,
                None if split is None else str(split[index]),
            )
        )
    return _hash_payload(sorted(values))


def _evidence_hash(generated) -> str:
    payload = {
        "partial": [
            (
                claim.claim_id,
                claim.issuer,
                claim.quota_slot,
                tuple(claim.record_ids),
                claim.valid,
            )
            for claim in generated.evidence.partial_handles
        ],
        "complete": [
            (
                claim.claim_id,
                claim.issuer,
                tuple(claim.record_ids),
                claim.complete_domain_attestation,
            )
            for claim in generated.evidence.complete_claims
        ],
        "registry": asdict(generated.registry),
    }
    return _hash_payload(payload)


def _load_completed(run_directory: Path) -> set[str]:
    if not run_directory.exists():
        return set()
    output = set()
    for path in run_directory.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("run_id") != path.stem:
            raise ValueError(f"run artifact {path} has a mismatched run_id")
        output.add(path.stem)
    return output


def _rewrite_jsonl(path: Path, run_directory: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(item.read_text(encoding="utf-8"))
        for item in sorted(run_directory.glob("*.json"))
    ]
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name, suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
                handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _load_lambda_selection(
    args: argparse.Namespace, methods: Sequence[MethodId]
) -> Mapping[str, object]:
    if args.protected_test_used_for_selection:
        raise ValueError("the protected test set cannot select lambda")
    if args.lambda_selection_source not in ALLOWED_LAMBDA_SOURCES:
        raise ValueError("lambda selection source is not admissible")
    if args.lambda_selection_record is not None:
        value = json.loads(args.lambda_selection_record.read_text(encoding="utf-8"))
    else:
        value = {
            "source": args.lambda_selection_source,
            "policy_id": args.lambda_policy_id,
            "selected_lambda": args.lambda_value,
            "candidates": sorted(
                set([0.0, 0.1, args.H / (args.H + args.B), 0.25, 0.5, 0.75, 0.9, 1.0])
            ),
            "scores": None,
        }
    if str(value.get("source")) != args.lambda_selection_source:
        raise ValueError("lambda selection record and CLI source disagree")
    if str(value.get("policy_id")) != args.lambda_policy_id:
        raise ValueError("lambda selection record and CLI policy ID disagree")
    if not np.isclose(float(value.get("selected_lambda")), args.lambda_value):
        raise ValueError("lambda selection record and --lambda-value disagree")
    candidates = [float(candidate) for candidate in value.get("candidates", [])]
    if not any(np.isclose(candidate, args.lambda_value) for candidate in candidates):
        raise ValueError("the selected lambda is absent from the frozen candidate set")
    formal_tuned = MethodId.UNITSCOPE_TUNED in methods
    if not args.dry_run and formal_tuned:
        if str(value.get("dataset")) != args.dataset:
            raise ValueError("lambda selection record belongs to a different dataset")
        if str(value.get("data_fingerprint")) != str(args.data_fingerprint):
            raise ValueError(
                "lambda selection record belongs to a different data release"
            )
        if args.lambda_selection_source == "public-policy":
            if value.get("selection_basis") != "fixed-public-policy":
                raise ValueError("public-policy lambda records must name a fixed public policy")
        elif value.get("public_auxiliary_only") is not True:
            raise ValueError("lambda selection must use disjoint public auxiliary users")
        if value.get("protected_test_used") is not False:
            raise ValueError(
                "lambda selection record does not certify test-set isolation"
            )
        if int(value.get("H")) != args.H or int(value.get("B")) != args.B:
            raise ValueError("lambda selection record and public H/B disagree")
        if int(value.get("n_silos")) != args.n_silos:
            raise ValueError("lambda selection record and n_silos disagree")
        if args.dataset == "femnist":
            if not np.isclose(float(value.get("overlap")), float(args.overlap)):
                raise ValueError("lambda selection record and FEMNIST overlap disagree")
            if bool(value.get("only_digits")) != bool(args.only_digits):
                raise ValueError(
                    "lambda selection record and FEMNIST digit setting disagree"
                )
            if value.get("maximum_writers") != args.maximum_writers:
                raise ValueError("lambda selection record and maximum_writers disagree")
            if int(value.get("maximum_examples_per_writer")) != int(
                args.maximum_examples_per_writer
            ):
                raise ValueError(
                    "lambda selection record and maximum_examples_per_writer disagree"
                )
        if not np.isclose(
            float(value.get("public_normalizer")), float(args.public_normalizer)
        ):
            raise ValueError("lambda selection record and public normalizer disagree")
        if int(value.get("rounds")) != int(args.rounds):
            raise ValueError("lambda selection record and rounds disagree")
        if int(value.get("records_per_round")) != int(args.records_per_round):
            raise ValueError("lambda selection record and records_per_round disagree")
        if not np.isclose(float(value.get("learning_rate")), float(args.learning_rate)):
            raise ValueError("lambda selection record and learning rate disagree")
        if str(value.get("evidence_view")) != args.evidence_view:
            raise ValueError("lambda selection record and evidence view disagree")
        if args.evidence_view == "controlled":
            if not np.isclose(float(value.get("q_user")), args.q_user):
                raise ValueError("lambda selection record and q_user disagree")
            if not np.isclose(
                float(value.get("q_record_given_handle")), args.q_record_given_handle
            ):
                raise ValueError(
                    "lambda selection record and q_record_given_handle disagree"
                )
        else:
            if str(value.get("issuance_date")) != str(args.issuance_date):
                raise ValueError(
                    "lambda selection record and health-card issuance date disagree"
                )
            if int(value.get("health_card_domains")) != args.health_card_domains:
                raise ValueError(
                    "lambda selection record and health-card domain count disagree"
                )
            recorded_silos = value.get("participating_silos")
            if recorded_silos is not None and args.participating_silos is not None:
                if sorted(map(str, recorded_silos)) != sorted(
                    map(str, args.participating_silos)
                ):
                    raise ValueError(
                        "lambda selection record and participating silos disagree"
                    )
    if formal_tuned and not args.dry_run and args.lambda_selection_record is None:
        raise ValueError(
            "formal UnitScope-Tuned runs require --lambda-selection-record"
        )
    return value


def run(args: argparse.Namespace) -> None:
    if args.dataset == "femnist" and args.overlap is None:
        args.overlap = 0.5
    if args.dataset in {"sent140", "movielens1m"} and args.overlap is None:
        args.overlap = 0.5
    if args.dataset == "synthea" and args.overlap is not None:
        raise ValueError(
            "Synthea overlap is realized from organizations and cannot be set by --overlap"
        )
    if (
        args.dataset == "synthea"
        and args.evidence_view == "controlled"
        and args.scenario != "custom"
    ):
        raise ValueError(
            "controlled Synthea views must use --scenario custom because overlap is realized, not requested"
        )
    if args.scenario in FROZEN_CONTROLLED_SLICES:
        expected = FROZEN_CONTROLLED_SLICES[args.scenario]
        checks = {
            "H": args.H,
            "B": args.B,
            "q_user": args.q_user,
            "q_record": args.q_record_given_handle,
        }
        if args.dataset == "femnist":
            checks["overlap"] = args.overlap
        mismatches = {
            name: (value, expected[name])
            for name, value in checks.items()
            if not np.isclose(float(value), float(expected[name]))
        }
        if mismatches:
            raise ValueError(
                f"{args.scenario} parameters differ from the frozen matrix: {mismatches}"
            )
    if args.dry_run:
        args.rounds = 1
        args.seeds = args.seeds[:1]
        args.maximum_writers = min(args.maximum_writers or 20, 20)
        args.maximum_examples_per_writer = min(args.maximum_examples_per_writer or 8, 8)
        args.maximum_accounts = min(args.maximum_accounts or 20, 20)
        args.maximum_records_per_account = min(args.maximum_records_per_account or 8, 8)
        args.records_per_round = min(args.records_per_round, 24)
        if not args.dry_run_all_methods:
            args.methods = [
                MethodId.STABLE_LOCAL_FALLBACK.value,
                MethodId.UNITSCOPE_TUNED.value,
            ]
        args.device = "cpu"
        args.data_fingerprint = args.data_fingerprint or "dry-run-unverified"
    task, source_dataset = prepare_task(args)
    train_records, train_truth = _training_view(task)
    realized_overlap = _realized_train_overlap(train_records, train_truth)
    if args.public_normalizer is None:
        raise ValueError(
            "public-normalizer must be frozen explicitly; it cannot be inferred from protected data"
        )
    if not args.data_fingerprint:
        raise ValueError(
            "formal runs require a frozen --data-fingerprint from the dataset provenance manifest"
        )
    if args.evidence_view == "health-card" and args.scenario != "policy-grounded":
        raise ValueError(
            "health-card evidence must be labeled --scenario policy-grounded"
        )
    if args.evidence_view == "controlled" and args.scenario == "policy-grounded":
        raise ValueError(
            "policy-grounded is reserved for the health-card evidence view"
        )
    noise_multiplier = calibrate_noise_multiplier(args.epsilon, args.rounds, args.delta)
    public_config = PublicConfig(
        task_domain=f"{args.dataset}-unitscope",
        epoch=args.epoch,
        clip_norm=args.clip_norm,
        max_handle_groups=args.H,
        max_fallback_slots=args.B,
        lambda_=args.lambda_value,
        public_normalizer=args.public_normalizer,
        noise_multiplier=noise_multiplier,
        rounds=args.rounds,
        delta=args.delta,
        n_silos=args.n_silos,
    )
    if args.evidence_view == "health-card":
        if args.dataset != "synthea":
            raise ValueError(
                "the health-card evidence view is defined only for Synthea"
            )
        participating_silos = args.participating_silos or [
            f"silo-{index}" for index in range(min(4, args.n_silos))
        ]
        evidence = generate_health_card_policy_evidence(
            source_dataset,
            [record.record_id for record in train_records],
            public_config,
            issuance_date=args.issuance_date,
            participating_silos=participating_silos,
            handle_domains=args.health_card_domains,
        )
    else:
        evidence_policy = EvidencePolicy(
            args.q_user,
            args.q_record_given_handle,
            complete_user_probability=0.0,
            handles_per_user=min(args.handles_per_user, args.H),
        )
        evidence = generate_evidence_view(
            train_records,
            train_truth,
            public_config,
            evidence_policy,
            args.evidence_seed,
        )
    full_identity = generate_evidence_view(
        train_records,
        train_truth,
        public_config,
        EvidencePolicy(0.0, 0.0, complete_user_probability=1.0),
        args.evidence_seed + 1,
    )
    evaluation_split = "validation" if args.validation_only else "test"
    evaluation_indices = np.flatnonzero(task.record_split == evaluation_split)
    evaluation_records = tuple(task.records[index] for index in evaluation_indices)
    evaluation_truth = _subset_truth(task.truth, evaluation_records)
    if args.evidence_view == "health-card":
        evaluation_evidence = generate_health_card_policy_evidence(
            source_dataset,
            [record.record_id for record in evaluation_records],
            public_config,
            issuance_date=args.issuance_date,
            participating_silos=participating_silos,
            handle_domains=args.health_card_domains,
        )
    else:
        evaluation_evidence = generate_evidence_view(
            evaluation_records,
            evaluation_truth,
            public_config,
            evidence_policy,
            args.evidence_seed + 100_000,
        )
    evaluation_groups = _evaluation_evidence_groups(
        evaluation_records, evaluation_truth, evaluation_evidence
    )
    methods = tuple(MethodId(value) for value in args.methods)
    lambda_selection = _load_lambda_selection(args, methods)
    # The controlled and policy-driven evidence views share the same compiler;
    # only the oracle-side credential generator differs.
    primary_metric = (
        "patient_or_user_auprc" if args.dataset == "synthea" else "user_macro_accuracy"
    )
    frozen_configuration = {
        "code_tree_hash": args.frozen_code_tree_hash or _code_tree_hash(),
        "dataset": args.dataset,
        "data_fingerprint": args.data_fingerprint,
        "data_path_name": args.data.name,
        "public_config": asdict(public_config),
        "task": {
            "n_silos": args.n_silos,
            "overlap_requested": args.overlap
            if args.dataset in {"femnist", "sent140", "movielens1m"}
            else None,
            "only_digits": args.only_digits,
            "maximum_writers": args.maximum_writers,
            "maximum_examples_per_writer": args.maximum_examples_per_writer,
            "writer_selection_seed": (
                args.writer_selection_seed
                if args.dataset == "femnist" and args.writer_selection_seed is not None
                else (args.split_seed if args.dataset == "femnist" else None)
            ),
            "writer_selection_offset": args.writer_selection_offset
            if args.dataset == "femnist"
            else None,
            "index_date": args.index_date if args.dataset == "synthea" else None,
            "lookback_days": args.lookback_days if args.dataset == "synthea" else None,
            "horizon_days": args.horizon_days if args.dataset == "synthea" else None,
            "dirichlet_alpha": args.dirichlet_alpha
            if args.dataset == "femnist"
            else None,
            "minimum_records_per_account": args.minimum_records_per_account
            if args.dataset == "sent140"
            else None,
            "maximum_accounts": args.maximum_accounts
            if args.dataset in {"sent140", "movielens1m"}
            else None,
            "maximum_records_per_account": args.maximum_records_per_account
            if args.dataset in {"sent140", "movielens1m"}
            else None,
            "account_selection_seed": args.account_selection_seed
            if args.dataset in {"sent140", "movielens1m"}
            else None,
            "text_hash_dimension": args.text_hash_dimension
            if args.dataset == "sent140"
            else None,
            "ambiguous_duplicate_policy": "drop-all-occurrences-of-repeated-tweet-id"
            if args.dataset == "sent140"
            else None,
            "stable_user_label": task.stable_user_label,
            "movielens_task_variant": getattr(source_dataset, "task_variant", None),
            "balance_seed": args.balance_seed
            if args.dataset == "movielens1m"
            else None,
            "balance_majority_keep_probability": getattr(
                source_dataset, "balance_majority_keep_probability", None
            ),
        },
        "training": {
            "rounds": args.rounds,
            "evaluation_checkpoints": sorted(
                {0, args.rounds, *range(10, args.rounds + 1, 10)}
            ),
            "records_per_round": args.records_per_round,
            "per_sample_grad_chunk": args.grad_chunk,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "evaluation_split": evaluation_split,
        },
        "methods": [method.value for method in methods],
        "seeds": args.seeds,
        "seed_bases": {
            "split": args.split_seed,
            "topology": args.topology_seed,
            "evidence": args.evidence_seed,
            "schedule": args.schedule_seed,
            "noise": args.noise_seed,
        },
        "evidence": {
            "view": args.evidence_view,
            "q_user_requested": args.q_user
            if args.evidence_view == "controlled"
            else None,
            "q_record_given_handle_requested": args.q_record_given_handle
            if args.evidence_view == "controlled"
            else None,
            "handles_per_user": args.handles_per_user,
            "issuance_date": args.issuance_date
            if args.evidence_view == "health-card"
            else None,
            "participating_silos": args.participating_silos
            if args.evidence_view == "health-card"
            else None,
            "health_card_domains": args.health_card_domains
            if args.evidence_view == "health-card"
            else None,
            "evidence_hash": _evidence_hash(evidence),
            "full_identity_reference_hash": _evidence_hash(full_identity),
        },
        "record_view_hash": _record_view_hash(task.records, task.record_split),
        "lambda_selection": lambda_selection,
        "scenario": args.scenario,
        "primary_metric": primary_metric,
        "sampling_amplification_used": False,
    }
    config_hash = _hash_payload(frozen_configuration)
    manifest = {
        "config_hash": config_hash,
        "plan_config_hash": public_config.config_hash,
        "frozen_configuration": frozen_configuration,
        "realized": {
            "q_user_realized": evidence.realized_handle_user_fraction,
            "q_record_given_handle_realized": evidence.realized_record_coverage_given_handle,
            "q_record_given_handle_pooled_realized": evidence.realized_pooled_record_coverage_given_handle,
            "q_record_total_realized": evidence.realized_total_record_coverage,
            "handle_users_with_residual_fraction": evidence.handle_users_with_residual_fraction,
            "handle_users_fully_covered_fraction": evidence.handle_users_fully_covered_fraction,
            "mean_residual_records_per_handle_user": evidence.mean_residual_records_per_handle_user,
            "overlap_fraction": realized_overlap,
            "ambiguous_duplicate_rows_removed": getattr(
                source_dataset, "ambiguous_duplicate_rows_removed", 0
            ),
        },
        "privacy": {
            "epsilon": args.epsilon,
            "delta": args.delta,
            "sampling_amplification_used": False,
        },
        "environment": environment_fingerprint(),
        "dry_run": args.dry_run,
    }
    manifest_hash = _hash_payload(manifest)
    manifest["manifest_hash"] = manifest_hash
    manifest_path = args.output / "learning_manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing_manifest.get("config_hash") != config_hash
            or existing_manifest.get("manifest_hash") != manifest_hash
        ):
            raise ValueError(
                "output directory already belongs to a different frozen configuration or environment; "
                "use one directory per config hash"
            )
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {"prepared_config_hash": config_hash, "manifest_hash": manifest_hash},
            ensure_ascii=False,
        )
    )
    if args.prepare_only:
        return
    if args.confirmatory:
        if args.confirmatory_registry is None:
            raise ValueError("confirmatory runs require --confirmatory-registry")
        registry = json.loads(args.confirmatory_registry.read_text(encoding="utf-8"))
        if config_hash not in set(registry.get("config_hashes", [])):
            raise ValueError(
                "configuration is absent from the frozen confirmatory registry"
            )
    result_path = args.output / "learning_results.jsonl"
    run_directory = args.output / "runs"
    completed = _load_completed(run_directory)
    for seed in args.seeds:
        for method in methods:
            run_id = hashlib.sha256(
                f"{config_hash}|{seed}|{method.value}".encode("utf-8")
            ).hexdigest()
            if run_id in completed:
                continue
            training = TrainingConfig(
                args.rounds,
                args.records_per_round,
                args.grad_chunk,
                args.learning_rate,
                args.device,
            )
            seeds = TrainingSeeds(
                seed, args.schedule_seed + seed, args.noise_seed + seed * 10_000
            )
            output = train_method(
                method,
                task,
                evidence.evidence,
                evidence.registry,
                full_identity.evidence,
                full_identity.registry,
                public_config,
                training,
                seeds,
                evaluation_groups,
                evaluation_split=evaluation_split,
            )
            spec = PRIMARY_METHODS[method]
            row = {
                "run_id": run_id,
                "config_hash": config_hash,
                "manifest_hash": manifest_hash,
                "plan_config_hash": public_config.config_hash,
                "dataset": args.dataset,
                "evaluation_split": evaluation_split,
                "scenario": args.scenario,
                "seed": seed,
                "split_seed": args.split_seed,
                "topology_seed": args.topology_seed,
                "evidence_seed": args.evidence_seed,
                "schedule_seed": args.schedule_seed + seed,
                "noise_seed": args.noise_seed + seed * 10_000,
                "method": method.value,
                "implementation_provenance": spec.provenance.value,
                "privacy_semantics": spec.privacy_semantics.value,
                "comparison_tier": spec.comparison_tier.value,
                "lambda": _reported_lambda(method, public_config),
                "H": public_config.H,
                "B": public_config.B,
                "C": public_config.C,
                "overlap_probability": args.overlap
                if args.dataset in {"femnist", "sent140", "movielens1m"}
                else None,
                "realized_overlap_fraction": realized_overlap,
                "certified_sensitivity": output.domain_plan.certified_sensitivity,
                "structural_sensitivity_bound": output.domain_plan.structural_sensitivity_bound,
                "certificate_type": output.domain_plan.certificate_type,
                "noise_sensitivity": output.domain_plan.noise_sensitivity,
                "validation_pass": output.domain_plan.validation_pass,
                "complete_units": output.domain_plan.complete_units,
                "partial_handle_units": output.domain_plan.partial_handle_units,
                "local_atom_units": output.domain_plan.local_atom_units,
                "local_mixed_units": output.domain_plan.local_mixed_units,
                "dropped_units": output.domain_plan.dropped_units,
                "public_normalizer": public_config.public_normalizer,
                "noise_multiplier": public_config.noise_multiplier,
                "epsilon": None
                if spec.privacy_semantics.value in {"non-private", "unsafe-diagnostic"}
                else args.epsilon,
                "delta": None
                if spec.privacy_semantics.value in {"non-private", "unsafe-diagnostic"}
                else args.delta,
                "q_user_requested": args.q_user
                if args.evidence_view == "controlled"
                else None,
                "q_record_given_handle_requested": args.q_record_given_handle
                if args.evidence_view == "controlled"
                else None,
                "q_user": evidence.realized_handle_user_fraction,
                "q_record_given_handle": evidence.realized_record_coverage_given_handle,
                "q_record_given_handle_pooled": evidence.realized_pooled_record_coverage_given_handle,
                "q_record_total": evidence.realized_total_record_coverage,
                "handle_users_with_residual_fraction": evidence.handle_users_with_residual_fraction,
                "handle_users_fully_covered_fraction": evidence.handle_users_fully_covered_fraction,
                "mean_residual_records_per_handle_user": evidence.mean_residual_records_per_handle_user,
                "evidence_view": args.evidence_view,
                **asdict(output.result),
            }
            row["metric_name"] = primary_metric
            row["metric_value"] = row[row["metric_name"]]
            validate_result_row(row)
            atomic_write_json(
                args.output / "trajectories" / f"{run_id}.json",
                {
                    "run_id": run_id,
                    "dataset": args.dataset,
                    "seed": seed,
                    "method": method.value,
                    "lambda": _reported_lambda(method, public_config),
                    "trajectory": [asdict(point) for point in output.trajectory],
                },
            )
            atomic_write_json(
                args.output / "clusters" / f"{run_id}.json",
                {
                    "run_id": run_id,
                    "dataset": args.dataset,
                    "seed": seed,
                    "method": method.value,
                    "clusters": [asdict(cluster) for cluster in output.clusters],
                },
            )
            atomic_write_json(
                args.output / "residual_influence" / f"{run_id}.json",
                {
                    "run_id": run_id,
                    "dataset": args.dataset,
                    "seed": seed,
                    "method": method.value,
                    "residual_influence": [
                        asdict(point) for point in output.residual_influence
                    ],
                },
            )
            atomic_write_json(run_directory / f"{run_id}.json", row)
            _rewrite_jsonl(result_path, run_directory)
            completed.add(run_id)
            print(json.dumps(row, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the UnitScope learning protocol"
    )
    parser.add_argument(
        "--dataset",
        choices=("femnist", "synthea", "sent140", "movielens1m"),
        required=True,
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--data-fingerprint",
        help="Source SHA-256 from the data provenance file; required outside dry-run",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-all-methods", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--confirmatory", action="store_true")
    parser.add_argument("--confirmatory-registry", type=Path)
    parser.add_argument(
        "--frozen-code-tree-hash",
        help="Code-tree hash recorded in the run registry",
    )
    parser.add_argument(
        "--scenario",
        choices=(
            "controlled-favorable",
            "controlled-middle",
            "controlled-stress",
            "policy-grounded",
            "custom",
        ),
        default="controlled-middle",
    )
    parser.add_argument(
        "--methods", nargs="+", default=[method.value for method in DEFAULT_METHODS]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--split-seed", type=int, default=31001)
    parser.add_argument("--topology-seed", type=int, default=31002)
    parser.add_argument("--evidence-seed", type=int, default=31003)
    parser.add_argument("--schedule-seed", type=int, default=31004)
    parser.add_argument("--noise-seed", type=int, default=31005)
    parser.add_argument("--n-silos", type=int, default=6)
    parser.add_argument("--overlap", type=float)
    parser.add_argument("--q-user", type=float, default=0.5)
    parser.add_argument("--q-record-given-handle", type=float, default=0.5)
    parser.add_argument(
        "--evidence-view", choices=("controlled", "health-card"), default="controlled"
    )
    parser.add_argument("--issuance-date", default="2019-07-01")
    parser.add_argument("--participating-silos", nargs="+")
    parser.add_argument("--health-card-domains", type=int, default=1)
    parser.add_argument("--handles-per-user", type=int, default=1)
    parser.add_argument("--H", type=int, default=1)
    parser.add_argument("--B", type=int, default=6)
    parser.add_argument("--lambda-value", type=float, default=0.5)
    parser.add_argument(
        "--lambda-selection-source",
        choices=tuple(sorted(ALLOWED_LAMBDA_SOURCES)),
        default="public-policy",
    )
    parser.add_argument("--lambda-policy-id", default="unitscope-default")
    parser.add_argument("--lambda-selection-record", type=Path)
    parser.add_argument("--protected-test-used-for-selection", action="store_true")
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--public-normalizer", type=float, required=True)
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--records-per-round", type=int, default=512)
    parser.add_argument("--grad-chunk", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epoch", default="frozen-v1")
    parser.add_argument("--only-digits", action="store_true")
    parser.add_argument("--maximum-writers", type=int)
    parser.add_argument("--maximum-examples-per-writer", type=int, default=128)
    parser.add_argument("--writer-selection-seed", type=int)
    parser.add_argument("--writer-selection-offset", type=int, default=0)
    parser.add_argument("--dirichlet-alpha", type=float)
    parser.add_argument("--minimum-records-per-account", type=int, default=2)
    parser.add_argument("--maximum-accounts", type=int)
    parser.add_argument("--maximum-records-per-account", type=int, default=32)
    parser.add_argument("--account-selection-seed", type=int, default=31002)
    parser.add_argument(
        "--movielens-task-variant",
        choices=("original", "balanced-v1"),
        default="original",
    )
    parser.add_argument("--balance-seed", type=int, default=37101)
    parser.add_argument("--text-hash-dimension", type=int, default=512)
    parser.add_argument("--index-date", default="2020-01-01")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--horizon-days", type=int, default=365)
    parser.add_argument("--validation-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
