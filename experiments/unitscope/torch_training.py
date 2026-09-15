from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np

from .accounting import conservative_gaussian_rdp
from .compiler import (
    compile_plan,
    validate_plan_with_oracle,
    validate_record_level_plan,
    with_oracle_validation,
)
from .model import (
    CertificateType,
    ContributionPlan,
    EvidenceView,
    GroundTruth,
    MethodId,
    PermanentRegistry,
    PublicConfig,
    PublicRecord,
    UnitKind,
)


@dataclass(frozen=True)
class LearningTask:
    records: Tuple[PublicRecord, ...]
    truth: GroundTruth
    x: np.ndarray
    y: np.ndarray
    record_split: np.ndarray
    n_classes: int
    task_kind: str
    stable_user_label: bool = False


@dataclass(frozen=True)
class TrainingSeeds:
    initialization: int
    schedule: int
    noise: int


@dataclass(frozen=True)
class TrainingConfig:
    rounds: int
    records_per_round: int
    per_sample_grad_chunk: int
    learning_rate: float
    device: str


@dataclass(frozen=True)
class TrainingResult:
    method: str
    sample_accuracy: float
    user_macro_accuracy: float
    patient_or_user_auprc: float | None
    clip_fraction: float
    effective_record_utilization: float
    partial_handle_record_fraction: float
    residual_record_fraction: float
    aggregate_snr: float | None
    gradient_distortion_l2: float
    partial_gradient_distortion_l2: float
    residual_gradient_distortion_l2: float
    partial_clip_fraction: float
    residual_clip_fraction: float
    residual_norm_share: float
    residual_clip_saturation_rate: float | None
    residual_only_subgroup_metric: float | None
    residual_only_subgroup_people: int
    residual_only_subgroup_records: int
    record_gradient_evaluations: int
    non_dummy_record_gradient_evaluations: int
    contributing_record_gradient_evaluations: int
    group_gradient_evaluations: int
    noisy_releases: int
    wall_time_seconds: float
    oracle_audit_seconds: float
    diagnostic_gradient_seconds: float
    diagnostic_record_gradient_evaluations: int
    positive_prediction_fraction: float | None
    prediction_entropy_nats: float | None
    class_zero_recall: float | None
    class_one_recall: float | None
    majority_class_accuracy: float


@dataclass(frozen=True)
class ClusterEvaluation:
    cluster_id: str
    records: int
    accuracy: float
    binary_label: int | None
    positive_probability: float | None
    evidence_group: str | None = None


@dataclass(frozen=True)
class TrainingOutput:
    result: TrainingResult
    clusters: Tuple[ClusterEvaluation, ...]
    domain_plan: "DomainPlanSummary"
    trajectory: Tuple["TrajectoryPoint", ...]
    residual_influence: Tuple["ResidualInfluencePoint", ...]


@dataclass(frozen=True)
class TrajectoryPoint:
    round: int
    epsilon: float | None
    sample_accuracy: float
    user_macro_accuracy: float
    patient_or_user_auprc: float | None


@dataclass(frozen=True)
class ResidualInfluencePoint:
    round: int
    residual_norm_share: float
    residual_clip_saturation_rate: float | None
    active_residual_units: int
    saturated_residual_units: int


@dataclass(frozen=True)
class DomainPlanSummary:
    certificate_type: str
    certified_sensitivity: float | None
    structural_sensitivity_bound: float | None
    noise_sensitivity: float
    validation_pass: bool
    validation_errors: Tuple[str, ...]
    complete_units: int
    partial_handle_units: int
    local_atom_units: int
    local_mixed_units: int
    dropped_units: int


@dataclass(frozen=True)
class PlanExecution:
    released: object
    clip_fraction: float
    utilization: float
    active_units: int
    aggregate_snr: float
    gradient_distortion_l2: float
    partial_gradient_distortion_l2: float
    residual_gradient_distortion_l2: float
    partial_clip_fraction: float
    residual_clip_fraction: float
    residual_norm_share: float
    residual_clip_saturation_rate: float | None
    active_residual_units: int
    saturated_residual_units: int


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from torch.func import functional_call, grad, vmap
    except ImportError as error:
        raise RuntimeError("PyTorch is required for learning experiments") from error
    return torch, nn, functional, functional_call, grad, vmap


def make_model(task: LearningTask):
    torch, nn, _, _, _, _ = require_torch()
    if task.task_kind == "femnist":

        class SmallCNN(nn.Module):
            def __init__(self, n_classes: int):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(32 * 7 * 7, 64),
                    nn.ReLU(),
                    nn.Linear(64, n_classes),
                )

            def forward(self, value):
                return self.classifier(self.features(value))

        return SmallCNN(task.n_classes)
    if task.task_kind in {"structured", "movielens-balanced"}:
        input_dim = int(np.prod(task.x.shape[1:]))
        hidden = 64 if task.task_kind == "movielens-balanced" else 32
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, task.n_classes),
        )
    if task.task_kind == "text":
        input_dim = int(np.prod(task.x.shape[1:]))
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, task.n_classes),
        )
    raise ValueError(f"unknown task kind {task.task_kind}")


def _prepare_x(task: LearningTask, indices: np.ndarray, device: str):
    torch, _, _, _, _, _ = require_torch()
    value = torch.as_tensor(task.x[indices], device=device)
    if task.task_kind == "femnist":
        value = 1.0 - value.float().unsqueeze(1) / 255.0
    else:
        value = value.float()
    return value


def per_sample_gradients(model, x, y, chunk_size: int):
    torch, _, functional, functional_call, grad, vmap = require_torch()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    parameters = dict(model.named_parameters())
    names = tuple(parameters)

    def single_loss(current_parameters, sample, target):
        logits = functional_call(model, current_parameters, (sample.unsqueeze(0),))
        return functional.cross_entropy(logits, target.unsqueeze(0), reduction="sum")

    gradient_function = grad(single_loss)
    chunks = []
    for start in range(0, len(x), chunk_size):
        stop = min(len(x), start + chunk_size)
        gradients = vmap(gradient_function, in_dims=(None, 0, 0))(
            parameters, x[start:stop], y[start:stop]
        )
        chunks.append(
            torch.cat(
                [gradients[name].reshape(stop - start, -1) for name in names], dim=1
            )
        )
    return torch.cat(chunks, dim=0), names


def _flat_parameter_size(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def apply_flat_update(model, update, learning_rate: float) -> None:
    torch, _, _, _, _, _ = require_torch()
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            length = parameter.numel()
            parameter.add_(
                update[offset : offset + length].reshape_as(parameter),
                alpha=-learning_rate,
            )
            offset += length
    if offset != update.numel():
        raise ValueError("flat update does not match model parameters")


def execute_plan_torch(
    records: Sequence[PublicRecord],
    record_gradients,
    plan: ContributionPlan,
    public_config: PublicConfig,
    standard_normal,
    reference_record_gradients=None,
):
    torch, _, _, _, _, _ = require_torch()
    if not plan.validation.valid:
        raise ValueError(f"invalid plan: {plan.validation.errors}")
    if record_gradients.ndim != 2 or record_gradients.shape[0] != len(records):
        raise ValueError("record gradient shape mismatch")
    reference_gradients = (
        record_gradients
        if reference_record_gradients is None
        else reference_record_gradients
    )
    if reference_gradients.shape != record_gradients.shape:
        raise ValueError("reference gradient shape mismatch")
    index = {record.record_id: position for position, record in enumerate(records)}
    unit_for_record = plan.unit_for_record()
    if set(index) != set(unit_for_record):
        raise ValueError("plan and active record set differ")
    if plan.certificate_type == CertificateType.NON_PRIVATE:
        weights = torch.as_tensor(
            [record.query_weight for record in records], device=record_gradients.device
        )
        deterministic = (record_gradients * weights[:, None]).sum(dim=0)
        return PlanExecution(
            deterministic / public_config.public_normalizer,
            0.0,
            1.0,
            len(records),
            float("inf"),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            None,
            0,
            0,
        )

    unit_index = {unit.label: position for position, unit in enumerate(plan.units)}
    assignment = torch.as_tensor(
        [unit_index[unit_for_record[record.record_id].label] for record in records],
        device=record_gradients.device,
        dtype=torch.long,
    )
    weights = torch.as_tensor(
        [record.query_weight for record in records],
        device=record_gradients.device,
        dtype=record_gradients.dtype,
    )
    sums = torch.zeros(
        (len(plan.units), record_gradients.shape[1]),
        device=record_gradients.device,
        dtype=record_gradients.dtype,
    )
    sums.index_add_(0, assignment, record_gradients * weights[:, None])
    reference_sums = torch.zeros_like(sums)
    reference_sums.index_add_(0, assignment, reference_gradients * weights[:, None])
    radii = torch.as_tensor(
        [unit.radius for unit in plan.units],
        device=record_gradients.device,
        dtype=record_gradients.dtype,
    )
    norms = torch.linalg.vector_norm(sums, dim=1)
    factors = torch.ones_like(norms)
    positive = norms > 0
    factors[positive] = torch.minimum(
        torch.ones_like(norms[positive]), radii[positive] / norms[positive]
    )
    factors[radii == 0] = 0
    clipped_sums = sums * factors[:, None]
    deterministic = clipped_sums.sum(dim=0)
    noise = standard_normal * public_config.noise_multiplier * plan.noise_sensitivity
    released = (deterministic + noise) / public_config.public_normalizer
    utilized = sum(
        len(unit.record_ids) for unit in plan.units if unit.kind != UnitKind.DROPPED
    )
    active_mask = torch.as_tensor(
        [unit.radius > 0 and unit.kind != UnitKind.DROPPED for unit in plan.units],
        device=record_gradients.device,
        dtype=torch.bool,
    )
    partial_mask = torch.as_tensor(
        [unit.kind == UnitKind.PARTIAL_HANDLE for unit in plan.units],
        device=record_gradients.device,
        dtype=torch.bool,
    )
    residual_mask = torch.as_tensor(
        [
            unit.kind in (UnitKind.LOCAL_ATOM, UnitKind.LOCAL_MIXED, UnitKind.DROPPED)
            for unit in plan.units
        ],
        device=record_gradients.device,
        dtype=torch.bool,
    )
    clipped_mask = factors < 1 - 1e-12

    def _masked_fraction(mask) -> float:
        effective = mask & active_mask
        return (
            float(clipped_mask[effective].float().mean().item())
            if bool(effective.any())
            else 0.0
        )

    def _masked_distortion(mask) -> float:
        if not bool(mask.any()):
            return 0.0
        difference = (reference_sums[mask] - clipped_sums[mask]).sum(
            dim=0
        ) / public_config.public_normalizer
        return float(torch.linalg.vector_norm(difference).item())

    clip_fraction = _masked_fraction(active_mask)
    utilization = utilized / len(records) if records else 0.0
    noise_norm = float(torch.linalg.vector_norm(noise).item())
    signal_norm = float(torch.linalg.vector_norm(deterministic).item())
    snr = signal_norm / noise_norm if noise_norm > 0 else float("inf")
    # Compute diagnostics on CPU to avoid changing later CUDA reductions.
    clipped_sums_cpu = clipped_sums.detach().cpu().numpy()
    residual_mask_cpu = residual_mask.detach().cpu().numpy()
    active_mask_cpu = active_mask.detach().cpu().numpy()
    clipped_mask_cpu = clipped_mask.detach().cpu().numpy()
    active_residual_mask_cpu = residual_mask_cpu & active_mask_cpu
    active_nonresidual_mask_cpu = (~residual_mask_cpu) & active_mask_cpu

    def _masked_aggregate_norm_cpu(mask: np.ndarray) -> float:
        if not bool(mask.any()):
            return 0.0
        aggregate = clipped_sums_cpu[mask].sum(axis=0)
        return float(np.linalg.norm(aggregate))

    residual_norm = _masked_aggregate_norm_cpu(active_residual_mask_cpu)
    nonresidual_norm = _masked_aggregate_norm_cpu(active_nonresidual_mask_cpu)
    norm_denominator = residual_norm + nonresidual_norm
    residual_norm_share = (
        residual_norm / norm_denominator if norm_denominator > 0 else 0.0
    )
    active_residual_units = int(active_residual_mask_cpu.sum())
    saturated_residual_units = int((clipped_mask_cpu & active_residual_mask_cpu).sum())
    residual_clip_saturation_rate = (
        saturated_residual_units / active_residual_units
        if active_residual_units
        else None
    )
    all_mask = torch.ones(
        len(plan.units), device=record_gradients.device, dtype=torch.bool
    )
    return PlanExecution(
        released,
        clip_fraction,
        utilization,
        len(plan.active_units),
        snr,
        _masked_distortion(all_mask),
        _masked_distortion(partial_mask),
        _masked_distortion(residual_mask),
        _masked_fraction(partial_mask),
        _masked_fraction(residual_mask),
        residual_norm_share,
        residual_clip_saturation_rate,
        active_residual_units,
        saturated_residual_units,
    )


def _stable_schedule(
    record_ids: Sequence[str], round_index: int, seed: int, count: int
) -> np.ndarray:
    if count >= len(record_ids):
        return np.arange(len(record_ids), dtype=np.int64)
    ranked = sorted(
        range(len(record_ids)),
        key=lambda index: int.from_bytes(
            hashlib.sha256(
                f"{seed}|{round_index}|{record_ids[index]}".encode("utf-8")
            ).digest()[:8],
            "big",
        ),
    )
    return np.asarray(ranked[:count], dtype=np.int64)


def _subset_truth(truth: GroundTruth, records: Sequence[PublicRecord]) -> GroundTruth:
    truth_map = truth.as_dict()
    return GroundTruth(
        tuple((record.record_id, truth_map[record.record_id]) for record in records)
    )


def _evaluate(
    model,
    task: LearningTask,
    indices: np.ndarray,
    device: str,
    evaluation_groups: Mapping[str, str] | None = None,
):
    torch, _, _, _, _, _ = require_torch()
    from sklearn.metrics import average_precision_score

    model.eval()
    truth_map = task.truth.as_dict()
    probabilities = []
    predictions = []
    labels = []
    users = []
    batch_size = 512
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            x = _prepare_x(task, batch_indices, device)
            logits = model(x)
            probability = torch.softmax(logits, dim=1)
            probabilities.append(probability.cpu().numpy())
            predictions.append(probability.argmax(dim=1).cpu().numpy())
            labels.append(task.y[batch_indices])
            users.extend(
                truth_map[task.records[index].record_id] for index in batch_indices
            )
    probability = np.concatenate(probabilities)
    prediction = np.concatenate(predictions)
    label = np.concatenate(labels)
    sample_accuracy = float(np.mean(prediction == label))
    label_counts = np.bincount(label, minlength=task.n_classes)
    majority_class_accuracy = float(label_counts.max() / label_counts.sum())
    prediction_counts = np.bincount(prediction, minlength=task.n_classes)
    prediction_frequencies = prediction_counts / prediction_counts.sum()
    nonzero_prediction_frequencies = prediction_frequencies[prediction_frequencies > 0]
    prediction_entropy_nats = float(
        -np.sum(nonzero_prediction_frequencies * np.log(nonzero_prediction_frequencies))
    )
    positive_prediction_fraction = None
    class_zero_recall = None
    class_one_recall = None
    if task.n_classes == 2:
        positive_prediction_fraction = float(prediction_frequencies[1])
        class_zero_mask = label == 0
        class_one_mask = label == 1
        class_zero_recall = (
            float(np.mean(prediction[class_zero_mask] == 0))
            if class_zero_mask.any()
            else None
        )
        class_one_recall = (
            float(np.mean(prediction[class_one_mask] == 1))
            if class_one_mask.any()
            else None
        )
    per_user = {}
    cluster_rows = []
    for user in sorted(set(users)):
        mask = np.asarray([candidate == user for candidate in users])
        per_user[user] = float(np.mean(prediction[mask] == label[mask]))
    user_macro_accuracy = float(np.mean(list(per_user.values())))
    auprc = None
    if task.n_classes == 2 and task.stable_user_label:
        user_probability = []
        user_label = []
        for user in sorted(per_user):
            mask = np.asarray([candidate == user for candidate in users])
            user_probability.append(float(probability[mask, 1].mean()))
            unique_labels = np.unique(label[mask])
            if len(unique_labels) != 1:
                raise ValueError(
                    "patient/user labels must be stable within the evaluation horizon"
                )
            user_label.append(int(unique_labels[0]))
            cluster_rows.append(
                ClusterEvaluation(
                    cluster_id=hashlib.sha256(
                        f"evaluation-cluster|{user}".encode("utf-8")
                    ).hexdigest()[:24],
                    records=int(mask.sum()),
                    accuracy=per_user[user],
                    binary_label=int(unique_labels[0]),
                    positive_probability=float(probability[mask, 1].mean()),
                    evidence_group=None
                    if evaluation_groups is None
                    else evaluation_groups.get(user),
                )
            )
        if len(set(user_label)) > 1:
            auprc = float(average_precision_score(user_label, user_probability))
    else:
        for user in sorted(per_user):
            mask = np.asarray([candidate == user for candidate in users])
            cluster_rows.append(
                ClusterEvaluation(
                    cluster_id=hashlib.sha256(
                        f"evaluation-cluster|{user}".encode("utf-8")
                    ).hexdigest()[:24],
                    records=int(mask.sum()),
                    accuracy=per_user[user],
                    binary_label=None,
                    positive_probability=None,
                    evidence_group=None
                    if evaluation_groups is None
                    else evaluation_groups.get(user),
                )
            )
    return (
        sample_accuracy,
        user_macro_accuracy,
        auprc,
        tuple(cluster_rows),
        positive_prediction_fraction,
        prediction_entropy_nats,
        class_zero_recall,
        class_one_recall,
        majority_class_accuracy,
    )


def train_method(
    method: MethodId,
    task: LearningTask,
    evidence: EvidenceView,
    registry: PermanentRegistry,
    full_identity_evidence: EvidenceView,
    full_identity_registry: PermanentRegistry,
    public_config: PublicConfig,
    training_config: TrainingConfig,
    seeds: TrainingSeeds,
    evaluation_groups: Mapping[str, str] | None = None,
    evaluation_split: str = "test",
) -> TrainingOutput:
    torch, _, _, _, _, _ = require_torch()
    started = time.perf_counter()
    torch.manual_seed(seeds.initialization)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeds.initialization)
    model = make_model(task).to(training_config.device)
    train_indices = np.flatnonzero(task.record_split == "train")
    evaluation_indices = np.flatnonzero(task.record_split == evaluation_split)
    if len(train_indices) == 0 or len(evaluation_indices) == 0:
        raise ValueError("training task needs non-empty train and evaluation splits")
    train_index_by_record = {
        task.records[index].record_id: int(index) for index in train_indices
    }
    frozen_record_universe = tuple(
        evidence_record for evidence_record in registry.record_universe
    )
    if not set(train_index_by_record).issubset(set(frozen_record_universe)):
        raise ValueError(
            "every active training record must belong to the permanent slot universe"
        )
    full_training_records = tuple(task.records[index] for index in train_indices)
    full_training_truth = _subset_truth(task.truth, full_training_records)
    oracle_identity_methods = {
        MethodId.FULL_IDENTITY,
        MethodId.ULDP_AVG_WEIGHTED,
        MethodId.GROUP_PRIVACY_2,
    }
    if method in oracle_identity_methods:
        domain_evidence, domain_registry = (
            full_identity_evidence,
            full_identity_registry,
        )
    else:
        domain_evidence, domain_registry = evidence, registry
    domain_plan = compile_plan(
        method, full_training_records, domain_evidence, domain_registry, public_config
    )
    oracle_audit_started = time.perf_counter()
    if method == MethodId.RECORD_LEVEL_DP:
        domain_validation = validate_record_level_plan(
            domain_plan, full_training_records, public_config
        )
    else:
        domain_validation = validate_plan_with_oracle(
            domain_plan, full_training_records, full_training_truth, public_config
        )
    oracle_audit_seconds = time.perf_counter() - oracle_audit_started
    if not domain_validation.valid:
        raise RuntimeError(
            f"the full admissible training domain failed before round 0: {domain_validation.errors}"
        )
    domain_plan = with_oracle_validation(domain_plan, domain_validation)
    domain_summary = DomainPlanSummary(
        certificate_type=domain_plan.certificate_type.value,
        certified_sensitivity=None
        if domain_plan.certified_sensitivity is None
        else domain_plan.certified_sensitivity.value,
        structural_sensitivity_bound=domain_plan.structural_sensitivity_bound,
        noise_sensitivity=domain_plan.noise_sensitivity,
        validation_pass=domain_validation.valid,
        validation_errors=domain_validation.errors,
        complete_units=sum(
            unit.kind == UnitKind.COMPLETE for unit in domain_plan.units
        ),
        partial_handle_units=sum(
            unit.kind == UnitKind.PARTIAL_HANDLE for unit in domain_plan.units
        ),
        local_atom_units=sum(
            unit.kind == UnitKind.LOCAL_ATOM for unit in domain_plan.units
        ),
        local_mixed_units=sum(
            unit.kind == UnitKind.LOCAL_MIXED for unit in domain_plan.units
        ),
        dropped_units=sum(unit.kind == UnitKind.DROPPED for unit in domain_plan.units),
    )
    clip_fractions = []
    utilizations = []
    group_evaluations = 0
    aggregate_snrs = []
    gradient_distortions = []
    partial_gradient_distortions = []
    residual_gradient_distortions = []
    partial_clip_fractions = []
    residual_clip_fractions = []
    residual_norm_shares = []
    residual_active_unit_events = 0
    residual_saturated_unit_events = 0
    partial_records_total = 0
    residual_records_total = 0
    active_records_total = 0
    record_gradient_evaluations = 0
    non_dummy_record_gradient_evaluations = 0
    contributing_record_gradient_evaluations = 0
    diagnostic_gradient_seconds = 0.0
    diagnostic_record_gradient_evaluations = 0
    checkpoint_rounds = {0, training_config.rounds}
    checkpoint_rounds.update(range(10, training_config.rounds + 1, 10))
    trajectory = []
    residual_influence = []

    def evaluate_checkpoint(completed_rounds: int) -> None:
        sample, macro, auprc, _, _, _, _, _, _ = _evaluate(
            model, task, evaluation_indices, training_config.device, evaluation_groups
        )
        epsilon = None
        if completed_rounds == 0 and domain_plan.certified_sensitivity is not None:
            epsilon = 0.0
        elif completed_rounds > 0 and domain_plan.certified_sensitivity is not None:
            epsilon = conservative_gaussian_rdp(
                domain_plan.certified_sensitivity,
                public_config.noise_multiplier * domain_plan.noise_sensitivity,
                completed_rounds,
                public_config.delta,
            ).epsilon
        trajectory.append(
            TrajectoryPoint(completed_rounds, epsilon, sample, macro, auprc)
        )

    if 0 in checkpoint_rounds:
        evaluate_checkpoint(0)
    for round_index in range(training_config.rounds):
        selected_positions = _stable_schedule(
            frozen_record_universe,
            round_index,
            seeds.schedule,
            min(training_config.records_per_round, len(frozen_record_universe)),
        )
        selected_record_ids = [
            frozen_record_universe[position] for position in selected_positions
        ]
        # The schedule is frozen over permanent slots. An absent record leaves
        # an empty slot; no currently present record is promoted as a refill.
        selected_indices = np.asarray(
            [
                train_index_by_record[record_id]
                for record_id in selected_record_ids
                if record_id in train_index_by_record
            ],
            dtype=np.int64,
        )
        records = tuple(task.records[index] for index in selected_indices)
        current_truth = _subset_truth(task.truth, records)
        if method in oracle_identity_methods:
            current_evidence, current_registry = (
                full_identity_evidence,
                full_identity_registry,
            )
        else:
            current_evidence, current_registry = evidence, registry
        plan = compile_plan(
            method, records, current_evidence, current_registry, public_config
        )
        oracle_audit_started = time.perf_counter()
        if method == MethodId.RECORD_LEVEL_DP:
            validation = validate_record_level_plan(plan, records, public_config)
        else:
            validation = validate_plan_with_oracle(
                plan, records, current_truth, public_config
            )
        oracle_audit_seconds += time.perf_counter() - oracle_audit_started
        plan = with_oracle_validation(plan, validation)
        if not validation.valid:
            raise RuntimeError(
                f"round {round_index} plan failed validation: {validation.errors}"
            )
        model.train()
        if len(selected_indices):
            x = _prepare_x(task, selected_indices, training_config.device)
            y = torch.as_tensor(
                task.y[selected_indices],
                device=training_config.device,
                dtype=torch.long,
            )
            dropped_ids = {
                record_id
                for unit in plan.units
                if unit.kind == UnitKind.DROPPED
                for record_id in unit.record_ids
            }
            dropped_mask = torch.as_tensor(
                [record.record_id in dropped_ids for record in records],
                device=training_config.device,
                dtype=torch.bool,
            )
            if bool(dropped_mask.any()):
                dummy_x = x.clone()
                dummy_y = y.clone()
                dummy_x[dropped_mask] = 0
                dummy_y[dropped_mask] = 0
                release_gradients, _ = per_sample_gradients(
                    model, dummy_x, dummy_y, training_config.per_sample_grad_chunk
                )
                diagnostic_started = time.perf_counter()
                reference_gradients, _ = per_sample_gradients(
                    model, x, y, training_config.per_sample_grad_chunk
                )
                diagnostic_gradient_seconds += time.perf_counter() - diagnostic_started
                diagnostic_record_gradient_evaluations += len(records)
            else:
                release_gradients, _ = per_sample_gradients(
                    model, x, y, training_config.per_sample_grad_chunk
                )
                reference_gradients = release_gradients
            release_gradients = release_gradients.clone()
            release_gradients[dropped_mask] = 0
            record_gradient_evaluations += len(records)
            non_dummy_record_gradient_evaluations += int((~dropped_mask).sum().item())
            contributing_record_gradient_evaluations += int(
                (~dropped_mask).sum().item()
            )
        else:
            release_gradients = torch.zeros(
                (0, _flat_parameter_size(model)), device=training_config.device
            )
            reference_gradients = release_gradients
        generator = torch.Generator(device=training_config.device)
        generator.manual_seed(seeds.noise + round_index)
        standard_normal = torch.randn(
            release_gradients.shape[1],
            generator=generator,
            device=training_config.device,
        )
        execution = execute_plan_torch(
            records,
            release_gradients,
            plan,
            public_config,
            standard_normal,
            reference_record_gradients=reference_gradients,
        )
        apply_flat_update(model, execution.released, training_config.learning_rate)
        clip_fractions.append(execution.clip_fraction)
        utilizations.append(execution.utilization)
        group_evaluations += execution.active_units
        aggregate_snrs.append(execution.aggregate_snr)
        gradient_distortions.append(execution.gradient_distortion_l2)
        partial_gradient_distortions.append(execution.partial_gradient_distortion_l2)
        residual_gradient_distortions.append(execution.residual_gradient_distortion_l2)
        partial_clip_fractions.append(execution.partial_clip_fraction)
        residual_clip_fractions.append(execution.residual_clip_fraction)
        residual_norm_shares.append(execution.residual_norm_share)
        residual_active_unit_events += execution.active_residual_units
        residual_saturated_unit_events += execution.saturated_residual_units
        residual_influence.append(
            ResidualInfluencePoint(
                round=round_index + 1,
                residual_norm_share=execution.residual_norm_share,
                residual_clip_saturation_rate=execution.residual_clip_saturation_rate,
                active_residual_units=execution.active_residual_units,
                saturated_residual_units=execution.saturated_residual_units,
            )
        )
        active_records_total += len(records)
        partial_records_total += sum(
            len(unit.record_ids)
            for unit in plan.units
            if unit.kind == UnitKind.PARTIAL_HANDLE
        )
        residual_records_total += sum(
            len(unit.record_ids)
            for unit in plan.units
            if unit.kind in (UnitKind.LOCAL_ATOM, UnitKind.LOCAL_MIXED)
        )
        completed_rounds = round_index + 1
        if completed_rounds in checkpoint_rounds:
            evaluate_checkpoint(completed_rounds)
    (
        sample_accuracy,
        user_macro_accuracy,
        auprc,
        clusters,
        positive_prediction_fraction,
        prediction_entropy_nats,
        class_zero_recall,
        class_one_recall,
        majority_class_accuracy,
    ) = _evaluate(
        model, task, evaluation_indices, training_config.device, evaluation_groups
    )
    residual_only_clusters = tuple(
        cluster for cluster in clusters if cluster.evidence_group == "no-handle"
    )
    residual_only_subgroup_metric = None
    if residual_only_clusters:
        if task.n_classes == 2 and task.stable_user_label:
            from sklearn.metrics import average_precision_score

            residual_labels = [
                cluster.binary_label for cluster in residual_only_clusters
            ]
            residual_probabilities = [
                cluster.positive_probability for cluster in residual_only_clusters
            ]
            if len(set(residual_labels)) > 1:
                residual_only_subgroup_metric = float(
                    average_precision_score(residual_labels, residual_probabilities)
                )
        else:
            residual_only_subgroup_metric = float(
                np.mean([cluster.accuracy for cluster in residual_only_clusters])
            )
    return TrainingOutput(
        result=TrainingResult(
            method=method.value,
            sample_accuracy=sample_accuracy,
            user_macro_accuracy=user_macro_accuracy,
            patient_or_user_auprc=auprc,
            clip_fraction=float(np.mean(clip_fractions)),
            effective_record_utilization=float(np.mean(utilizations)),
            partial_handle_record_fraction=partial_records_total / active_records_total
            if active_records_total
            else 0.0,
            residual_record_fraction=residual_records_total / active_records_total
            if active_records_total
            else 0.0,
            aggregate_snr=None
            if method == MethodId.NON_PRIVATE
            else float(np.mean(aggregate_snrs)),
            gradient_distortion_l2=float(np.mean(gradient_distortions)),
            partial_gradient_distortion_l2=float(np.mean(partial_gradient_distortions)),
            residual_gradient_distortion_l2=float(
                np.mean(residual_gradient_distortions)
            ),
            partial_clip_fraction=float(np.mean(partial_clip_fractions)),
            residual_clip_fraction=float(np.mean(residual_clip_fractions)),
            residual_norm_share=float(np.mean(residual_norm_shares)),
            residual_clip_saturation_rate=(
                residual_saturated_unit_events / residual_active_unit_events
                if residual_active_unit_events
                else None
            ),
            residual_only_subgroup_metric=residual_only_subgroup_metric,
            residual_only_subgroup_people=len(residual_only_clusters),
            residual_only_subgroup_records=sum(
                cluster.records for cluster in residual_only_clusters
            ),
            record_gradient_evaluations=record_gradient_evaluations,
            non_dummy_record_gradient_evaluations=non_dummy_record_gradient_evaluations,
            contributing_record_gradient_evaluations=contributing_record_gradient_evaluations,
            group_gradient_evaluations=group_evaluations,
            noisy_releases=0
            if method == MethodId.NON_PRIVATE
            else training_config.rounds,
            wall_time_seconds=time.perf_counter()
            - started
            - oracle_audit_seconds
            - diagnostic_gradient_seconds,
            oracle_audit_seconds=oracle_audit_seconds,
            diagnostic_gradient_seconds=diagnostic_gradient_seconds,
            diagnostic_record_gradient_evaluations=diagnostic_record_gradient_evaluations,
            positive_prediction_fraction=positive_prediction_fraction,
            prediction_entropy_nats=prediction_entropy_nats,
            class_zero_recall=class_zero_recall,
            class_one_recall=class_one_recall,
            majority_class_accuracy=majority_class_accuracy,
        ),
        clusters=clusters,
        domain_plan=domain_summary,
        trajectory=tuple(trajectory),
        residual_influence=tuple(residual_influence),
    )
