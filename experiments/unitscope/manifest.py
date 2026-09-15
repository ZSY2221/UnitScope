from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from .baselines import MethodSpec
from .model import PublicConfig


ALLOWED_LAMBDA_SOURCES = {
    "public-policy",
    "public-auxiliary",
    "out-of-domain-independent",
}


@dataclass(frozen=True)
class SeedBundle:
    split_seed: int
    topology_seed: int
    feature_seed: int
    evidence_seed: int
    initialization_seed: int
    schedule_seed: int
    noise_seed: int


@dataclass(frozen=True)
class ExperimentManifest:
    experiment_id: str
    scenario: str
    dataset: str
    public_config: PublicConfig
    seeds: Tuple[SeedBundle, ...]
    lambda_selection_source: str
    primary_metric: str
    methods: Tuple[str, ...]
    writer_or_user_disjoint_split: bool
    protected_test_used_for_selection: bool = False

    def validate(self) -> None:
        errors = []
        if not self.experiment_id or not self.scenario or not self.dataset:
            errors.append("experiment_id, scenario, and dataset are required")
        if self.lambda_selection_source not in ALLOWED_LAMBDA_SOURCES:
            errors.append("lambda must be frozen by a public/independent source")
        if self.protected_test_used_for_selection:
            errors.append(
                "the protected test set cannot select lambda or any protocol parameter"
            )
        if not self.writer_or_user_disjoint_split:
            errors.append("formal learning tasks require writer/user-disjoint splits")
        if len(self.seeds) < 1 or len(
            {seed.initialization_seed for seed in self.seeds}
        ) != len(self.seeds):
            errors.append("the manifest needs distinct paired initialization seeds")
        if self.public_config.use_sampling_amplification:
            errors.append(
                "the frozen primary protocol must not use sampling amplification"
            )
        if errors:
            raise ValueError("; ".join(errors))

    @property
    def config_hash(self) -> str:
        self.validate()
        payload = _jsonable(asdict(self))
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


RESULT_REQUIRED_COLUMNS = (
    "run_id",
    "config_hash",
    "dataset",
    "scenario",
    "seed",
    "method",
    "implementation_provenance",
    "privacy_semantics",
    "comparison_tier",
    "lambda",
    "H",
    "B",
    "C",
    "q_user",
    "q_record_given_handle",
    "q_record_given_handle_pooled",
    "q_record_total",
    "handle_users_with_residual_fraction",
    "overlap_probability",
    "realized_overlap_fraction",
    "certified_sensitivity",
    "certificate_type",
    "public_normalizer",
    "noise_multiplier",
    "noisy_releases",
    "effective_record_utilization",
    "partial_handle_record_fraction",
    "residual_record_fraction",
    "clip_fraction",
    "aggregate_snr",
    "gradient_distortion_l2",
    "partial_gradient_distortion_l2",
    "residual_gradient_distortion_l2",
    "partial_clip_fraction",
    "residual_clip_fraction",
    "record_gradient_evaluations",
    "non_dummy_record_gradient_evaluations",
    "contributing_record_gradient_evaluations",
    "metric_name",
    "metric_value",
    "wall_time_seconds",
    "oracle_audit_seconds",
    "diagnostic_gradient_seconds",
    "diagnostic_record_gradient_evaluations",
)


def validate_result_row(row: Mapping[str, Any]) -> None:
    missing = [column for column in RESULT_REQUIRED_COLUMNS if column not in row]
    if missing:
        raise ValueError(f"result row misses required columns: {missing}")
    if row["privacy_semantics"] == "unsafe-diagnostic" and row.get("epsilon") not in (
        None,
        "",
        "N/A",
    ):
        raise ValueError(
            "unsafe diagnostics must not publish a valid user-level epsilon"
        )


def build_method_fields(spec: MethodSpec) -> Dict[str, str]:
    return {
        "method": spec.method_id,
        "implementation_provenance": spec.provenance.value,
        "privacy_semantics": spec.privacy_semantics.value,
        "comparison_tier": spec.comparison_tier.value,
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name, suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def environment_fingerprint() -> Mapping[str, Any]:
    value: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
    }
    packages = {}
    for distribution in (
        "numpy",
        "pandas",
        "scikit-learn",
        "protobuf",
        "torch",
        "transformers",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    value["packages"] = packages
    try:
        import torch

        value["torch_cuda"] = torch.version.cuda
        value["cuda_available"] = bool(torch.cuda.is_available())
        value["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except ImportError:
        value["torch_cuda"] = None
        value["cuda_available"] = False
        value["gpu"] = None
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value
