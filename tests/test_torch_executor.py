from __future__ import annotations

# Executor imports follow the optional PyTorch availability guard.
# ruff: noqa: E402

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.unitscope.compiler import (
    compile_plan,
    validate_plan_with_oracle,
    with_oracle_validation,
)
from experiments.unitscope.evidence import EvidencePolicy, generate_evidence_view
from experiments.unitscope.executor import execute_release
from experiments.unitscope.model import (
    GroundTruth,
    MethodId,
    PublicConfig,
    PublicRecord,
)
from experiments.unitscope.torch_training import execute_plan_torch


def _fixture():
    config = PublicConfig(
        task_domain="torch-test",
        epoch="frozen",
        clip_norm=1.0,
        max_handle_groups=1,
        max_fallback_slots=2,
        lambda_=0.4,
        public_normalizer=4.0,
        noise_multiplier=1.2,
        rounds=2,
        delta=1e-5,
        n_silos=2,
    )
    records = (
        PublicRecord("r1", "s1", "slot1"),
        PublicRecord("r2", "s2", "slot2"),
    )
    truth = GroundTruth((("r1", "u"), ("r2", "u")))
    generated = generate_evidence_view(
        records, truth, config, EvidencePolicy(1.0, 0.5), 9
    )
    plan = compile_plan(
        MethodId.UNITSCOPE_TUNED,
        records,
        generated.evidence,
        generated.registry,
        config,
    )
    validation = validate_plan_with_oracle(plan, records, truth, config)
    assert validation.valid
    return config, records, with_oracle_validation(plan, validation)


def test_torch_executor_matches_numpy_reference() -> None:
    config, records, plan = _fixture()
    gradients = np.asarray([[0.5, -0.25, 0.1], [0.2, 0.3, -0.4]], dtype=np.float32)
    noise = np.asarray([0.4, -0.2, 0.3], dtype=np.float32)
    numpy_release = execute_release(records, gradients, plan, config, noise)
    torch_release = execute_plan_torch(
        records,
        torch.as_tensor(gradients),
        plan,
        config,
        torch.as_tensor(noise),
    )
    np.testing.assert_allclose(
        torch_release.released.numpy(), numpy_release.value, rtol=1e-6, atol=1e-7
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_and_cpu_plan_execution_agree() -> None:
    config, records, plan = _fixture()
    gradients = torch.tensor([[0.5, -0.25, 0.1], [0.2, 0.3, -0.4]], dtype=torch.float32)
    noise = torch.tensor([0.4, -0.2, 0.3], dtype=torch.float32)
    cpu = execute_plan_torch(records, gradients, plan, config, noise)
    cuda = execute_plan_torch(records, gradients.cuda(), plan, config, noise.cuda())
    torch.testing.assert_close(cuda.released.cpu(), cpu.released, rtol=1e-5, atol=1e-6)
