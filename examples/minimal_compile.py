from __future__ import annotations

# Imports follow the repository-root path setup when this file runs as a script.
# ruff: noqa: E402

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.unitscope.audit import full_recompile_audit
from experiments.unitscope.compiler import (
    compile_plan,
    validate_plan_with_oracle,
    with_oracle_validation,
)
from experiments.unitscope.evidence import EvidencePolicy, generate_evidence_view
from experiments.unitscope.model import GroundTruth, MethodId, PublicConfig
from experiments.unitscope.synthetic import make_vector_population


def main() -> None:
    config = PublicConfig(
        task_domain="minimal-example",
        epoch="frozen",
        clip_norm=1.0,
        max_handle_groups=1,
        max_fallback_slots=3,
        lambda_=0.5,
        public_normalizer=6.0,
        noise_multiplier=1.0,
        rounds=1,
        delta=1e-6,
        n_silos=3,
    )
    population = make_vector_population(
        n_users=6,
        n_silos=3,
        dimension=4,
        overlap_probability=1.0,
        gradient_correlation=0.5,
        norm=0.6,
        seed=11,
    )
    generated = generate_evidence_view(
        population.records,
        population.truth,
        config,
        EvidencePolicy(1.0, 0.5, handles_per_user=1),
        seed=12,
    )
    plan = compile_plan(
        MethodId.UNITSCOPE_TUNED,
        population.records,
        generated.evidence,
        generated.registry,
        config,
    )
    validation = validate_plan_with_oracle(
        plan, population.records, population.truth, config
    )
    if not validation.valid:
        raise RuntimeError(validation.errors)
    plan = with_oracle_validation(plan, validation)

    truth = population.truth.as_dict()
    removed_user = sorted(set(truth.values()))[0]
    keep = np.asarray(
        [truth[record.record_id] != removed_user for record in population.records],
        dtype=bool,
    )
    new_records = tuple(
        record for record, retained in zip(population.records, keep) if retained
    )
    new_truth = GroundTruth(
        tuple((record.record_id, truth[record.record_id]) for record in new_records)
    )
    audit = full_recompile_audit(
        MethodId.UNITSCOPE_TUNED,
        population.records,
        new_records,
        generated.evidence,
        generated.registry,
        config,
        population.vectors,
        population.vectors[keep],
    )
    output = {
        "validation": "PASS",
        "units": [
            {
                "label": unit.label,
                "kind": unit.kind.value,
                "records": list(unit.record_ids),
                "radius": unit.radius,
            }
            for unit in plan.units
        ],
        "certified_sensitivity": plan.certified_sensitivity.value,
        "removed_user": removed_user,
        "observed_wur": audit.observed.value,
        "actual_query_difference": audit.actual_query_difference,
        "new_truth_users": len(set(new_truth.as_dict().values())),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
