from __future__ import annotations

import pandas as pd

from experiments.unitscope.analyze_learning import analyze
from experiments.unitscope.model import MethodId


def test_two_dataset_confirmatory_analysis_forms_exactly_four_holm_tests() -> None:
    rows = []
    method_values = {
        MethodId.STABLE_LOCAL_FALLBACK.value: 0.60,
        MethodId.HANDLE_ONLY.value: 0.61,
        MethodId.UNITSCOPE_EQUAL.value: 0.63,
        MethodId.UNITSCOPE_TUNED.value: 0.65,
        MethodId.NAIVE_RECALIBRATED_2C.value: 0.62,
        MethodId.FULL_IDENTITY.value: 0.70,
    }
    hashes = {"femnist": "hash-f", "synthea": "hash-s"}
    for dataset, config_hash in hashes.items():
        for seed in range(10):
            for method, value in method_values.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "config_hash": config_hash,
                        "seed": seed,
                        "method": method,
                        "metric_value": value + seed * 1e-5,
                    }
                )
    summary, tests = analyze(
        pd.DataFrame(rows),
        "metric_value",
        cluster_directories=None,
        confirmatory_config_hashes=set(hashes.values()),
    )
    assert len(tests["raw_p_values"]) == 4
    tuned = summary[summary["unit_method"] == MethodId.UNITSCOPE_TUNED.value]
    assert set(tuned["inference_status"]) == {"preregistered-confirmatory"}
