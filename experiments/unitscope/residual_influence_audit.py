from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .manifest import atomic_write_json


BOOTSTRAP_SEED = 20270723
BOOTSTRAP_DRAWS = 20_000
ENDPOINT_TOLERANCES = {
    "femnist": 0.0003125,
    "synthea": 0.0011716461628588166,
}
METHODS = (
    "stable-local-fallback",
    "handle-only-drop-residual",
    "unitscope-lambda-sweep",
    "naive-recalibrated-2c",
)
DATASETS = ("femnist", "synthea")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_mean_interval(
    values: np.ndarray, draws: np.ndarray
) -> tuple[float, float, float]:
    sampled = values[draws].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(sampled, 0.025)),
        float(np.quantile(sampled, 0.975)),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_outputs(root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    runs: list[dict] = []
    rounds: list[dict] = []
    subgroup_rows: list[dict] = []
    for dataset in DATASETS:
        for replicate in range(20):
            directory = root / f"{dataset}_r{replicate:02d}"
            parent_directory = (
                root.parent
                / "fixed_allocation_replication"
                / f"{dataset}_r{replicate:02d}"
            )
            current_manifest = json.loads(
                (directory / "learning_manifest.json").read_text(encoding="utf-8")
            )["frozen_configuration"]
            parent_manifest = json.loads(
                (parent_directory / "learning_manifest.json").read_text(
                    encoding="utf-8"
                )
            )["frozen_configuration"]

            def project_like(current: object, parent: object) -> object:
                if isinstance(parent, dict):
                    if not isinstance(current, dict):
                        raise RuntimeError("manifest type mismatch")
                    return {
                        key: project_like(current[key], value)
                        for key, value in parent.items()
                    }
                if isinstance(parent, list):
                    if not isinstance(current, list) or len(current) != len(parent):
                        raise RuntimeError("manifest list mismatch")
                    return [
                        project_like(item, template)
                        for item, template in zip(current, parent)
                    ]
                return current

            if project_like(current_manifest, parent_manifest) != parent_manifest:
                raise RuntimeError(
                    f"execution protocol differs from the frozen parent for {dataset} r{replicate:02d}"
                )
            run_files = sorted((directory / "runs").glob("*.json"))
            if len(run_files) != 4:
                raise RuntimeError(
                    f"{directory} has {len(run_files)} run rows, expected four"
                )
            cluster_membership: dict[str, tuple[str, ...]] = {}
            for run_file in run_files:
                row = json.loads(run_file.read_text(encoding="utf-8"))
                row["replicate"] = replicate
                runs.append(row)
                run_id = row["run_id"]
                influence_file = directory / "residual_influence" / f"{run_id}.json"
                influence = json.loads(influence_file.read_text(encoding="utf-8"))[
                    "residual_influence"
                ]
                if len(influence) != 50:
                    raise RuntimeError(
                        f"{influence_file} has {len(influence)} rounds, expected 50"
                    )
                for item in influence:
                    rounds.append(
                        {
                            "dataset": dataset,
                            "replicate": replicate,
                            "seed": row["seed"],
                            "method": row["method"],
                            **item,
                        }
                    )
                cluster_file = directory / "clusters" / f"{run_id}.json"
                clusters = json.loads(cluster_file.read_text(encoding="utf-8"))[
                    "clusters"
                ]
                residual_clusters = tuple(
                    sorted(
                        cluster["cluster_id"]
                        for cluster in clusters
                        if cluster["evidence_group"] == "no-handle"
                    )
                )
                cluster_membership[row["method"]] = residual_clusters
                subgroup_rows.append(
                    {
                        "dataset": dataset,
                        "replicate": replicate,
                        "seed": row["seed"],
                        "method": row["method"],
                        "metric_name": "writer_macro_accuracy"
                        if dataset == "femnist"
                        else "patient_auprc",
                        "metric_value": row["residual_only_subgroup_metric"],
                        "people": row["residual_only_subgroup_people"],
                        "records": row["residual_only_subgroup_records"],
                    }
                )
            memberships = list(cluster_membership.values())
            if not memberships or any(
                value != memberships[0] for value in memberships[1:]
            ):
                raise RuntimeError(
                    f"residual-only subgroup differs across methods for {dataset} r{replicate:02d}"
                )
    if len(runs) != 160 or len(rounds) != 8000 or len(subgroup_rows) != 160:
        raise RuntimeError(
            f"completeness gate failed: runs={len(runs)}, rounds={len(rounds)}, subgroup={len(subgroup_rows)}"
        )
    return runs, rounds, subgroup_rows


def analyze(
    root: Path,
    endpoint_table: Path,
    endpoint_tolerances: Mapping[str, float] | None = None,
    analysis_name: str = "residual-influence-audit",
) -> None:
    runs, rounds, subgroup_rows = _load_outputs(root)
    frozen = _read_csv(endpoint_table)
    frozen_index = {
        (row["dataset"], row["method"]): float(row["metric_mean"]) for row in frozen
    }
    endpoint_gate: list[dict] = []
    tolerances = endpoint_tolerances or ENDPOINT_TOLERANCES
    for dataset in DATASETS:
        for method in METHODS:
            selected = [
                row
                for row in runs
                if row["dataset"] == dataset and row["method"] == method
            ]
            observed = float(np.mean([float(row["metric_value"]) for row in selected]))
            expected = frozen_index[(dataset, method)]
            difference = observed - expected
            tolerance = float(tolerances[dataset])
            endpoint_gate.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "frozen_mean": expected,
                    "rerun_mean": observed,
                    "difference": difference,
                    "absolute_tolerance": tolerance,
                    "pass": abs(difference) <= tolerance,
                }
            )
    endpoint_gate_pass = all(row["pass"] for row in endpoint_gate)
    if not endpoint_gate_pass:
        _write_csv(
            root / "analysis" / "endpoint_gate.csv",
            endpoint_gate,
            tuple(endpoint_gate[0]),
        )
        raise RuntimeError(
            "endpoint validity gate failed; see analysis/endpoint_gate.csv"
        )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, 20, size=(BOOTSTRAP_DRAWS, 20))
    method_summary: list[dict] = []
    run_index = {
        (row["dataset"], row["method"], int(row["replicate"])): row for row in runs
    }
    subgroup_index = {
        (row["dataset"], row["method"], int(row["replicate"])): row
        for row in subgroup_rows
    }
    for dataset in DATASETS:
        for method in METHODS:
            norm = np.asarray(
                [
                    float(
                        run_index[(dataset, method, replicate)]["residual_norm_share"]
                    )
                    for replicate in range(20)
                ]
            )
            subgroup = np.asarray(
                [
                    float(subgroup_index[(dataset, method, replicate)]["metric_value"])
                    for replicate in range(20)
                ]
            )
            saturation_values = [
                run_index[(dataset, method, replicate)]["residual_clip_saturation_rate"]
                for replicate in range(20)
            ]
            norm_mean, norm_low, norm_high = _bootstrap_mean_interval(norm, draws)
            subgroup_mean, subgroup_low, subgroup_high = _bootstrap_mean_interval(
                subgroup, draws
            )
            if all(value is None for value in saturation_values):
                saturation_mean = saturation_low = saturation_high = None
            elif any(value is None for value in saturation_values):
                raise RuntimeError(
                    f"partially undefined saturation values for {dataset}, {method}"
                )
            else:
                saturation = np.asarray([float(value) for value in saturation_values])
                saturation_mean, saturation_low, saturation_high = (
                    _bootstrap_mean_interval(saturation, draws)
                )
            people = [
                int(subgroup_index[(dataset, method, replicate)]["people"])
                for replicate in range(20)
            ]
            records = [
                int(subgroup_index[(dataset, method, replicate)]["records"])
                for replicate in range(20)
            ]
            method_summary.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "replicates": 20,
                    "residual_norm_share_mean": norm_mean,
                    "residual_norm_share_ci_lower": norm_low,
                    "residual_norm_share_ci_upper": norm_high,
                    "residual_clip_saturation_mean": saturation_mean,
                    "residual_clip_saturation_ci_lower": saturation_low,
                    "residual_clip_saturation_ci_upper": saturation_high,
                    "residual_only_metric": "writer_macro_accuracy"
                    if dataset == "femnist"
                    else "patient_auprc",
                    "residual_only_metric_mean": subgroup_mean,
                    "residual_only_metric_ci_lower": subgroup_low,
                    "residual_only_metric_ci_upper": subgroup_high,
                    "residual_only_people_min": min(people),
                    "residual_only_people_max": max(people),
                    "residual_only_people_mean": float(np.mean(people)),
                    "residual_only_records_min": min(records),
                    "residual_only_records_max": max(records),
                    "residual_only_records_mean": float(np.mean(records)),
                }
            )

    paired_differences: list[dict] = []
    for dataset in DATASETS:
        differences = np.asarray(
            [
                float(
                    subgroup_index[(dataset, "unitscope-lambda-sweep", replicate)][
                        "metric_value"
                    ]
                )
                - float(
                    subgroup_index[(dataset, "handle-only-drop-residual", replicate)][
                        "metric_value"
                    ]
                )
                for replicate in range(20)
            ]
        )
        mean, low, high = _bootstrap_mean_interval(differences, draws)
        paired_differences.append(
            {
                "dataset": dataset,
                "contrast": "unitscope-lambda-sweep minus handle-only-drop-residual",
                "mean_difference": mean,
                "ci_lower": low,
                "ci_upper": high,
                "zero_margin_hypothesis_supported": mean >= 0,
                "replicates": 20,
            }
        )

    analysis = root / "analysis"
    run_columns = (
        "dataset",
        "replicate",
        "seed",
        "method",
        "metric_name",
        "metric_value",
        "residual_norm_share",
        "residual_clip_saturation_rate",
        "residual_only_subgroup_metric",
        "residual_only_subgroup_people",
        "residual_only_subgroup_records",
        "effective_record_utilization",
        "partial_handle_record_fraction",
        "residual_record_fraction",
        "residual_clip_fraction",
        "clip_fraction",
        "noise_multiplier",
        "epsilon",
        "delta",
        "rounds",
        "run_id",
        "config_hash",
        "manifest_hash",
    )
    _write_csv(analysis / "run_metrics.csv", runs, run_columns)
    _write_csv(analysis / "round_metrics.csv", rounds, tuple(rounds[0]))
    _write_csv(
        analysis / "subgroup_metrics.csv", subgroup_rows, tuple(subgroup_rows[0])
    )
    _write_csv(analysis / "endpoint_gate.csv", endpoint_gate, tuple(endpoint_gate[0]))
    _write_csv(
        analysis / "method_summary.csv", method_summary, tuple(method_summary[0])
    )
    _write_csv(
        analysis / "paired_differences.csv",
        paired_differences,
        tuple(paired_differences[0]),
    )

    output_files = sorted(path for path in analysis.iterdir() if path.is_file())
    output_hashes = {path.name: _sha256(path) for path in output_files}
    analysis_hash = hashlib.sha256(
        "".join(
            f"{name}\0{digest}\n" for name, digest in sorted(output_hashes.items())
        ).encode("utf-8")
    ).hexdigest()
    atomic_write_json(
        analysis / "analysis_manifest.json",
        {
            "analysis": analysis_name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "endpoint_tolerance_by_dataset": tolerances,
            "method_runs": len(runs),
            "round_rows": len(rounds),
            "subgroup_rows": len(subgroup_rows),
            "zero_discarded": True,
            "endpoint_gate_pass": endpoint_gate_pass,
            "output_hashes": output_hashes,
            "analysis_hash": analysis_hash,
        },
    )
    print(
        json.dumps(
            {"endpoint_gate_pass": endpoint_gate_pass, "analysis_hash": analysis_hash},
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the UnitScope residual-influence experiment"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--endpoint-table", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(args.root, args.endpoint_table)


if __name__ == "__main__":
    main()
