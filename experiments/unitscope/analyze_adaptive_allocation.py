from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(values: list[float], seed: int) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), size=(20_000, len(data)))].mean(axis=1)
    return float(data.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the adaptive allocation confirmation")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.runs.resolve()
    out = args.output.resolve()
    if not (run / "CONFIRMATION_STARTED.json").exists():
        raise RuntimeError("confirmation was not started")
    if out.exists():
        raise RuntimeError(f"analysis output already exists: {out}")
    out.mkdir(parents=True)
    all_rows: list[dict[str, object]] = []
    for pair in sorted(run.glob("q*_r*_rep??")):
        parts = pair.name.split("_")
        q_user = int(parts[0][1:]) / 100
        q_record = int(parts[1][1:]) / 100
        rep = int(parts[2][3:])
        for route in ("adaptive", "baselines"):
            path = pair / route / "learning_results.jsonl"
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for row in rows:
                if route == "adaptive":
                    method = "adaptive-v2"
                elif row["method"] == "unitscope-lambda-sweep":
                    method = "fixed-0.9"
                elif row["method"] == "stable-local-fallback":
                    method = "stable-local"
                elif row["method"] == "handle-only-drop-residual":
                    method = "handle-only"
                else:
                    raise RuntimeError(f"unexpected method {row['method']}")
                all_rows.append(
                    {
                        "q_user": q_user,
                        "q_record": q_record,
                        "rep": rep,
                        "seed": row["seed"],
                        "method": method,
                        "lambda": row["lambda"],
                        "epsilon_learning": row["epsilon"],
                        "metric_value": row["metric_value"],
                        "run_id": row["run_id"],
                        "config_hash": row["config_hash"],
                        "validation_pass": row["validation_pass"],
                    }
                )
    if len(all_rows) != 360:
        raise RuntimeError(f"expected 360 result rows, found {len(all_rows)}")
    if not all(bool(row["validation_pass"]) for row in all_rows):
        raise RuntimeError("one or more compiler validations failed")
    key_counts: dict[tuple[float, float, int], set[str]] = {}
    for row in all_rows:
        key = (float(row["q_user"]), float(row["q_record"]), int(row["rep"]))
        key_counts.setdefault(key, set()).add(str(row["method"]))
    expected = {"adaptive-v2", "fixed-0.9", "stable-local", "handle-only"}
    if len(key_counts) != 90 or any(methods != expected for methods in key_counts.values()):
        raise RuntimeError("paired method coverage is incomplete")

    all_path = out / "all_results.csv"
    with all_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    cells = []
    cell_keys = sorted({(float(row["q_user"]), float(row["q_record"])) for row in all_rows})
    for index, (q_user, q_record) in enumerate(cell_keys):
        rows = [row for row in all_rows if row["q_user"] == q_user and row["q_record"] == q_record]
        by_rep: dict[int, dict[str, float]] = {}
        for row in rows:
            by_rep.setdefault(int(row["rep"]), {})[str(row["method"])] = float(row["metric_value"])
        adaptive_gain = [values["adaptive-v2"] - values["stable-local"] for values in by_rep.values()]
        fixed_gain = [values["fixed-0.9"] - values["stable-local"] for values in by_rep.values()]
        adaptive_handle = [values["adaptive-v2"] - values["handle-only"] for values in by_rep.values()]
        ag = bootstrap(adaptive_gain, 2026072600 + index * 10)
        fg = bootstrap(fixed_gain, 2026072601 + index * 10)
        ah = bootstrap(adaptive_handle, 2026072602 + index * 10)
        retention = ag[0] / fg[0] if fg[0] != 0 else None
        cells.append(
            {
                "q_user": q_user,
                "q_record": q_record,
                "pairs": len(by_rep),
                "selected_lambda": sorted(
                    {float(row["lambda"]) for row in rows if row["method"] == "adaptive-v2"}
                ),
                "adaptive_mean": float(np.mean([values["adaptive-v2"] for values in by_rep.values()])),
                "fixed_mean": float(np.mean([values["fixed-0.9"] for values in by_rep.values()])),
                "stable_mean": float(np.mean([values["stable-local"] for values in by_rep.values()])),
                "handle_mean": float(np.mean([values["handle-only"] for values in by_rep.values()])),
                "adaptive_gain_over_stable": ag[0],
                "adaptive_gain_ci_lower": ag[1],
                "adaptive_gain_ci_upper": ag[2],
                "significantly_negative": ag[2] < 0,
                "fixed_gain_over_stable": fg[0],
                "fixed_gain_ci_lower": fg[1],
                "fixed_gain_ci_upper": fg[2],
                "gain_retention": retention,
                "retention_ge_90pct": ag[0] >= 0.9 * fg[0],
                "adaptive_minus_handle": ah[0],
                "adaptive_handle_ci_lower": ah[1],
                "adaptive_handle_ci_upper": ah[2],
                "high_coverage_cell": q_user == 0.75,
            }
        )
    cell_path = out / "cell_results.csv"
    with cell_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    high = [row for row in cells if row["high_coverage_cell"]]
    high_path = out / "high_coverage_results.csv"
    with high_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(high)

    gate_rows = list(csv.DictReader((run / "gate_records/gate_releases.csv").open(encoding="utf-8")))
    no_negative = all(not bool(row["significantly_negative"]) for row in cells)
    high_retention = all(bool(row["retention_ge_90pct"]) for row in high)
    manifest = {
        "profile": "method-v2-direction2v2-recalibrated-confirmation",
        "result_rows": len(all_rows),
        "paired_configurations": len(key_counts),
        "zero_discarded": True,
        "all_validation_pass": True,
        "gate_release_rows": len(gate_rows),
        "gate_misclassifications": sum(row["misclassified"].lower() == "true" for row in gate_rows),
        "criterion_no_significantly_negative_cells": no_negative,
        "high_coverage_definition": "q_user=0.75 (three predeclared cells)",
        "criterion_high_coverage_retention_ge_90pct": high_retention,
        "overall_pass": no_negative and high_retention,
        "wording_selected": "success" if no_negative and high_retention else "failure",
        "registry_sha256": sha256(run / "confirmation_registry.json"),
        "all_results_sha256": sha256(all_path),
        "cell_results_sha256": sha256(cell_path),
        "cells": cells,
    }
    (out / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
