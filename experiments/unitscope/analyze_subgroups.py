from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _group_metric(rows: list[dict[str, object]]) -> tuple[str, float | None]:
    labels = [row.get("binary_label") for row in rows]
    probabilities = [row.get("positive_probability") for row in rows]
    if rows and all(value is not None for value in labels + probabilities):
        numeric_labels = np.asarray(labels, dtype=np.int64)
        if len(np.unique(numeric_labels)) < 2:
            return "patient_auprc", None
        return "patient_auprc", float(
            average_precision_score(
                numeric_labels, np.asarray(probabilities, dtype=np.float64)
            )
        )
    return "writer_macro_accuracy", float(
        np.mean([float(row["accuracy"]) for row in rows])
    )


def load_runs(cluster_directories: list[Path]) -> pd.DataFrame:
    output = []
    for directory in cluster_directories:
        run_set = directory.parent.name
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            run_metadata_path = directory.parent / "runs" / f"{payload['run_id']}.json"
            run_metadata = (
                json.loads(run_metadata_path.read_text(encoding="utf-8"))
                if run_metadata_path.exists()
                else {}
            )
            lambda_value = run_metadata.get("lambda")
            method_variant = str(payload["method"])
            if lambda_value is not None:
                method_variant = f"{method_variant}[lambda={float(lambda_value):g}]"
            clusters = payload.get("clusters", [])
            grouped: dict[str, list[dict[str, object]]] = {"all-users": list(clusters)}
            for row in clusters:
                group = str(row.get("evidence_group") or "unlabeled")
                grouped.setdefault(group, []).append(row)
            for group, rows in grouped.items():
                metric_name, metric_value = _group_metric(rows)
                output.append(
                    {
                        "run_set": run_set,
                        "dataset": payload["dataset"],
                        "seed": int(payload["seed"]),
                        "method": payload["method"],
                        "method_variant": method_variant,
                        "lambda": lambda_value,
                        "evidence_group": group,
                        "users": len(rows),
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                    }
                )
    return pd.DataFrame(output)


def paired_effects(frame: pd.DataFrame) -> pd.DataFrame:
    output = []
    for keys, group in frame.groupby(
        ["dataset", "evidence_group", "metric_name"], dropna=False
    ):
        pivot = group.pivot_table(
            index="seed",
            columns="method_variant",
            values="metric_value",
            aggfunc="first",
        )
        unit_methods = [
            column for column in pivot if str(column).startswith("unitscope-")
        ]
        for unit_method in unit_methods:
            for baseline in ("stable-local-fallback", "handle-only-drop-residual"):
                baseline_columns = [
                    column for column in pivot if str(column).startswith(f"{baseline}[")
                ]
                if baseline in pivot:
                    baseline_column = baseline
                elif len(baseline_columns) == 1:
                    baseline_column = baseline_columns[0]
                else:
                    continue
                paired = pivot[[unit_method, baseline_column]].dropna()
                if paired.empty:
                    continue
                differences = paired[unit_method] - paired[baseline_column]
                output.append(
                    {
                        "dataset": keys[0],
                        "evidence_group": keys[1],
                        "metric_name": keys[2],
                        "method": unit_method,
                        "baseline": baseline_column,
                        "paired_seeds": len(differences),
                        "mean_difference": float(differences.mean()),
                        "positive_seed_fraction": float(np.mean(differences > 0)),
                    }
                )
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze UnitScope evidence-coverage subgroups"
    )
    parser.add_argument("--cluster-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = load_runs(args.cluster_dirs)
    if frame.empty:
        raise ValueError("no cluster result files were found")
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "subgroup_metrics.csv", index=False)
    effects = paired_effects(frame)
    effects.to_csv(args.output / "subgroup_paired_effects.csv", index=False)
    summary = (
        frame.groupby(
            ["run_set", "dataset", "method", "evidence_group", "metric_name"],
            dropna=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            users_mean=("users", "mean"),
            metric_mean=("metric_value", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(args.output / "subgroup_summary.csv", index=False)
    print(json.dumps({"rows": len(frame), "effects": len(effects)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
