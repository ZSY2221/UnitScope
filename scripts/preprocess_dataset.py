from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.unitscope.datasets.movielens import (
    balance_movielens_partition,
    load_and_partition_movielens1m,
)
from experiments.unitscope.datasets.sent140 import load_and_partition_sent140
from experiments.unitscope.datasets.synthea import (
    build_synthea_future_inpatient_task,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a frozen UnitScope dataset transform"
    )
    parser.add_argument("dataset", choices=("synthea", "sent140", "movielens"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset == "synthea":
        task = build_synthea_future_inpatient_task(
            args.input,
            index_date="2020-01-01",
            lookback_days=365,
            horizon_days=365,
            n_silos=6,
            seed=20260714,
        )
        summary = {
            "records": len(task.records),
            "people": len(set(task.truth.as_dict().values())),
        }
    elif args.dataset == "sent140":
        part = load_and_partition_sent140(
            args.input,
            n_silos=6,
            train_overlap_probability=0.5,
            split_seed=51001,
            topology_seed=61001,
            maximum_accounts=2000,
            maximum_records_per_account=32,
            account_selection_seed=31002,
            text_hash_dimension=512,
        )
        summary = {
            "records": len(part.records),
            "accounts": len(set(part.truth.as_dict().values())),
            "ambiguous_duplicate_rows_removed": part.ambiguous_duplicate_rows_removed,
        }
    else:
        part = load_and_partition_movielens1m(
            args.input,
            n_silos=6,
            train_overlap_probability=0.5,
            split_seed=51001,
            topology_seed=61001,
            maximum_accounts=3000,
            maximum_records_per_account=32,
            account_selection_seed=31002,
        )
        part = balance_movielens_partition(part, balance_seed=20260714)
        summary = {
            "records": len(part.records),
            "accounts": len(set(part.truth.as_dict().values())),
            "task_variant": part.task_variant,
            "majority_keep_probability": part.balance_majority_keep_probability,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": args.dataset, "input": args.input.name, "summary": summary}
    if args.input.is_file():
        payload["input_sha256"] = file_sha256(args.input)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
