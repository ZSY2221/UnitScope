"""Prepare the common six-silo FEBRL inputs."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

import pandas as pd
import recordlinkage
from recordlinkage.datasets import load_febrl1, load_febrl2, load_febrl3


LOADERS = {
    "febrl1": load_febrl1,
    "febrl2": load_febrl2,
    "febrl3": load_febrl3,
}


def stable_int(text: str, domain: str) -> int:
    payload = f"{domain}|{text}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def entity_id(record_id: str) -> int:
    match = re.match(r"rec-(\d+)-", str(record_id))
    if match is None:
        raise ValueError(f"unexpected FEBRL record id: {record_id}")
    return int(match.group(1))


def assign_silos(frame: pd.DataFrame, n_silos: int) -> pd.Series:
    assignments: dict[str, int] = {}
    for entity, members in frame.groupby("entity_id").groups.items():
        ordered = sorted(
            members, key=lambda value: stable_int(str(value), "silo-order")
        )
        if len(ordered) > n_silos:
            raise ValueError(
                f"entity {entity} has multiplicity {len(ordered)}, above n_silos={n_silos}"
            )
        offset = stable_int(str(entity), "silo-offset") % n_silos
        for position, record_id in enumerate(ordered):
            assignments[str(record_id)] = (offset + position) % n_silos
    return pd.Series(assignments, name="silo").reindex(frame.index).astype(int)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize(dataset: str, output_root: Path, n_silos: int) -> dict:
    frame, _ = LOADERS[dataset](return_links=True)
    frame = frame.copy()
    frame.index = frame.index.map(str)
    frame["entity_id"] = [entity_id(record_id) for record_id in frame.index]
    frame["silo"] = assign_silos(frame, n_silos)

    dataset_dir = output_root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    records_path = dataset_dir / "records.csv"
    truth_path = dataset_dir / "truth_pairs.csv"

    records = frame.reset_index(names="id")
    records.to_csv(records_path, index=False, lineterminator="\n")

    truth_rows: list[dict[str, object]] = []
    for entity, members in frame.groupby("entity_id"):
        ids = sorted(map(str, members.index))
        for left, right in itertools.combinations(ids, 2):
            left_silo = int(frame.at[left, "silo"])
            right_silo = int(frame.at[right, "silo"])
            if left_silo == right_silo:
                raise AssertionError(
                    "deterministic assignment put duplicate records in one silo"
                )
            truth_rows.append(
                {
                    "left": left,
                    "right": right,
                    "entity_id": int(entity),
                    "left_silo": left_silo,
                    "right_silo": right_silo,
                }
            )
    truth = pd.DataFrame(
        truth_rows,
        columns=("left", "right", "entity_id", "left_silo", "right_silo"),
    )
    truth.to_csv(truth_path, index=False, lineterminator="\n")

    multiplicity = frame.groupby("entity_id").size()
    return {
        "dataset": dataset,
        "records": int(len(frame)),
        "entities": int(frame["entity_id"].nunique()),
        "truth_pairs": int(len(truth)),
        "multi_entities": int((multiplicity > 1).sum()),
        "maximum_multiplicity": int(multiplicity.max()),
        "n_silos": n_silos,
        "records_sha256": sha256(records_path),
        "truth_pairs_sha256": sha256(truth_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-silos", type=int, default=6)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(LOADERS),
        default=list(LOADERS),
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    summaries = [materialize(name, args.output, args.n_silos) for name in args.datasets]
    manifest = {
        "artifact_role": "UnitScope input adapter for official ER baselines",
        "official_baseline_code_modified": False,
        "source": "recordlinkage.datasets.load_febrl1/load_febrl2/load_febrl3",
        "recordlinkage_version": recordlinkage.__version__,
        "silo_assignment": "same SHA-256 rule as experiments/stableunit_pilot.py",
        "datasets": summaries,
    }
    (args.output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
