from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    Callable,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import pandas as pd

from .manifest import atomic_write_json


Pair = Tuple[str, str]


@dataclass(frozen=True)
class LinkageMetrics:
    pair_precision: float
    pair_recall: float
    pair_f1: float
    bcubed_precision: float
    bcubed_recall: float
    bcubed_f1: float
    candidate_recall: float | None = None
    conditional_candidate_precision: float | None = None
    conditional_candidate_recall: float | None = None
    conditional_candidate_f1: float | None = None


@dataclass(frozen=True)
class DeletionWitness:
    entity_id_hash: str
    removed_records: int
    changed_old_components: int
    changed_new_components: int
    witness_wur_over_c: float
    surviving_records_reassigned: int


def canonical_pair(left: str, right: str) -> Pair:
    if left == right:
        raise ValueError("self pairs are not valid linkage candidates")
    return (left, right) if left < right else (right, left)


def components_from_pairs(
    record_ids: Sequence[str], positive_pairs: Iterable[Pair]
) -> Mapping[str, str]:
    parent = {record_id: record_id for record_id in record_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a == b:
            return
        if a > b:
            a, b = b, a
        parent[b] = a

    for left, right in sorted(
        {canonical_pair(left, right) for left, right in positive_pairs}
    ):
        if left not in parent or right not in parent:
            raise ValueError(
                "linkage pair references a record outside the active universe"
            )
        union(left, right)
    members: MutableMapping[str, List[str]] = {}
    for record_id in sorted(record_ids):
        members.setdefault(find(record_id), []).append(record_id)
    labels = {}
    for values in members.values():
        digest = hashlib.sha256("|".join(sorted(values)).encode("utf-8")).hexdigest()[
            :24
        ]
        for record_id in values:
            labels[record_id] = f"component-{digest}"
    return labels


def _pairs_within_groups(
    labels: Mapping[str, str], candidate_pairs: Set[Pair] | None = None
) -> Set[Pair]:
    records_by_group: MutableMapping[str, List[str]] = {}
    for record_id, label in labels.items():
        records_by_group.setdefault(label, []).append(record_id)
    output = set()
    for records in records_by_group.values():
        ordered = sorted(records)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                pair = (left, right)
                if candidate_pairs is None or pair in candidate_pairs:
                    output.add(pair)
    return output


def linkage_metrics(
    predicted_labels: Mapping[str, str],
    true_labels: Mapping[str, str],
    candidate_pairs: Set[Pair] | None = None,
) -> LinkageMetrics:
    if set(predicted_labels) != set(true_labels):
        raise ValueError("predicted and true labels must cover the same records")
    # The headline pair metrics are end-to-end clustering metrics.  They must
    # include true matches omitted by candidate generation and predicted pairs
    # induced by transitive closure, rather than conditioning both sides on the
    # candidate universe.
    predicted_pairs = _pairs_within_groups(predicted_labels)
    true_pairs = _pairs_within_groups(true_labels)
    true_positive = len(predicted_pairs & true_pairs)
    precision = true_positive / len(predicted_pairs) if predicted_pairs else 1.0
    recall = true_positive / len(true_pairs) if true_pairs else 1.0
    pair_f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )

    predicted_members: MutableMapping[str, Set[str]] = {}
    true_members: MutableMapping[str, Set[str]] = {}
    for record_id in predicted_labels:
        predicted_members.setdefault(predicted_labels[record_id], set()).add(record_id)
        true_members.setdefault(true_labels[record_id], set()).add(record_id)
    precisions, recalls = [], []
    for record_id in predicted_labels:
        predicted_group = predicted_members[predicted_labels[record_id]]
        true_group = true_members[true_labels[record_id]]
        intersection = len(predicted_group & true_group)
        precisions.append(intersection / len(predicted_group))
        recalls.append(intersection / len(true_group))
    bc_precision = float(np.mean(precisions))
    bc_recall = float(np.mean(recalls))
    bc_f1 = (
        2 * bc_precision * bc_recall / (bc_precision + bc_recall)
        if bc_precision + bc_recall
        else 0.0
    )
    candidate_recall = conditional_precision = conditional_recall = conditional_f1 = (
        None
    )
    if candidate_pairs is not None:
        candidates = {canonical_pair(*pair) for pair in candidate_pairs}
        true_candidate_pairs = true_pairs & candidates
        predicted_candidate_pairs = predicted_pairs & candidates
        candidate_recall = (
            len(true_candidate_pairs) / len(true_pairs) if true_pairs else 1.0
        )
        conditional_true_positive = len(
            predicted_candidate_pairs & true_candidate_pairs
        )
        conditional_precision = (
            conditional_true_positive / len(predicted_candidate_pairs)
            if predicted_candidate_pairs
            else 1.0
        )
        conditional_recall = (
            conditional_true_positive / len(true_candidate_pairs)
            if true_candidate_pairs
            else 1.0
        )
        conditional_f1 = (
            2
            * conditional_precision
            * conditional_recall
            / (conditional_precision + conditional_recall)
            if conditional_precision + conditional_recall
            else 0.0
        )
    return LinkageMetrics(
        precision,
        recall,
        pair_f1,
        bc_precision,
        bc_recall,
        bc_f1,
        candidate_recall,
        conditional_precision,
        conditional_recall,
        conditional_f1,
    )


def partition_wur(
    old_labels: Mapping[str, str], new_labels: Mapping[str, str]
) -> tuple[float, int, int, int]:
    old_components = {
        label: {record for record, value in old_labels.items() if value == label}
        for label in set(old_labels.values())
    }
    new_components = {
        label: {record for record, value in new_labels.items() if value == label}
        for label in set(new_labels.values())
    }
    changed_old = set(old_components) - set(new_components)
    changed_new = set(new_components) - set(old_components)
    common_records = set(old_labels) & set(new_labels)
    reassigned = sum(
        old_labels[record] != new_labels[record] for record in common_records
    )
    return (
        float(len(changed_old) + len(changed_new)),
        len(changed_old),
        len(changed_new),
        reassigned,
    )


def deletion_witnesses(
    record_ids: Sequence[str],
    true_labels: Mapping[str, str],
    linker: Callable[[Sequence[str]], Mapping[str, str]],
) -> List[DeletionWitness]:
    old_labels = linker(record_ids)
    records_by_entity: MutableMapping[str, List[str]] = {}
    for record_id in record_ids:
        records_by_entity.setdefault(true_labels[record_id], []).append(record_id)
    output = []
    for entity_id, removed in sorted(records_by_entity.items()):
        removed_set = set(removed)
        remaining = [
            record_id for record_id in record_ids if record_id not in removed_set
        ]
        new_labels = linker(remaining)
        wur, changed_old, changed_new, reassigned = partition_wur(
            old_labels, new_labels
        )
        output.append(
            DeletionWitness(
                hashlib.sha256(
                    f"evaluation-entity|{entity_id}".encode("utf-8")
                ).hexdigest()[:24],
                len(removed),
                changed_old,
                changed_new,
                wur,
                reassigned,
            )
        )
    return output


def evaluate_cached_pair_scores(
    records: pd.DataFrame,
    pair_scores: pd.DataFrame,
    threshold: float,
) -> tuple[LinkageMetrics, List[DeletionWitness]]:
    required_records = {"record_id", "true_entity_id"}
    required_pairs = {"left_id", "right_id", "score"}
    if not required_records.issubset(records.columns) or not required_pairs.issubset(
        pair_scores.columns
    ):
        raise ValueError("input tables do not match the frozen identity/WUR schema")
    record_ids = records["record_id"].astype(str).tolist()
    true_labels = dict(
        zip(records["record_id"].astype(str), records["true_entity_id"].astype(str))
    )
    scores = {
        canonical_pair(str(row.left_id), str(row.right_id)): float(row.score)
        for row in pair_scores.itertuples(index=False)
    }
    candidate_pairs = set(scores)

    def linker(active_record_ids: Sequence[str]) -> Mapping[str, str]:
        active = set(active_record_ids)
        positive = [
            pair
            for pair, score in scores.items()
            if score >= threshold and pair[0] in active and pair[1] in active
        ]
        return components_from_pairs(active_record_ids, positive)

    predicted = linker(record_ids)
    metrics = linkage_metrics(predicted, true_labels, candidate_pairs)
    return metrics, deletion_witnesses(record_ids, true_labels, linker)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Common cached-score identity and WUR adapter"
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = pd.read_csv(args.records)
    pair_scores = pd.read_csv(args.pairs)
    metrics, witnesses = evaluate_cached_pair_scores(
        records, pair_scores, args.threshold
    )
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(witness) for witness in witnesses]).to_csv(
        args.output / "deletion_witnesses.csv", index=False
    )
    values = np.asarray([witness.witness_wur_over_c for witness in witnesses])
    atomic_write_json(
        args.output / "identity_wur_summary.json",
        {
            "metrics": asdict(metrics),
            "deletions": len(witnesses),
            "witness_wur_over_c_max": float(values.max()) if len(values) else 0.0,
            "witness_wur_over_c_p99": float(np.quantile(values, 0.99))
            if len(values)
            else 0.0,
            "warning": "Cached scores require a fixed candidate set and inference-only matcher.",
        },
    )


if __name__ == "__main__":
    main()
