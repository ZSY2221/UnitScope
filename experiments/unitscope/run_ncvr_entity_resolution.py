from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import zipfile
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Features:
    first: str
    last: str
    middle: str
    zip5: str
    birth_year: str
    gender: str


def normalize(value: str) -> str:
    return "".join(
        character for character in value.upper().strip() if character.isalnum()
    )


def stable_u64(*parts: str) -> int:
    payload = "|".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def corrupt(
    value: str, identity: str, field: str, view: str, rate: float, missing: float
) -> str:
    if not value:
        return value
    unit = stable_u64(identity, field, view) / 2**64
    if unit < missing:
        return ""
    if unit >= missing + rate:
        return value
    selector = stable_u64(identity, field, view, "operation") % 3
    position = stable_u64(identity, field, view, "position") % len(value)
    if selector == 0 and len(value) > 1:
        return value[:position] + value[position + 1 :]
    if selector == 1 and len(value) > 1:
        other = min(len(value) - 1, position + 1)
        characters = list(value)
        characters[position], characters[other] = (
            characters[other],
            characters[position],
        )
        return "".join(characters)
    replacement = chr(ord("A") + stable_u64(identity, field, view, "replacement") % 26)
    return value[:position] + replacement + value[position + 1 :]


def derive(
    feature: Features, identity: str, view: str, rate: float, missing: float
) -> Features:
    return Features(
        corrupt(feature.first, identity, "first", view, rate, missing),
        corrupt(feature.last, identity, "last", view, rate, missing),
        corrupt(feature.middle, identity, "middle", view, rate, missing),
        corrupt(feature.zip5, identity, "zip", view, rate / 2, missing / 2),
        corrupt(feature.birth_year, identity, "birth", view, rate / 3, missing / 3),
        corrupt(feature.gender, identity, "gender", view, rate / 4, missing / 4),
    )


def soundex(value: str) -> str:
    if not value:
        return ""
    mapping = {
        **dict.fromkeys("BFPV", "1"),
        **dict.fromkeys("CGJKQSXZ", "2"),
        **dict.fromkeys("DT", "3"),
        **dict.fromkeys("L", "4"),
        **dict.fromkeys("MN", "5"),
        **dict.fromkeys("R", "6"),
    }
    output = [value[0]]
    previous = mapping.get(value[0], "")
    for character in value[1:]:
        code = mapping.get(character, "")
        if code and code != previous:
            output.append(code)
        previous = code
        if len(output) == 4:
            break
    return ("".join(output) + "000")[:4]


def grams(value: str) -> set[str]:
    if not value:
        return set()
    padded = f"^{value}$"
    return {padded[index : index + 2] for index in range(len(padded) - 1)}


def jaccard(left: str, right: str) -> float:
    a, b = grams(left), grams(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def score(left: Features, right: Features) -> float:
    name = 0.38 * jaccard(left.last, right.last) + 0.30 * jaccard(
        left.first, right.first
    )
    middle = 0.08 * jaccard(left.middle, right.middle)
    zip_score = 0.12 * (left.zip5 == right.zip5 and bool(left.zip5)) + 0.04 * (
        left.zip5[:3] == right.zip5[:3] and bool(left.zip5[:3])
    )
    birth = 0.06 * (left.birth_year == right.birth_year and bool(left.birth_year))
    gender = 0.02 * (left.gender == right.gender and bool(left.gender))
    return float(name + middle + zip_score + birth + gender)


def blocking_keys(feature: Features) -> tuple[str, ...]:
    keys = []
    if feature.last and feature.zip5:
        keys.append("lz:" + feature.last[:2] + ":" + feature.zip5[:3])
    if feature.first and feature.birth_year:
        keys.append("fb:" + feature.first[:1] + ":" + feature.birth_year)
    sx = soundex(feature.last)
    if sx and feature.birth_year:
        keys.append("sb:" + sx + ":" + feature.birth_year)
    return tuple(keys)


def load_views(
    zip_path: Path, modulus: int, maximum: int
) -> tuple[list[Features], list[Features], dict[str, object]]:
    selected: list[tuple[int, Features, Features]] = []
    scanned = 0
    eligible = 0
    started = time.perf_counter()
    with zipfile.ZipFile(zip_path) as archive:
        member = max(archive.infolist(), key=lambda item: item.file_size)
        with archive.open(member) as binary:
            text = io.TextIOWrapper(
                binary, encoding="utf-8", errors="replace", newline=""
            )
            reader = csv.DictReader(text, delimiter="\t", quotechar='"')
            required = {
                "ncid",
                "last_name",
                "first_name",
                "zip_code",
                "birth_year",
                "gender_code",
                "status_cd",
                "confidential_ind",
            }
            missing_columns = required - set(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(
                    f"NCVR archive misses required fields: {sorted(missing_columns)}"
                )
            for row in reader:
                scanned += 1
                identity = normalize(row.get("ncid", ""))
                first = normalize(row.get("first_name", ""))
                last = normalize(row.get("last_name", ""))
                if not identity or not first or not last:
                    continue
                if (
                    normalize(row.get("status_cd", "")) != "A"
                    or normalize(row.get("confidential_ind", "")) == "Y"
                ):
                    continue
                eligible += 1
                rank = stable_u64(identity, "ncvr-sample")
                if rank % modulus:
                    continue
                base = Features(
                    first,
                    last,
                    normalize(row.get("middle_name", "")),
                    normalize(row.get("zip_code", ""))[:5],
                    normalize(row.get("birth_year", "")),
                    normalize(row.get("gender_code", "")),
                )
                selected.append(
                    (
                        rank,
                        derive(base, identity, "A", rate=0.08, missing=0.03),
                        derive(base, identity, "B", rate=0.12, missing=0.05),
                    )
                )
    selected.sort(key=lambda item: item[0])
    selected = selected[:maximum]
    return (
        [item[1] for item in selected],
        [item[2] for item in selected],
        {
            "archive_member": member.filename,
            "archive_uncompressed_bytes": member.file_size,
            "rows_scanned": scanned,
            "eligible_active_nonconfidential": eligible,
            "selected_people": len(selected),
            "selection_modulus": modulus,
            "load_seconds": time.perf_counter() - started,
        },
    )


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def build_edges(left: list[Features], right: list[Features], block_cap: int = 128):
    index: dict[str, list[int]] = defaultdict(list)
    for position, feature in enumerate(right):
        for key in blocking_keys(feature):
            bucket = index[key]
            if len(bucket) < block_cap:
                bucket.append(position)
    edge_left, edge_right, edge_score = array("I"), array("I"), array("f")
    true_in_candidates = 0
    candidate_total = 0
    for position, feature in enumerate(left):
        candidates: set[int] = set()
        for key in blocking_keys(feature):
            candidates.update(index.get(key, ()))
        if position in candidates:
            true_in_candidates += 1
        candidate_total += len(candidates)
        for candidate in candidates:
            value = score(feature, right[candidate])
            if value >= 0.55:
                edge_left.append(position)
                edge_right.append(candidate)
                edge_score.append(value)
    return edge_left, edge_right, edge_score, true_in_candidates, candidate_total


def deletion_wur(
    n: int,
    uf: UnionFind,
    edge_left: array,
    edge_right: array,
    edge_score: array,
    threshold: float,
    sample_size: int = 1000,
) -> tuple[float, float, float, float]:
    sample = sorted(range(n), key=lambda value: stable_u64(str(value), "delete"))[
        : min(sample_size, n)
    ]
    touched_roots = {
        uf.find(value) for person in sample for value in (person, n + person)
    }
    adjacency: dict[int, list[int]] = defaultdict(list)
    nodes_by_root: dict[int, set[int]] = defaultdict(set)
    for person in sample:
        for node in (person, n + person):
            nodes_by_root[uf.find(node)].add(node)
    for left, right, value in zip(edge_left, edge_right, edge_score):
        if value < threshold:
            continue
        a, b = int(left), n + int(right)
        root = uf.find(a)
        if root in touched_roots:
            adjacency[a].append(b)
            adjacency[b].append(a)
            nodes_by_root[root].update((a, b))
    values = []
    survivor_changes = []
    for person in sample:
        removed = {person, n + person}
        roots = {uf.find(node) for node in removed}
        old_labels = len(roots)
        new_labels = 0
        changed_survivors = 0
        for root in roots:
            remaining = nodes_by_root[root] - removed
            changed_survivors += len(remaining)
            unseen = set(remaining)
            while unseen:
                new_labels += 1
                stack = [unseen.pop()]
                while stack:
                    node = stack.pop()
                    for neighbor in adjacency.get(node, ()):
                        if neighbor in unseen:
                            unseen.remove(neighbor)
                            stack.append(neighbor)
        values.append(old_labels + new_labels)
        survivor_changes.append(changed_survivors)
    vector = np.asarray(values, dtype=np.float64)
    return (
        float(vector.mean()),
        float(np.quantile(vector, 0.99)),
        float(vector.max()),
        float(max(survivor_changes, default=0)),
    )


def evaluate_scale(
    left: list[Features], right: list[Features], thresholds: tuple[float, ...]
) -> list[dict[str, object]]:
    started = time.perf_counter()
    edge_left, edge_right, edge_score, true_candidates, candidate_total = build_edges(
        left, right
    )
    edge_seconds = time.perf_counter() - started
    n = len(left)
    rows = []
    for threshold in thresholds:
        started = time.perf_counter()
        uf = UnionFind(2 * n)
        retained = 0
        for a, b, value in zip(edge_left, edge_right, edge_score):
            if value >= threshold:
                uf.union(int(a), n + int(b))
                retained += 1
        roots = np.asarray([uf.find(index) for index in range(2 * n)], dtype=np.int32)
        _, sizes = np.unique(roots, return_counts=True)
        predicted_pairs = int(
            np.sum(sizes.astype(np.int64) * (sizes.astype(np.int64) - 1) // 2)
        )
        recovered = sum(uf.find(index) == uf.find(n + index) for index in range(n))
        precision = recovered / predicted_pairs if predicted_pairs else 0.0
        recall = recovered / n if n else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        wur_mean, wur_p99, wur_max, changed_survivors = deletion_wur(
            n, uf, edge_left, edge_right, edge_score, threshold
        )
        rows.append(
            {
                "people": n,
                "records": 2 * n,
                "threshold": threshold,
                "candidate_recall": true_candidates / n,
                "mean_candidates": candidate_total / n,
                "retained_edges": retained,
                "pair_precision": precision,
                "pair_recall": recall,
                "pair_f1": f1,
                "components": int(len(sizes)),
                "largest_component": int(sizes.max(initial=0)),
                "deletion_wur_mean_over_c": wur_mean,
                "deletion_wur_p99_over_c": wur_p99,
                "deletion_wur_max_over_c": wur_max,
                "max_changed_survivor_records": changed_survivors,
                "edge_build_seconds": edge_seconds,
                "evaluate_seconds": time.perf_counter() - started,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-people", type=int, default=250_000)
    parser.add_argument("--selection-modulus", type=int, default=32)
    parser.add_argument(
        "--scales", type=int, nargs="+", default=[50_000, 100_000, 250_000]
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    left, right, source = load_views(
        args.zip, args.selection_modulus, args.maximum_people
    )
    if len(left) < min(args.scales):
        raise RuntimeError(f"NCVR selection produced only {len(left)} people")
    rows = []
    for scale in args.scales:
        actual = min(scale, len(left))
        rows.extend(evaluate_scale(left[:actual], right[:actual], (0.70, 0.80, 0.90)))
        print(json.dumps({"scale": actual, "completed": True}), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "ncvr_entity_resolution_results.csv", index=False)
    digest = hashlib.sha256()
    with args.zip.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    manifest = {
        **source,
        "zip": str(args.zip.resolve()),
        "zip_sha256": digest.hexdigest(),
        "scales": args.scales,
        "maximum_people": args.maximum_people,
        "two_view_derivation": "deterministic field corruption from one official snapshot",
        "raw_fields_persisted": False,
        "result_rows": len(frame),
        "warning": "Semi-synthetic multi-source ER view on real public voter-field distributions; not a real hospital federation.",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
