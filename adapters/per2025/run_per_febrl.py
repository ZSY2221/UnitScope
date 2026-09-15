"""Run the 2025 PER implementation on the common FEBRL inputs."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import jellyfish
import networkx as nx
import numpy as np
import pandas as pd
from pyjedai.datamodel import Data
from pyjedai.block_building import StandardBlocking
from pyjedai.prioritization import PESM
from pyjedai.workflow import ProgressiveWorkFlow


ATTRIBUTES = (
    "given_name",
    "surname",
    "street_number",
    "address_1",
    "address_2",
    "suburb",
    "postcode",
    "state",
    "date_of_birth",
    "soc_sec_id",
)

SIMILARITY_WEIGHTS = (
    ("given_name", 0.16),
    ("surname", 0.20),
    ("address_1", 0.14),
    ("suburb", 0.08),
    ("postcode", 0.10),
    ("date_of_birth", 0.17),
    ("soc_sec_id", 0.15),
)


def normalize(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def common_pair_score(left: pd.Series, right: pd.Series) -> float:
    weighted = 0.0
    total = 0.0
    for column, weight in SIMILARITY_WEIGHTS:
        a = left[f"_{column}"]
        b = right[f"_{column}"]
        if not a or not b:
            continue
        weighted += weight * jellyfish.jaro_winkler_similarity(a, b)
        total += weight
    return weighted / total if total else 0.0


def git_value(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True, encoding="utf-8"
    ).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truth_for_silo_pair(
    truth: pd.DataFrame, left_silo: int, right_silo: int
) -> pd.DataFrame:
    forward = truth[
        (truth["left_silo"] == left_silo) & (truth["right_silo"] == right_silo)
    ][["left", "right"]]
    reverse = truth[
        (truth["left_silo"] == right_silo) & (truth["right_silo"] == left_silo)
    ][["right", "left"]]
    reverse = reverse.rename(columns={"right": "left", "left": "right"})
    return pd.concat([forward, reverse], ignore_index=True).astype(str)


def run_official_per(
    dataset: str,
    records: pd.DataFrame,
    truth: pd.DataFrame,
    n_silos: int,
    budget_multiplier: int,
    minimum_budget: int,
) -> tuple[pd.DataFrame, list[dict]]:
    retained: dict[tuple[str, str], dict] = {}
    run_rows: list[dict] = []

    for left_silo, right_silo in itertools.combinations(range(n_silos), 2):
        left = records[records["silo"] == left_silo].copy()
        right = records[records["silo"] == right_silo].copy()
        pair_truth = truth_for_silo_pair(truth, left_silo, right_silo)
        budget = max(minimum_budget, budget_multiplier * len(pair_truth))
        budget = min(budget, len(left) * len(right))

        data = Data(
            dataset_1=left,
            id_column_name_1="id",
            attributes_1=list(ATTRIBUTES),
            dataset_name_1=f"{dataset}-s{left_silo}",
            dataset_2=right,
            id_column_name_2="id",
            attributes_2=list(ATTRIBUTES),
            dataset_name_2=f"{dataset}-s{right_silo}",
            ground_truth=pair_truth,
        )
        started = time.perf_counter()
        if len(pair_truth):
            workflow = ProgressiveWorkFlow(
                name=f"official-PER-{dataset}-s{left_silo}-s{right_silo}"
            )
            workflow.run(
                data=data,
                matcher="PESM",
                algorithm="TOP",
                dataset=f"{dataset}_s{left_silo}_s{right_silo}",
                weighting_scheme="CN-CBS",
                indexing="inorder",
                budget=budget,
                workflow_step_tqdm_disable=True,
                workflow_tqdm_enable=False,
            )
            final_pairs = workflow.final_pairs
            official_seconds = workflow.workflow_exec_time
            status = "completed_official_workflow"
        else:
            # pyJedAI 0.2.0's workflow wrapper divides by zero while evaluating
            # a source pair with no ground-truth matches.  Candidate generation
            # itself does not use the labels, so invoke the same official core
            # classes directly and record the wrapper bypass in provenance.
            blocker = StandardBlocking()
            blocks = blocker.build_blocks(
                data,
                attributes_1=list(ATTRIBUTES),
                attributes_2=list(ATTRIBUTES),
                tqdm_disable=True,
            )
            matcher = PESM(weighting_scheme="CN-CBS")
            final_pairs = matcher.predict(
                data=data,
                blocks=blocks,
                dataset_identifier=f"{dataset}_s{left_silo}_s{right_silo}",
                budget=budget,
                algorithm="TOP",
                indexing="inorder",
                tqdm_disable=True,
            )
            official_seconds = blocker.execution_time + matcher.execution_time
            status = "completed_official_core_zero_truth_wrapper_bypass"
        elapsed = time.perf_counter() - started
        pair_count = 0
        for official_score, first, second in final_pairs:
            left_id, right_id = sorted((str(first), str(second)))
            if left_id == right_id:
                continue
            key = (left_id, right_id)
            candidate = {
                "dataset": dataset,
                "left": left_id,
                "right": right_id,
                "official_schedule_score": float(official_score),
                "source_silo_pair": f"{left_silo}-{right_silo}",
            }
            if (
                key not in retained
                or candidate["official_schedule_score"]
                > retained[key]["official_schedule_score"]
            ):
                retained[key] = candidate
            pair_count += 1
        run_rows.append(
            {
                "dataset": dataset,
                "left_silo": left_silo,
                "right_silo": right_silo,
                "records_left": int(len(left)),
                "records_right": int(len(right)),
                "truth_pairs": int(len(pair_truth)),
                "budget": int(budget),
                "emitted_pairs_raw": int(pair_count),
                "official_reported_seconds": float(official_seconds),
                "adapter_wall_seconds": float(elapsed),
                "status": status,
            }
        )

    return pd.DataFrame(retained.values()), run_rows


def pairwise_metrics(components: list[set[str]], truth_map: dict[str, int]) -> dict:
    true_sizes = Counter(truth_map.values())
    total_true = sum(size * (size - 1) // 2 for size in true_sizes.values())
    predicted = 0
    true_positive = 0
    for component in components:
        predicted += len(component) * (len(component) - 1) // 2
        counts = Counter(truth_map[node] for node in component)
        true_positive += sum(count * (count - 1) // 2 for count in counts.values())
    precision = true_positive / predicted if predicted else 1.0
    recall = true_positive / total_true if total_true else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": f1,
        "predicted_pairs": predicted,
        "true_positive_pairs": true_positive,
        "total_true_pairs": total_true,
    }


def deletion_witnesses(
    graph: nx.Graph, truth_map: dict[str, int], threshold: float
) -> tuple[pd.DataFrame, dict]:
    entity_nodes: dict[int, set[str]] = defaultdict(set)
    for node, entity in truth_map.items():
        entity_nodes[entity].add(node)

    components = [set(component) for component in nx.connected_components(graph)]
    node_component: dict[str, int] = {}
    for component_id, members in enumerate(components):
        for member in members:
            node_component[member] = component_id

    rows: list[dict] = []
    for entity, deleted in entity_nodes.items():
        touched_ids = {node_component[node] for node in deleted}
        old_tags: set[frozenset[str]] = set()
        new_tags: set[frozenset[str]] = set()
        touched_nodes: set[str] = set()
        for component_id in touched_ids:
            members = components[component_id]
            touched_nodes.update(members)
            old_tags.add(frozenset(members))
            remaining = members - deleted
            if remaining:
                induced = graph.subgraph(remaining)
                new_tags.update(
                    frozenset(part) for part in nx.connected_components(induced)
                )
        deleted_tags = old_tags - new_tags
        inserted_tags = new_tags - old_tags
        affected = (
            set().union(*(set(tag) for tag in deleted_tags | inserted_tags))
            if deleted_tags or inserted_tags
            else set()
        )
        rows.append(
            {
                "threshold": threshold,
                "entity_id": entity,
                "deleted_records": len(deleted),
                "touched_components": len(touched_ids),
                "touched_records": len(touched_nodes),
                "deleted_group_tags": len(deleted_tags),
                "inserted_group_tags": len(inserted_tags),
                "witnessed_wur": len(deleted_tags) + len(inserted_tags),
                "changed_surviving_records": len(affected - deleted),
            }
        )
    frame = pd.DataFrame(rows)
    worst = frame.sort_values(
        ["witnessed_wur", "changed_surviving_records"], ascending=False
    ).iloc[0]
    return frame, {
        "mean_witnessed_wur": float(frame["witnessed_wur"].mean()),
        "p95_witnessed_wur": float(frame["witnessed_wur"].quantile(0.95)),
        "max_witnessed_wur": int(frame["witnessed_wur"].max()),
        "max_changed_surviving_records": int(frame["changed_surviving_records"].max()),
        "worst_entity_id": int(worst["entity_id"]),
    }


def evaluate_thresholds(
    dataset: str,
    records: pd.DataFrame,
    truth: pd.DataFrame,
    candidates: pd.DataFrame,
    thresholds: list[float],
) -> tuple[list[dict], list[pd.DataFrame]]:
    records = records.set_index("id", drop=False)
    truth_map = {str(row.id): int(row.entity_id) for row in records.itertuples()}
    true_pairs = {
        tuple(sorted((str(row.left), str(row.right)))) for row in truth.itertuples()
    }
    if candidates.empty:
        raise RuntimeError(f"official PER emitted no candidates for {dataset}")

    common_scores: list[float] = []
    true_flags: list[int] = []
    for row in candidates.itertuples():
        common_scores.append(
            common_pair_score(records.loc[row.left], records.loc[row.right])
        )
        true_flags.append(int((row.left, row.right) in true_pairs))
    candidates["common_adapter_score"] = common_scores
    candidates["is_true_pair"] = true_flags

    candidate_truth_coverage = len(
        {(row.left, row.right) for row in candidates.itertuples() if row.is_true_pair}
    ) / len(true_pairs)
    multi_entities = {
        entity for entity, count in Counter(truth_map.values()).items() if count > 1
    }

    rows: list[dict] = []
    witness_frames: list[pd.DataFrame] = []
    all_nodes = list(truth_map)
    for threshold in thresholds:
        selected = candidates[candidates["common_adapter_score"] >= threshold]
        graph = nx.Graph()
        graph.add_nodes_from(all_nodes)
        graph.add_edges_from(
            selected[["left", "right"]].itertuples(index=False, name=None)
        )
        components = [set(component) for component in nx.connected_components(graph)]
        node_component = {
            node: component_id
            for component_id, members in enumerate(components)
            for node in members
        }
        recovered_entities = {
            entity
            for entity in multi_entities
            if len(
                {
                    node_component[node]
                    for node, node_entity in truth_map.items()
                    if node_entity == entity
                }
            )
            < sum(1 for node_entity in truth_map.values() if node_entity == entity)
        }
        incident = set(selected["left"]) | set(selected["right"])
        direct_true = int(selected["is_true_pair"].sum())
        witnesses, witness_summary = deletion_witnesses(graph, truth_map, threshold)
        witnesses.insert(0, "dataset", dataset)
        witness_frames.append(witnesses)
        rows.append(
            {
                "dataset": dataset,
                "method": "PER-2025 official scheduler + common FEBRL adapter",
                "threshold": threshold,
                "candidate_pairs": int(len(candidates)),
                "candidate_truth_pair_coverage": candidate_truth_coverage,
                "selected_edges": int(len(selected)),
                "direct_true_edge_recall": direct_true / len(true_pairs),
                "record_edge_coverage": len(incident) / len(records),
                "multi_entity_link_coverage": len(recovered_entities)
                / len(multi_entities),
                "largest_component": max(map(len, components)),
                **pairwise_metrics(components, truth_map),
                **witness_summary,
                "wur_status": "empirical deletion witness, not a deterministic certificate",
            }
        )
    return rows, witness_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["febrl1", "febrl2", "febrl3"])
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=[0.72, 0.76, 0.80, 0.84, 0.88]
    )
    parser.add_argument("--n-silos", type=int, default=6)
    parser.add_argument("--budget-multiplier", type=int, default=10)
    parser.add_argument("--minimum-budget", type=int, default=100)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    all_candidates: list[pd.DataFrame] = []
    all_threshold_rows: list[dict] = []
    all_witnesses: list[pd.DataFrame] = []
    official_runs: list[dict] = []

    for dataset in args.datasets:
        records_path = args.input / dataset / "records.csv"
        truth_path = args.input / dataset / "truth_pairs.csv"
        records = pd.read_csv(records_path, dtype={"id": str}).fillna("")
        truth = pd.read_csv(truth_path, dtype={"left": str, "right": str})
        for column, _ in SIMILARITY_WEIGHTS:
            records[f"_{column}"] = records[column].map(normalize)

        candidates, run_rows = run_official_per(
            dataset,
            records,
            truth,
            args.n_silos,
            args.budget_multiplier,
            args.minimum_budget,
        )
        threshold_rows, witness_frames = evaluate_thresholds(
            dataset, records, truth, candidates, args.thresholds
        )
        all_candidates.append(candidates)
        all_threshold_rows.extend(threshold_rows)
        all_witnesses.extend(witness_frames)
        official_runs.extend(run_rows)

    candidates_path = args.output / "per_official_candidates.csv"
    scan_path = args.output / "per_febrl_threshold_scan.csv"
    witnesses_path = args.output / "per_febrl_wur_witnesses.csv"
    pd.concat(all_candidates, ignore_index=True).to_csv(
        candidates_path, index=False, lineterminator="\n"
    )
    pd.DataFrame(all_threshold_rows).to_csv(scan_path, index=False, lineterminator="\n")
    pd.concat(all_witnesses, ignore_index=True).to_csv(
        witnesses_path, index=False, lineterminator="\n"
    )

    manifest = {
        "baseline": "Progressive Entity Resolution: A Design Space Exploration",
        "venue": "PACMMOD 2025 / SIGMOD 2025",
        "official_repository": git_value(
            args.official_repo, "remote", "get-url", "origin"
        ),
        "official_commit": git_value(args.official_repo, "rev-parse", "HEAD"),
        "official_tree_clean": not bool(
            git_value(args.official_repo, "status", "--porcelain")
        ),
        "repository_license": "none detected in repository or GitHub metadata",
        "execution_class": "official pyJedAI/PER code with our FEBRL input and evaluation adapter",
        "official_code_modified": False,
        "official_method": {
            "matcher": "PESM",
            "algorithm": "TOP",
            "weighting_scheme": "CN-CBS",
            "indexing": "inorder",
            "budget_rule": f"max({args.minimum_budget}, {args.budget_multiplier} * truth pairs) per silo pair",
        },
        "adapter_method": {
            "silos": args.n_silos,
            "thresholds": args.thresholds,
            "similarity": "same weighted Jaro-Winkler columns as stableunit_pilot.py",
            "clustering": "connected components over thresholded official candidates",
            "wur": "delete each truth entity and count changed component tags",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "networkx": nx.__version__,
            "pyjedai_distribution": "0.2.0",
        },
        "official_silo_runs": official_runs,
        "outputs": {
            path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in (candidates_path, scan_path, witnesses_path)
        },
    }
    (args.output / "per_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
