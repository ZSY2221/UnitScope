from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.unitscope.datasets.femnist import (
    load_leaf_femnist,
    partition_femnist_writers,
)
from experiments.unitscope.datasets.sent140 import load_and_partition_sent140
from experiments.unitscope.datasets.movielens import load_and_partition_movielens1m
from experiments.unitscope.datasets.synthea import (
    build_synthea_future_inpatient_task,
    generate_health_card_policy_evidence,
)
from experiments.unitscope.model import PublicConfig


def test_femnist_writer_split_and_no_image_duplication(tmp_path: Path) -> None:
    users = [f"writer-{index}" for index in range(10)]
    payload = {
        "users": users,
        "num_samples": [4] * len(users),
        "user_data": {
            user: {
                "x": [[float(index), float(sample)] for sample in range(4)],
                "y": [sample % 2 for sample in range(4)],
            }
            for index, user in enumerate(users)
        },
    }
    path = tmp_path / "femnist.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    writers = load_leaf_femnist([path])
    partition = partition_femnist_writers(
        writers,
        n_silos=3,
        train_overlap_probability=0.5,
        split_seed=10,
        topology_seed=20,
        train_fraction=0.6,
        validation_fraction=0.2,
    )
    split_sets = [
        set(partition.writer_split[name])
        for name in ("public_auxiliary", "train", "validation", "test")
    ]
    assert all(
        not split_sets[left] & split_sets[right]
        for left in range(len(split_sets))
        for right in range(left + 1, len(split_sets))
    )
    assert len(partition.records) == 40
    assert len({record.record_id for record in partition.records}) == 40
    assert partition.x.shape == (40, 2)
    assert partition.realized_train_overlap == 0.5
    public_writers = partition.writer_split["public_auxiliary"]
    if public_writers:
        truth = partition.truth.as_dict()
        public_silos = {
            writer: {
                record.silo_id
                for record in partition.records
                if truth[record.record_id] == writer
                and partition.record_split[partition.records.index(record)]
                == "public_auxiliary"
            }
            for writer in public_writers
        }
        assert all(len(silos) >= 1 for silos in public_silos.values())


def test_synthea_adapter_builds_multi_record_patient_task(tmp_path: Path) -> None:
    patients = pd.DataFrame(
        {
            "Id": ["p1", "p2"],
            "BIRTHDATE": ["1980-01-01", "1990-01-01"],
            "GENDER": ["M", "F"],
        }
    )
    encounters = pd.DataFrame(
        {
            "Id": ["e1", "e2", "e3", "e4"],
            "START": [
                "2019-06-01T00:00:00Z",
                "2019-07-01T00:00:00Z",
                "2019-08-01T00:00:00Z",
                "2020-03-01T00:00:00Z",
            ],
            "PATIENT": ["p1", "p1", "p2", "p1"],
            "ORGANIZATION": ["o1", "o2", "o1", "o2"],
            "ENCOUNTERCLASS": ["ambulatory", "emergency", "wellness", "inpatient"],
        }
    )
    patients.to_csv(tmp_path / "patients.csv", index=False)
    encounters.to_csv(tmp_path / "encounters.csv", index=False)
    task = build_synthea_future_inpatient_task(
        tmp_path,
        index_date="2020-01-01",
        lookback_days=365,
        horizon_days=365,
        n_silos=3,
    )
    assert len(task.records) == 3
    assert task.x.shape == (3, len(task.feature_names))
    assert task.y.tolist() == [1, 1, 0]
    assert len(set(record.record_id for record in task.records)) == 3
    truth = task.truth.as_dict()
    assert sum(user == "p1" for user in truth.values()) == 2
    cfg = PublicConfig("health", "e1", 1.0, 1, 3, 0.5, 2.0, 1.0, 1, 1e-6, 3)
    participating = [task.records[0].silo_id, task.records[1].silo_id]
    evidence = generate_health_card_policy_evidence(
        task,
        [record.record_id for record in task.records],
        cfg,
        issuance_date="2019-06-15",
        participating_silos=participating,
    )
    assert evidence.realized_handle_user_fraction > 0
    covered = {
        record_id
        for claim in evidence.evidence.partial_handles
        for record_id in claim.record_ids
    }
    first_record = task.records[0].record_id
    assert first_record not in covered  # historical residual remains local


def test_sent140_account_split_and_no_tweet_duplication(tmp_path: Path) -> None:
    rows = []
    for account_index in range(10):
        for tweet_index in range(3):
            rows.append(
                [
                    4 if tweet_index % 2 else 0,
                    f"tweet-{account_index}-{tweet_index}",
                    "date",
                    "NO_QUERY",
                    f"account-{account_index}",
                    f"text {tweet_index}",
                ]
            )
    path = tmp_path / "sent140.csv"
    pd.DataFrame(rows).to_csv(path, index=False, header=False, encoding="latin-1")
    partition = load_and_partition_sent140(
        path,
        n_silos=3,
        train_overlap_probability=0.5,
        split_seed=10,
        topology_seed=20,
        train_fraction=0.6,
        validation_fraction=0.2,
    )
    split_sets = [
        set(partition.account_split[name]) for name in ("train", "validation", "test")
    ]
    assert not split_sets[0] & split_sets[1]
    assert not split_sets[0] & split_sets[2]
    assert not split_sets[1] & split_sets[2]
    assert len(partition.records) == 30
    assert len(set(record.record_id for record in partition.records)) == 30
    assert partition.x.shape == (30, 512)
    assert partition.realized_train_overlap == 0.5
    assert partition.ambiguous_duplicate_rows_removed == 0


def test_sent140_public_cap_is_deterministic(tmp_path: Path) -> None:
    rows = []
    for account_index in range(8):
        for tweet_index in range(10):
            rows.append(
                [
                    4 if tweet_index % 2 else 0,
                    f"tweet-{account_index}-{tweet_index}",
                    "date",
                    "NO_QUERY",
                    f"account-{account_index}",
                    f"public text {tweet_index}",
                ]
            )
    path = tmp_path / "sent140.csv"
    pd.DataFrame(rows).to_csv(path, index=False, header=False, encoding="latin-1")
    first = load_and_partition_sent140(
        path,
        n_silos=3,
        train_overlap_probability=0.5,
        split_seed=10,
        topology_seed=20,
        maximum_records_per_account=4,
        account_selection_seed=30,
    )
    second = load_and_partition_sent140(
        path,
        n_silos=3,
        train_overlap_probability=0.5,
        split_seed=10,
        topology_seed=20,
        maximum_records_per_account=4,
        account_selection_seed=30,
    )
    assert len(first.records) == 32
    assert [record.record_id for record in first.records] == [
        record.record_id for record in second.records
    ]
    assert np.array_equal(first.x, second.x)


def test_sent140_drops_all_conflicting_duplicate_tweet_ids(tmp_path: Path) -> None:
    rows = []
    for account_index in range(6):
        for tweet_index in range(3):
            rows.append(
                [
                    0,
                    f"tweet-{account_index}-{tweet_index}",
                    "date",
                    "NO_QUERY",
                    f"account-{account_index}",
                    "text",
                ]
            )
    rows.extend(
        [
            [0, "ambiguous", "date", "NO_QUERY", "account-0", "same text"],
            [4, "ambiguous", "date", "NO_QUERY", "account-0", "same text"],
        ]
    )
    path = tmp_path / "sent140-conflict.csv"
    pd.DataFrame(rows).to_csv(path, index=False, header=False, encoding="latin-1")
    partition = load_and_partition_sent140(
        path,
        n_silos=2,
        train_overlap_probability=0.5,
        split_seed=10,
        topology_seed=20,
    )
    assert partition.ambiguous_duplicate_rows_removed == 2
    assert len(partition.records) == 18


def test_femnist_dirichlet_assignment_is_deterministic_and_no_duplication(
    tmp_path: Path,
) -> None:
    users = [f"writer-{index}" for index in range(12)]
    payload = {
        "users": users,
        "num_samples": [12] * len(users),
        "user_data": {
            user: {
                "x": [[float(index), float(sample)] for sample in range(12)],
                "y": [sample % 3 for sample in range(12)],
            }
            for index, user in enumerate(users)
        },
    }
    path = tmp_path / "femnist.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    writers = load_leaf_femnist([path])
    first = partition_femnist_writers(
        writers,
        n_silos=4,
        train_overlap_probability=0.5,
        split_seed=11,
        topology_seed=21,
        dirichlet_alpha=0.1,
    )
    second = partition_femnist_writers(
        writers,
        n_silos=4,
        train_overlap_probability=0.5,
        split_seed=11,
        topology_seed=21,
        dirichlet_alpha=0.1,
    )
    assert len(first.records) == len({record.record_id for record in first.records})
    assert [record.silo_id for record in first.records] == [
        record.silo_id for record in second.records
    ]
    assert first.realized_train_overlap == 0.5


def test_movielens_account_split_features_and_no_duplication(tmp_path: Path) -> None:
    users = []
    ratings = []
    for account in range(1, 11):
        users.append(
            f"{account}::{('F' if account % 2 else 'M')}::25::{account % 5}::00000"
        )
        for index in range(5):
            ratings.append(
                f"{account}::{1 + index}::{1 + (account + index) % 5}::{946684800 + index * 86400}"
            )
    movies = [
        "1::Movie One (1995)::Action|Comedy",
        "2::Movie Two (1996)::Drama",
        "3::Movie Three (1997)::Romance",
        "4::Movie Four (1998)::Sci-Fi|Thriller",
        "5::Movie Five (1999)::Documentary",
    ]
    (tmp_path / "users.dat").write_text("\n".join(users), encoding="latin-1")
    (tmp_path / "ratings.dat").write_text("\n".join(ratings), encoding="latin-1")
    (tmp_path / "movies.dat").write_text("\n".join(movies), encoding="latin-1")
    partition = load_and_partition_movielens1m(
        tmp_path,
        n_silos=3,
        train_overlap_probability=0.5,
        split_seed=10,
        topology_seed=20,
        maximum_records_per_account=4,
        train_fraction=0.6,
        validation_fraction=0.2,
    )
    split_sets = [
        set(partition.account_split[name]) for name in ("train", "validation", "test")
    ]
    assert all(
        not split_sets[left] & split_sets[right]
        for left in range(len(split_sets))
        for right in range(left + 1, len(split_sets))
    )
    assert len(partition.records) == 40
    assert len({record.record_id for record in partition.records}) == 40
    assert partition.x.shape == (40, len(partition.feature_names))
    assert np.isfinite(partition.x).all()
    assert partition.realized_train_overlap == 0.5
