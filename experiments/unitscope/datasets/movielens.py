from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Mapping, MutableMapping, Tuple

import numpy as np
import pandas as pd

from ..model import GroundTruth, LocalKind, PublicRecord


GENRES = (
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)


@dataclass(frozen=True)
class MovieLensPartition:
    records: Tuple[PublicRecord, ...]
    truth: GroundTruth
    x: np.ndarray
    y: np.ndarray
    account_split: Mapping[str, Tuple[str, ...]]
    record_split: np.ndarray
    realized_train_overlap: float
    feature_names: Tuple[str, ...]
    task_variant: str = "original"
    balance_majority_keep_probability: float = 1.0


def _rank(seed: int, *parts: object) -> int:
    value = "|".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _opaque(seed: int, *parts: object) -> str:
    value = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _movie_year(title: str) -> float:
    if len(title) >= 6 and title[-6] == "(" and title[-1] == ")":
        try:
            return float(np.clip((int(title[-5:-1]) - 1919) / 100.0, 0.0, 1.0))
        except ValueError:
            pass
    return 0.5


def load_and_partition_movielens1m(
    data_directory: Path,
    *,
    n_silos: int,
    train_overlap_probability: float,
    split_seed: int,
    topology_seed: int,
    maximum_accounts: int | None = 3000,
    maximum_records_per_account: int | None = 32,
    account_selection_seed: int = 31002,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.1,
) -> MovieLensPartition:
    """Build an account-disjoint MovieLens 1M task without record duplication."""

    if n_silos <= 0:
        raise ValueError("n_silos must be positive")
    if not 0 <= train_overlap_probability <= 1:
        raise ValueError("train overlap probability must lie in [0,1]")
    if maximum_records_per_account is not None and maximum_records_per_account <= 0:
        raise ValueError("maximum_records_per_account must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must leave a test set")

    root = Path(data_directory)
    ratings = pd.read_csv(
        root / "ratings.dat",
        sep="::",
        engine="python",
        names=("account", "movie", "rating", "timestamp"),
        dtype={
            "account": "int64",
            "movie": "int64",
            "rating": "int64",
            "timestamp": "int64",
        },
    )
    movies = pd.read_csv(
        root / "movies.dat",
        sep="::",
        engine="python",
        names=("movie", "title", "genres"),
        encoding="latin-1",
    ).set_index("movie")
    users = pd.read_csv(
        root / "users.dat",
        sep="::",
        engine="python",
        names=("account", "gender", "age", "occupation", "zip"),
        dtype={
            "account": "int64",
            "gender": "string",
            "age": "int64",
            "occupation": "int64",
            "zip": "string",
        },
    ).set_index("account")

    accounts = sorted(
        (str(value) for value in ratings["account"].unique()),
        key=lambda account: _rank(account_selection_seed, "eligible", account),
    )
    if maximum_accounts is not None:
        accounts = accounts[:maximum_accounts]
    account_set = {int(account) for account in accounts}
    ratings = ratings[ratings["account"].isin(account_set)].copy()
    ordered_accounts = sorted(
        accounts, key=lambda account: _rank(split_seed, "account-split", account)
    )
    n_train = int(round(train_fraction * len(ordered_accounts)))
    n_validation = int(round(validation_fraction * len(ordered_accounts)))
    account_split = {
        "train": tuple(ordered_accounts[:n_train]),
        "validation": tuple(ordered_accounts[n_train : n_train + n_validation]),
        "test": tuple(ordered_accounts[n_train + n_validation :]),
    }
    split_for_account = {
        account: split_name
        for split_name, split_accounts in account_split.items()
        for account in split_accounts
    }
    overlap_count = int(round(train_overlap_probability * len(account_split["train"])))
    overlapping = set(
        sorted(
            account_split["train"],
            key=lambda account: _rank(topology_seed, "overlap", account),
        )[:overlap_count]
    )

    feature_names = (
        *(f"genre:{genre}" for genre in GENRES),
        "release_year",
        "week_sin",
        "week_cos",
        "day_sin",
        "day_cos",
        "gender_female",
        "age_scaled",
        *(f"occupation:{index}" for index in range(21)),
    )
    records: List[PublicRecord] = []
    assignments: List[Tuple[str, str]] = []
    features: List[np.ndarray] = []
    labels: List[int] = []
    splits: List[str] = []
    train_silos: MutableMapping[str, set[int]] = {
        account: set() for account in account_split["train"]
    }
    for numeric_account, part in ratings.groupby("account", sort=False):
        account = str(int(numeric_account))
        rows = list(part.itertuples(index=False))
        rows = sorted(
            rows,
            key=lambda row: _rank(
                account_selection_seed,
                "record-cap",
                account,
                int(row.movie),
                int(row.timestamp),
            ),
        )
        if maximum_records_per_account is not None:
            rows = rows[:maximum_records_per_account]
        multiplicity = (
            2 if account in overlapping and len(rows) >= 2 and n_silos >= 2 else 1
        )
        selected_silos = sorted(
            range(n_silos),
            key=lambda silo: _rank(topology_seed, "silo", account, silo),
        )[:multiplicity]
        ordered_rows = sorted(
            rows,
            key=lambda row: _rank(
                topology_seed, "rating", account, int(row.movie), int(row.timestamp)
            ),
        )
        user = users.loc[int(numeric_account)]
        split_name = split_for_account[account]
        for position, row in enumerate(ordered_rows):
            silo = selected_silos[position % len(selected_silos)]
            movie = movies.loc[int(row.movie)]
            vector = np.zeros(len(feature_names), dtype=np.float32)
            for genre in str(movie["genres"]).split("|"):
                if genre in GENRES:
                    vector[GENRES.index(genre)] = 1.0
            offset = len(GENRES)
            vector[offset] = _movie_year(str(movie["title"]))
            moment = datetime.fromtimestamp(int(row.timestamp), tz=timezone.utc)
            vector[offset + 1] = np.sin(2 * np.pi * moment.weekday() / 7)
            vector[offset + 2] = np.cos(2 * np.pi * moment.weekday() / 7)
            vector[offset + 3] = np.sin(2 * np.pi * moment.hour / 24)
            vector[offset + 4] = np.cos(2 * np.pi * moment.hour / 24)
            vector[offset + 5] = float(str(user["gender"]) == "F")
            vector[offset + 6] = float(np.clip(float(user["age"]) / 56.0, 0.0, 1.5))
            occupation = int(user["occupation"])
            if 0 <= occupation <= 20:
                vector[offset + 7 + occupation] = 1.0
            record_id = f"movielens-{_opaque(split_seed, account, int(row.movie), int(row.timestamp))}"
            fallback_slot = f"local-{_opaque(topology_seed, account, silo)}"
            records.append(
                PublicRecord(
                    record_id,
                    f"silo-{silo}",
                    fallback_slot,
                    LocalKind.ATOM,
                    1.0,
                )
            )
            assignments.append((record_id, account))
            features.append(vector)
            labels.append(int(int(row.rating) >= 4))
            splits.append(split_name)
            if split_name == "train":
                train_silos[account].add(silo)

    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError(
            "MovieLens rating events are not unique after canonicalization"
        )
    realized_overlap = (
        sum(len(value) >= 2 for value in train_silos.values()) / len(train_silos)
        if train_silos
        else 0.0
    )
    return MovieLensPartition(
        records=tuple(records),
        truth=GroundTruth(tuple(assignments)),
        x=np.stack(features),
        y=np.asarray(labels, dtype=np.int64),
        account_split=account_split,
        record_split=np.asarray(splits),
        realized_train_overlap=realized_overlap,
        feature_names=tuple(feature_names),
    )


def balance_movielens_partition(
    partition: MovieLensPartition,
    *,
    balance_seed: int,
) -> MovieLensPartition:
    """Balance MovieLens using a keep probability derived from training data."""

    train_mask = partition.record_split == "train"
    counts = np.bincount(partition.y[train_mask], minlength=2)
    if counts.min() <= 0:
        raise ValueError(
            "MovieLens balancing requires both classes in the training split"
        )
    majority = int(np.argmax(counts))
    minority_count = int(counts[1 - majority])
    majority_count = int(counts[majority])
    keep_probability = min(1.0, minority_count / majority_count)
    keep = np.ones(len(partition.records), dtype=bool)
    for index, (record, label) in enumerate(zip(partition.records, partition.y)):
        if int(label) == majority:
            rank = _rank(balance_seed, "majority-thinning-v1", record.record_id)
            uniform = rank / float((1 << 64) - 1)
            keep[index] = uniform < keep_probability
    if not np.any(keep & (partition.record_split == "train")):
        raise ValueError("MovieLens balancing removed the complete training split")
    records = tuple(
        record for index, record in enumerate(partition.records) if keep[index]
    )
    truth_map = partition.truth.as_dict()
    truth = GroundTruth(
        tuple((record.record_id, truth_map[record.record_id]) for record in records)
    )
    return MovieLensPartition(
        records=records,
        truth=truth,
        x=partition.x[keep],
        y=partition.y[keep],
        account_split=partition.account_split,
        record_split=partition.record_split[keep],
        realized_train_overlap=partition.realized_train_overlap,
        feature_names=partition.feature_names,
        task_variant="balanced-v1",
        balance_majority_keep_probability=float(keep_probability),
    )
