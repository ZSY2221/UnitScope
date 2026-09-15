from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, MutableMapping, Tuple

import numpy as np
import pandas as pd

from ..model import GroundTruth, LocalKind, PublicRecord


COLUMNS = ("target", "tweet_id", "date", "query", "account", "text")


@dataclass(frozen=True)
class Sent140Partition:
    records: Tuple[PublicRecord, ...]
    truth: GroundTruth
    x: np.ndarray
    texts: np.ndarray
    y: np.ndarray
    account_split: Mapping[str, Tuple[str, ...]]
    record_split: np.ndarray
    realized_train_overlap: float
    ambiguous_duplicate_rows_removed: int


def _rank(seed: int, *parts: object) -> int:
    value = "|".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _opaque(seed: int, *parts: object) -> str:
    value = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def load_and_partition_sent140(
    csv_path: Path,
    *,
    n_silos: int,
    train_overlap_probability: float,
    split_seed: int,
    topology_seed: int,
    minimum_records_per_account: int = 2,
    maximum_accounts: int | None = None,
    maximum_records_per_account: int | None = 32,
    account_selection_seed: int = 31002,
    text_hash_dimension: int = 512,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.1,
    nrows: int | None = None,
) -> Sent140Partition:
    """Build the account-disjoint Sent140 task without record duplication."""

    if n_silos <= 0 or minimum_records_per_account <= 0:
        raise ValueError("n_silos and minimum_records_per_account must be positive")
    if (
        maximum_records_per_account is not None
        and maximum_records_per_account < minimum_records_per_account
    ):
        raise ValueError(
            "the account cap must retain the minimum eligible record count"
        )
    if text_hash_dimension <= 0:
        raise ValueError("text_hash_dimension must be positive")
    if not 0 <= train_overlap_probability <= 1:
        raise ValueError("train overlap probability must lie in [0,1]")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must leave a test set")
    frame = pd.read_csv(
        Path(csv_path),
        names=COLUMNS,
        encoding="latin-1",
        header=None,
        nrows=nrows,
        usecols=(0, 1, 4, 5),
        dtype={
            "target": "int64",
            "tweet_id": "string",
            "account": "string",
            "text": "string",
        },
    )
    frame = frame.dropna(subset=["account", "text", "tweet_id"])
    # The public release repeats 1,685 tweet IDs with identical account and
    # text but conflicting sentiment targets.  A single event cannot have two
    # labels.  Drop every occurrence of an ambiguous ID by a label-independent
    # public rule rather than selecting one target after observing outcomes.
    ambiguous = frame["tweet_id"].duplicated(keep=False)
    ambiguous_duplicate_rows_removed = int(ambiguous.sum())
    frame = frame.loc[~ambiguous].copy()
    counts = frame.groupby("account", sort=False).size()
    eligible = counts[counts >= minimum_records_per_account].index.astype(str).tolist()
    # Population selection is public and independent of the replicate split.
    eligible = sorted(
        eligible,
        key=lambda account: _rank(account_selection_seed, "eligible", account),
    )
    if maximum_accounts is not None:
        eligible = eligible[:maximum_accounts]
    frame = frame[frame["account"].astype(str).isin(set(eligible))].copy()
    if frame.empty:
        raise ValueError("no eligible Sent140 accounts remain")
    ordered_accounts = sorted(
        eligible, key=lambda account: _rank(split_seed, "account-split", account)
    )
    n_train = int(round(train_fraction * len(ordered_accounts)))
    n_validation = int(round(validation_fraction * len(eligible)))
    account_split = {
        "train": tuple(ordered_accounts[:n_train]),
        "validation": tuple(ordered_accounts[n_train : n_train + n_validation]),
        "test": tuple(ordered_accounts[n_train + n_validation :]),
    }
    split_for_account = {
        account: split_name
        for split_name, accounts in account_split.items()
        for account in accounts
    }
    overlap_count = int(round(train_overlap_probability * len(account_split["train"])))
    overlapping = set(
        sorted(
            account_split["train"],
            key=lambda account: _rank(topology_seed, "overlap", account),
        )[:overlap_count]
    )

    records: List[PublicRecord] = []
    truth_pairs: List[Tuple[str, str]] = []
    texts: List[str] = []
    labels: List[int] = []
    splits: List[str] = []
    user_silos: MutableMapping[str, set[int]] = {
        account: set() for account in account_split["train"]
    }
    for account, part in frame.groupby(frame["account"].astype(str), sort=False):
        rows = list(part.itertuples(index=False))
        if maximum_records_per_account is not None:
            rows = sorted(
                rows,
                key=lambda row: _rank(
                    account_selection_seed,
                    "record-cap",
                    account,
                    str(row.tweet_id),
                ),
            )[:maximum_records_per_account]
        if account in overlapping and len(rows) >= 2 and n_silos >= 2:
            multiplicity = min(
                n_silos,
                len(rows),
                2
                + _rank(topology_seed, "multiplicity", account)
                % max(1, min(n_silos, len(rows)) - 1),
            )
        else:
            multiplicity = 1
        selected_silos = sorted(
            range(n_silos), key=lambda silo: _rank(topology_seed, "silo", account, silo)
        )[:multiplicity]
        ordered_rows = sorted(
            rows, key=lambda row: _rank(topology_seed, "tweet", str(row.tweet_id))
        )
        split_name = split_for_account[account]
        for position, row in enumerate(ordered_rows):
            silo = selected_silos[position % len(selected_silos)]
            record_id = f"sent140-{_opaque(split_seed, str(row.tweet_id))}"
            local_slot = f"local-{_opaque(topology_seed, account, silo)}"
            records.append(
                PublicRecord(record_id, f"silo-{silo}", local_slot, LocalKind.ATOM, 1.0)
            )
            truth_pairs.append((record_id, account))
            texts.append(str(row.text))
            labels.append(int(int(row.target) > 0))
            splits.append(split_name)
            if split_name == "train":
                user_silos[account].add(silo)
    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("tweet IDs are not unique after canonicalization")
    realized_overlap = (
        sum(len(silos) >= 2 for silos in user_silos.values()) / len(user_silos)
        if user_silos
        else 0.0
    )
    # HashingVectorizer is stateless.  It therefore neither fits a vocabulary
    # nor reads validation or test labels when constructing the public map.
    from sklearn.feature_extraction.text import HashingVectorizer

    vectorizer = HashingVectorizer(
        n_features=text_hash_dimension,
        alternate_sign=True,
        norm="l2",
        lowercase=True,
    )
    features = vectorizer.transform(texts).astype(np.float32).toarray()
    return Sent140Partition(
        records=tuple(records),
        truth=GroundTruth(tuple(truth_pairs)),
        x=features,
        texts=np.asarray(texts),
        y=np.asarray(labels, dtype=np.int64),
        account_split=account_split,
        record_split=np.asarray(splits),
        realized_train_overlap=realized_overlap,
        ambiguous_duplicate_rows_removed=ambiguous_duplicate_rows_removed,
    )
