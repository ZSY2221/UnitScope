from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from ..model import GroundTruth, LocalKind, PublicRecord


@dataclass(frozen=True)
class WriterData:
    writer_id: str
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class FEMNISTPartition:
    records: Tuple[PublicRecord, ...]
    truth: GroundTruth
    x: np.ndarray
    y: np.ndarray
    writer_split: Mapping[str, Tuple[str, ...]]
    record_split: np.ndarray
    realized_train_overlap: float


def _rank(seed: int, *parts: object) -> int:
    text = "|".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _opaque(seed: int, *parts: object) -> str:
    text = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def load_leaf_femnist(json_paths: Sequence[Path]) -> Mapping[str, WriterData]:
    """Load LEAF FEMNIST shards without changing writer IDs."""

    writers: Dict[str, WriterData] = {}
    for path in json_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        users = payload.get("users", [])
        user_data = payload.get("user_data", {})
        for writer in users:
            if writer not in user_data:
                raise ValueError(
                    f"writer {writer} is listed but absent from user_data in {path}"
                )
            x = np.asarray(user_data[writer]["x"], dtype=np.float32)
            y = np.asarray(user_data[writer]["y"], dtype=np.int64)
            if len(x) != len(y):
                raise ValueError(f"writer {writer} has mismatched x/y lengths")
            if writer in writers:
                prior = writers[writer]
                writers[writer] = WriterData(
                    writer, np.concatenate([prior.x, x]), np.concatenate([prior.y, y])
                )
            else:
                writers[writer] = WriterData(writer, x, y)
    if not writers:
        raise ValueError("no FEMNIST writers were loaded")
    return writers


def _example_message_class():
    file_descriptor = descriptor_pb2.FileDescriptorProto(
        name="unitscope_tensorflow_example.proto", package="tensorflow", syntax="proto3"
    )

    def add_message(name: str):
        message = file_descriptor.message_type.add()
        message.name = name
        return message

    def add_field(
        message,
        name: str,
        number: int,
        field_type: int,
        label: int = 1,
        type_name: str | None = None,
    ):
        field = message.field.add(
            name=name, number=number, type=field_type, label=label
        )
        if type_name is not None:
            field.type_name = type_name
        return field

    bytes_list = add_message("BytesList")
    add_field(bytes_list, "value", 1, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES, 3)
    float_list = add_message("FloatList")
    add_field(float_list, "value", 1, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT, 3)
    int_list = add_message("Int64List")
    add_field(int_list, "value", 1, descriptor_pb2.FieldDescriptorProto.TYPE_INT64, 3)
    feature = add_message("Feature")
    feature.oneof_decl.add(name="kind")
    for name, number, type_name in (
        ("bytes_list", 1, ".tensorflow.BytesList"),
        ("float_list", 2, ".tensorflow.FloatList"),
        ("int64_list", 3, ".tensorflow.Int64List"),
    ):
        field = add_field(
            feature,
            name,
            number,
            descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            type_name=type_name,
        )
        field.oneof_index = 0
    features = add_message("Features")
    entry = features.nested_type.add(name="FeatureEntry")
    entry.options.map_entry = True
    add_field(entry, "key", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    add_field(
        entry,
        "value",
        2,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".tensorflow.Feature",
    )
    add_field(
        features,
        "feature",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        3,
        ".tensorflow.Features.FeatureEntry",
    )
    example = add_message("Example")
    add_field(
        example,
        "features",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".tensorflow.Features",
    )
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_descriptor)
    return message_factory.GetMessageClass(
        pool.FindMessageTypeByName("tensorflow.Example")
    )


_TF_EXAMPLE = _example_message_class()


def decode_tff_emnist_example(serialized: bytes) -> tuple[np.ndarray, int]:
    example = _TF_EXAMPLE()
    example.ParseFromString(serialized)
    features = example.features.feature
    if "pixels" not in features or "label" not in features:
        raise ValueError("serialized EMNIST example lacks pixels or label")
    pixels = np.asarray(features["pixels"].float_list.value, dtype=np.float32)
    labels = features["label"].int64_list.value
    if pixels.size != 28 * 28 or len(labels) != 1:
        raise ValueError("unexpected federated EMNIST example shape")
    # Losslessly quantize the public dataset values to reduce the materialized
    # cache by 4x. Training converts back to [0,1] float32.
    image = np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8).reshape(28, 28)
    return image, int(labels[0])


def load_tff_federated_emnist(
    sqlite_path: Path,
    *,
    only_digits: bool = False,
    maximum_writers: int | None = None,
    maximum_examples_per_writer: int | None = 128,
    writer_selection_seed: int = 20260714,
    writer_selection_offset: int = 0,
) -> Mapping[str, WriterData]:
    """Load a bounded writer set from TFF's official LEAF-derived database."""

    path = Path(sqlite_path)
    if not path.exists():
        raise FileNotFoundError(path)
    split_prefix = "digits_only" if only_digits else "all"
    split_names = (f"{split_prefix}_train", f"{split_prefix}_test")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        placeholders = ",".join("?" for _ in split_names)
        rows = connection.execute(
            f"SELECT DISTINCT client_id FROM client_metadata WHERE split_name IN ({placeholders})",
            split_names,
        ).fetchall()
        if writer_selection_offset < 0:
            raise ValueError("writer_selection_offset must be nonnegative")
        writer_ids = sorted(
            (str(row[0]) for row in rows),
            key=lambda writer: _rank(writer_selection_seed, "writer", writer),
        )
        if maximum_writers is not None:
            writer_ids = writer_ids[
                writer_selection_offset : writer_selection_offset + maximum_writers
            ]
        elif writer_selection_offset:
            writer_ids = writer_ids[writer_selection_offset:]
        writers: Dict[str, WriterData] = {}
        for writer in writer_ids:
            examples = connection.execute(
                f"SELECT rowid, serialized_example_proto FROM examples "
                f"WHERE client_id=? AND split_name IN ({placeholders}) ORDER BY rowid",
                (writer, *split_names),
            ).fetchall()
            if (
                maximum_examples_per_writer is not None
                and len(examples) > maximum_examples_per_writer
            ):
                examples = sorted(
                    examples,
                    key=lambda row: _rank(
                        writer_selection_seed, "example", writer, int(row[0])
                    ),
                )[:maximum_examples_per_writer]
            images, labels = zip(
                *(decode_tff_emnist_example(row[1]) for row in examples)
            )
            writers[writer] = WriterData(
                writer, np.stack(images), np.asarray(labels, dtype=np.int64)
            )
        if not writers:
            raise ValueError("no writers were loaded from the TFF database")
        return writers
    finally:
        connection.close()


def _split_writers(
    writers: Sequence[str],
    train_fraction: float,
    validation_fraction: float,
    public_auxiliary_fraction: float,
    seed: int,
) -> Mapping[str, Tuple[str, ...]]:
    if (
        not 0 < train_fraction < 1
        or not 0 <= validation_fraction < 1
        or not 0 <= public_auxiliary_fraction < 1
    ):
        raise ValueError("invalid writer split fractions")
    if train_fraction + validation_fraction + public_auxiliary_fraction >= 1:
        raise ValueError(
            "train, validation, and public auxiliary fractions must leave a test split"
        )
    ordered = sorted(writers, key=lambda writer: _rank(seed, "writer-split", writer))
    n_public = int(round(public_auxiliary_fraction * len(ordered)))
    n_train = int(round(train_fraction * len(ordered)))
    n_validation = int(round(validation_fraction * len(ordered)))
    split = {
        "public_auxiliary": tuple(ordered[:n_public]),
        "train": tuple(ordered[n_public : n_public + n_train]),
        "validation": tuple(
            ordered[n_public + n_train : n_public + n_train + n_validation]
        ),
        "test": tuple(ordered[n_public + n_train + n_validation :]),
    }
    split_names = tuple(split)
    if any(
        set(split[split_names[left]]) & set(split[split_names[right]])
        for left in range(len(split_names))
        for right in range(left + 1, len(split_names))
    ):
        raise AssertionError("writer splits must be disjoint")
    return split


def _writer_silos(
    writer: str,
    n_examples: int,
    n_silos: int,
    overlap_writer: bool,
    seed: int,
    labels: np.ndarray | None = None,
    dirichlet_alpha: float | None = None,
) -> np.ndarray:
    if n_examples <= 0:
        return np.empty(0, dtype=np.int64)
    if not overlap_writer or n_silos == 1 or n_examples == 1:
        selected = [int(_rank(seed, "primary-silo", writer) % n_silos)]
    else:
        max_multiplicity = min(n_silos, n_examples)
        multiplicity = min(
            max_multiplicity,
            2 + int(_rank(seed, "multiplicity", writer) % max(1, max_multiplicity - 1)),
        )
        selected = sorted(
            range(n_silos), key=lambda silo: _rank(seed, "silo-choice", writer, silo)
        )[:multiplicity]
    if dirichlet_alpha is not None:
        if dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")
        if labels is None or len(labels) != n_examples:
            raise ValueError(
                "label-aware Dirichlet assignment requires one label per example"
            )
        if len(selected) == 1:
            return np.full(n_examples, selected[0], dtype=np.int64)
        assignments = np.empty(n_examples, dtype=np.int64)
        for label in sorted({int(value) for value in labels}):
            indices = np.flatnonzero(np.asarray(labels) == label)
            generator = np.random.default_rng(
                _rank(seed, "dirichlet", writer, label) & ((1 << 63) - 1)
            )
            probabilities = generator.dirichlet(
                np.full(len(selected), dirichlet_alpha, dtype=np.float64)
            )
            cumulative = np.cumsum(probabilities)
            for index in indices:
                uniform = _rank(
                    seed, "dirichlet-draw", writer, label, int(index)
                ) / float(1 << 64)
                assignments[index] = selected[
                    min(int(np.searchsorted(cumulative, uniform)), len(selected) - 1)
                ]
        # The topology declares this writer as overlapping.  A deterministic
        # correction keeps that public fact true in the finite sample without
        # duplicating or dropping a record.
        if n_examples >= 2 and len(set(assignments.tolist())) == 1:
            correction = min(
                range(n_examples),
                key=lambda index: _rank(
                    seed, "dirichlet-overlap-correction", writer, index
                ),
            )
            assignments[correction] = next(
                silo for silo in selected if silo != assignments[correction]
            )
        return assignments
    example_order = sorted(
        range(n_examples), key=lambda index: _rank(seed, "example-silo", writer, index)
    )
    assignments = np.empty(n_examples, dtype=np.int64)
    for position, example_index in enumerate(example_order):
        assignments[example_index] = selected[position % len(selected)]
    return assignments


def partition_femnist_writers(
    writers: Mapping[str, WriterData],
    *,
    n_silos: int,
    train_overlap_probability: float,
    split_seed: int,
    topology_seed: int,
    train_fraction: float = 0.65,
    validation_fraction: float = 0.1,
    public_auxiliary_fraction: float = 0.05,
    dirichlet_alpha: float | None = None,
) -> FEMNISTPartition:
    """Create a writer-disjoint, no-duplication semi-synthetic silo view."""

    if n_silos <= 0 or not 0 <= train_overlap_probability <= 1:
        raise ValueError(
            "n_silos must be positive and overlap probability must lie in [0,1]"
        )
    split = _split_writers(
        sorted(writers),
        train_fraction,
        validation_fraction,
        public_auxiliary_fraction,
        split_seed,
    )
    train_writers = split["train"]
    overlap_count = int(round(train_overlap_probability * len(train_writers)))
    overlapping_train = set(
        sorted(
            train_writers, key=lambda writer: _rank(topology_seed, "overlap", writer)
        )[:overlap_count]
    )
    public_writers = split["public_auxiliary"]
    public_overlap_count = int(round(train_overlap_probability * len(public_writers)))
    overlapping_public = set(
        sorted(
            public_writers,
            key=lambda writer: _rank(topology_seed, "overlap-public", writer),
        )[:public_overlap_count]
    )

    records: List[PublicRecord] = []
    assignments: List[Tuple[str, str]] = []
    features: List[np.ndarray] = []
    labels: List[int] = []
    record_splits: List[str] = []
    for split_name in ("public_auxiliary", "train", "validation", "test"):
        for writer in split[split_name]:
            data = writers[writer]
            overlap_writer = (
                split_name == "train" and writer in overlapping_train
            ) or (split_name == "public_auxiliary" and writer in overlapping_public)
            silos = _writer_silos(
                writer,
                len(data.y),
                n_silos,
                overlap_writer,
                topology_seed,
                labels=data.y,
                dirichlet_alpha=dirichlet_alpha,
            )
            for example_index, silo in enumerate(silos):
                record_id = f"femnist-{_opaque(split_seed, writer, example_index)}"
                fallback_slot = f"local-{_opaque(topology_seed, writer, int(silo))}"
                records.append(
                    PublicRecord(
                        record_id=record_id,
                        silo_id=f"silo-{int(silo)}",
                        fallback_slot=fallback_slot,
                        local_kind=LocalKind.ATOM,
                        query_weight=1.0,
                    )
                )
                assignments.append((record_id, writer))
                features.append(np.asarray(data.x[example_index], dtype=np.float32))
                labels.append(int(data.y[example_index]))
                record_splits.append(split_name)

    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise AssertionError("each FEMNIST image must occur exactly once")
    truth = GroundTruth(tuple(assignments))
    realized_overlap = 0.0
    if train_writers:
        user_silos: MutableMapping[str, set[str]] = {
            writer: set() for writer in train_writers
        }
        truth_map = truth.as_dict()
        for record, record_split in zip(records, record_splits):
            if record_split == "train":
                user_silos[truth_map[record.record_id]].add(record.silo_id)
        realized_overlap = sum(len(silos) >= 2 for silos in user_silos.values()) / len(
            user_silos
        )
    return FEMNISTPartition(
        records=tuple(records),
        truth=truth,
        x=np.stack(features),
        y=np.asarray(labels, dtype=np.int64),
        writer_split=split,
        record_split=np.asarray(record_splits),
        realized_train_overlap=realized_overlap,
    )
