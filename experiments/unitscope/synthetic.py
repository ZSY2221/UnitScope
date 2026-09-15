from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .model import GroundTruth, LocalKind, PublicRecord


@dataclass(frozen=True)
class SyntheticPopulation:
    records: Tuple[PublicRecord, ...]
    truth: GroundTruth
    vectors: np.ndarray
    n_users: int
    realized_overlap: float
    realized_pairwise_cosine_mean: float | None
    realized_pairwise_cosine_p10: float | None
    realized_pairwise_cosine_p50: float | None
    realized_pairwise_cosine_p90: float | None


def _token(seed: int, *parts: object) -> str:
    value = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _unit_vector(rng: np.random.Generator, dimension: int) -> np.ndarray:
    vector = rng.normal(size=dimension)
    return vector / max(np.linalg.norm(vector), 1e-12)


def make_vector_population(
    *,
    n_users: int,
    n_silos: int,
    dimension: int,
    overlap_probability: float,
    gradient_correlation: float,
    norm: float,
    seed: int,
    mixed_local_slots: bool = False,
    records_per_silo: int = 1,
) -> SyntheticPopulation:
    if not 0 <= overlap_probability <= 1:
        raise ValueError("overlap_probability must lie in [0, 1]")
    if not -1 <= gradient_correlation <= 1:
        raise ValueError("gradient_correlation must lie in [-1, 1]")
    if (
        n_users <= 0
        or n_silos <= 0
        or dimension <= 0
        or norm <= 0
        or records_per_silo <= 0
    ):
        raise ValueError("population dimensions and norm must be positive")
    if overlap_probability > 0 and n_silos > 1 and dimension < 2:
        raise ValueError("overlapping synthetic users require dimension >= 2")
    rng_topology = np.random.default_rng(seed + 101)
    rng_vectors = np.random.default_rng(seed + 202)
    records: List[PublicRecord] = []
    assignments: List[Tuple[str, str]] = []
    vectors: List[np.ndarray] = []
    overlapping = 0
    rho = gradient_correlation
    orthogonal_scale = np.sqrt(max(0.0, 1.0 - rho * rho))
    pairwise_cosines: List[float] = []
    for user_index in range(n_users):
        user_id = f"oracle-user-{user_index}"
        if rng_topology.random() < overlap_probability and n_silos > 1:
            # Exactly two institutions make the requested cross-institution
            # cosine feasible over the full [-1, 1] range.  Higher-order
            # equicorrelation matrices cannot realize arbitrary negative rho.
            multiplicity = 2
            overlapping += 1
        else:
            multiplicity = 1
        silos = rng_topology.choice(n_silos, size=multiplicity, replace=False)
        base = _unit_vector(rng_vectors, dimension)
        user_vectors = [base]
        if multiplicity == 2:
            residual = rng_vectors.normal(size=dimension)
            residual = residual - float(np.dot(residual, base)) * base
            residual_norm = np.linalg.norm(residual)
            if residual_norm <= 1e-12:
                raise RuntimeError(
                    "failed to construct an orthogonal synthetic direction"
                )
            residual = residual / residual_norm
            paired = rho * base + orthogonal_scale * residual
            paired = paired / max(np.linalg.norm(paired), 1e-12)
            user_vectors.append(paired)
            pairwise_cosines.append(float(np.dot(base, paired)))
        for vector_index, silo in enumerate(sorted(map(int, silos))):
            vector = norm * user_vectors[vector_index]
            # Multiple records may share one frozen institution-local slot.
            # This permits H-fragmentation scans without changing the
            # cross-institution cosine construction.
            local_slot = f"local-{_token(seed + 1, user_index, silo)}"
            for record_index in range(records_per_silo):
                record_id = f"r-{_token(seed, user_index, silo, record_index)}"
                records.append(
                    PublicRecord(
                        record_id=record_id,
                        silo_id=f"silo-{silo}",
                        fallback_slot=local_slot,
                        local_kind=LocalKind.ATOM
                        if not mixed_local_slots
                        else LocalKind.MIXED,
                        query_weight=1.0,
                    )
                )
                assignments.append((record_id, user_id))
                vectors.append(vector)
    order = np.argsort([record.record_id for record in records])
    ordered_records = tuple(records[index] for index in order)
    ordered_vectors = np.stack([vectors[index] for index in order])
    truth_map = dict(assignments)
    truth = GroundTruth(
        tuple(
            (record.record_id, truth_map[record.record_id])
            for record in ordered_records
        )
    )
    if pairwise_cosines:
        cosine_array = np.asarray(pairwise_cosines, dtype=np.float64)
        cosine_summary = (
            float(cosine_array.mean()),
            float(np.quantile(cosine_array, 0.10)),
            float(np.quantile(cosine_array, 0.50)),
            float(np.quantile(cosine_array, 0.90)),
        )
    else:
        cosine_summary = (None, None, None, None)
    return SyntheticPopulation(
        ordered_records,
        truth,
        ordered_vectors,
        n_users,
        overlapping / n_users,
        *cosine_summary,
    )
