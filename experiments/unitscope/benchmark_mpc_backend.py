from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def benchmark(
    n_units: int, dimension: int, seed: int, chunk_units: int = 512
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    scale = 1 << 20
    party_a = np.zeros(dimension, dtype=np.uint64)
    party_b = np.zeros(dimension, dtype=np.uint64)
    clear = np.zeros(dimension, dtype=np.int64)
    share_seconds = 0.0
    aggregate_seconds = 0.0
    processed = 0
    while processed < n_units:
        size = min(chunk_units, n_units - processed)
        values = rng.normal(0.0, 0.02, size=(size, dimension))
        fixed = np.rint(values * scale).astype(np.int64)
        started = time.perf_counter()
        share_a = rng.integers(
            0, np.iinfo(np.uint64).max, size=fixed.shape, dtype=np.uint64
        )
        share_b = fixed.astype(np.uint64) - share_a
        share_seconds += time.perf_counter() - started
        started = time.perf_counter()
        party_a += share_a.sum(axis=0, dtype=np.uint64)
        party_b += share_b.sum(axis=0, dtype=np.uint64)
        clear += fixed.sum(axis=0, dtype=np.int64)
        aggregate_seconds += time.perf_counter() - started
        processed += size
    started = time.perf_counter()
    reconstructed = (party_a + party_b).view(np.int64)
    reconstruction_seconds = time.perf_counter() - started
    max_fixed_error = int(np.max(np.abs(reconstructed - clear)))
    payload_bytes = 2 * n_units * dimension * 8
    response_bytes = 2 * dimension * 8
    total_bytes = payload_bytes + response_bytes
    # A declared deployment model used only for a reproducible projection.
    # The measured computation above remains actual local execution.
    lan_1gbps_seconds = total_bytes * 8 / 1e9 + 0.0005
    wan_100mbps_seconds = total_bytes * 8 / 1e8 + 0.020
    return {
        "seed": seed,
        "n_units": n_units,
        "dimension": dimension,
        "share_seconds": share_seconds,
        "party_aggregate_seconds": aggregate_seconds,
        "reconstruction_seconds": reconstruction_seconds,
        "measured_compute_seconds": share_seconds
        + aggregate_seconds
        + reconstruction_seconds,
        "payload_bytes": payload_bytes,
        "response_bytes": response_bytes,
        "total_communication_bytes": total_bytes,
        "lan_1gbps_projected_seconds": lan_1gbps_seconds,
        "wan_100mbps_projected_seconds": wan_100mbps_seconds,
        "max_fixed_point_error": max_fixed_error,
        "backend": "two-party additive sharing over uint64",
        "threat_model": "semi-honest, at most one non-colluding aggregation party",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for n_units in (1_000, 10_000, 100_000):
        for dimension in (128, 1024):
            for seed in args.seeds:
                row = benchmark(n_units, dimension, seed)
                rows.append(row)
                print(
                    json.dumps(
                        {"n_units": n_units, "dimension": dimension, "seed": seed}
                    ),
                    flush=True,
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "mpc_backend_results.csv", index=False)
    summary = frame.groupby(["n_units", "dimension"], as_index=False).agg(
        seeds=("seed", "nunique"),
        compute_seconds_mean=("measured_compute_seconds", "mean"),
        compute_seconds_std=("measured_compute_seconds", "std"),
        communication_mib=(
            "total_communication_bytes",
            lambda values: float(values.iloc[0]) / 2**20,
        ),
        lan_projected_seconds=("lan_1gbps_projected_seconds", "mean"),
        wan_projected_seconds=("wan_100mbps_projected_seconds", "mean"),
        max_fixed_point_error=("max_fixed_point_error", "max"),
    )
    summary.to_csv(args.output / "mpc_backend_summary.csv", index=False)
    manifest = {
        "seeds": args.seeds,
        "backend": "two-party additive-sharing prototype",
        "measured": "share generation, local party aggregation, reconstruction, exact communication bytes",
        "projected": "LAN and WAN transfer time from declared bandwidth/RTT",
        "excluded": "secret entity resolution, hidden access patterns, malicious security, remote TEE attestation",
        "rows": len(frame),
        "all_correct": bool((frame["max_fixed_point_error"] == 0).all()),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
