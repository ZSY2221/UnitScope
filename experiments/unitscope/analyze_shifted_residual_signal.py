"""Evaluate residual-signal detectability under the released path noise level."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def analyze() -> tuple[list[dict[str, float | bool]], dict[str, object]]:
    rows: list[dict[str, float | bool]] = []
    clip_norm = 1.0
    fallback_slots = 6
    noise_multiplier = 9.215
    allocation = 0.5
    radius = (1.0 - allocation) * clip_norm / fallback_slots
    for residual_fraction in (0.2, 0.4, 0.6):
        for zipf_exponent in (1.2, 2.0, 3.0):
            for rare_label_mass in (0.3, 0.5, 0.7, 0.9):
                signal = radius * (residual_fraction * rare_label_mass) ** 0.5
                noise = noise_multiplier * radius
                rows.append(
                    {
                        "residual_fraction": residual_fraction,
                        "zipf_exponent": zipf_exponent,
                        "rare_label_mass": rare_label_mass,
                        "lambda": allocation,
                        "residual_radius_over_C": radius,
                        "signal_proxy": signal,
                        "noise_sd_proxy": noise,
                        "snr": signal / noise,
                        "detectable": signal > noise,
                    }
                )
    best = max(rows, key=lambda row: float(row["snr"]))
    manifest: dict[str, object] = {
        "attempts": len(rows),
        "best": best,
        "detectable_any": any(bool(row["detectable"]) for row in rows),
        "construction": {
            "residual_fraction": 0.4,
            "zipf_exponent": 2.0,
            "rare_label_mass": 0.7,
            "lambda": allocation,
        },
        "c_envelope": {"handle": 0.5, "residual": 0.5, "total": 1.0},
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, manifest = analyze()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
