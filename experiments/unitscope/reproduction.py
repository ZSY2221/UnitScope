from __future__ import annotations

import csv
import hashlib
import json
import lzma
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .stats import paired_seed_bootstrap


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPRODUCED = ROOT / "reproduced"
METHODS = (
    "stable-local-fallback",
    "unitscope-lambda-sweep",
    "handle-only-drop-residual",
    "naive-recalibrated-2c",
)
REFERENCE_METHODS = (
    "full-identity-reference",
    "uldp-avg-weighted-oracle",
    "group-privacy-k2-person-corrected",
    "record-level-dp-sgd-reference",
)
FEMNIST_SHA256 = "239b48fb3b3d69e70406d18a0605c821d4db3c803443c64788736bcb9f79ac56"


@dataclass(frozen=True)
class Dataset:
    name: str
    url: str
    archive: str
    archive_sha256: str
    archive_bytes: int
    prepared: str
    prepared_sha256: str | None = None


DATASETS: Mapping[str, Dataset] = {
    "femnist": Dataset(
        "femnist",
        "https://storage.googleapis.com/tff-datasets-public/emnist_all.sqlite.lzma",
        "femnist/emnist_all.sqlite.lzma",
        "c3aaf91b4cf4e69d53d81627ad119eccdc878ff906f68e66063ff10f67faad48",
        170507172,
        "femnist/emnist_all.sqlite",
        FEMNIST_SHA256,
    ),
    "synthea": Dataset(
        "synthea",
        "https://github.com/synthetichealth/synthea-sample-data/raw/refs/heads/main/downloads/10k_synthea_covid19_csv.zip",
        "synthea/10k_synthea_covid19_csv.zip",
        "559757dc849f4361a328f456d2c0a20c6df72419068321c753c6be787161e937",
        56851927,
        "synthea/10k_synthea_covid19_csv",
    ),
    "sent140": Dataset(
        "sent140",
        "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip",
        "sent140/trainingandtestdata.zip",
        "004a3772c8a7ff9bbfeb875880f47f0679d93fc63e5cf9cff72d54a8a6162e57",
        81363704,
        "sent140/training.1600000.processed.noemoticon.csv",
    ),
    "movielens1m": Dataset(
        "movielens1m",
        "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "movielens1m/ml-1m.zip",
        "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
        5917549,
        "movielens1m/ml-1m",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(source: Dataset, destination: Path, max_bytes: int) -> None:
    if destination.exists() and sha256(destination) == source.archive_sha256:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(source.url, headers={"User-Agent": "UnitScope/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > max_bytes:
            raise ValueError(f"download exceeds limit: {source.name}")
        total = 0
        while block := response.read(1024 * 1024):
            total += len(block)
            if total > max_bytes:
                raise ValueError(f"download exceeds limit: {source.name}")
            output.write(block)
    observed = sha256(partial)
    if observed != source.archive_sha256:
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 mismatch for {source.name}: expected {source.archive_sha256}, got {observed}"
        )
    os.replace(partial, destination)


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if member.filename.startswith("__MACOSX/"):
                continue
            mode = member.external_attr >> 16
            if mode & 0o170000 == 0o120000:
                raise ValueError(f"archive contains a symbolic link: {member.filename}")
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"archive path escapes destination: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def _prepare(source: Dataset, archive: Path, prepared: Path) -> None:
    if source.name == "femnist":
        if prepared.exists() and sha256(prepared) == source.prepared_sha256:
            return
        prepared.parent.mkdir(parents=True, exist_ok=True)
        temporary = prepared.with_suffix(".sqlite.part")
        with lzma.open(archive, "rb") as input_file, temporary.open("wb") as output:
            shutil.copyfileobj(input_file, output, length=1024 * 1024)
        if sha256(temporary) != source.prepared_sha256:
            temporary.unlink(missing_ok=True)
            raise ValueError("decompressed FEMNIST SHA-256 mismatch")
        os.replace(temporary, prepared)
        return
    if prepared.exists():
        return
    _safe_extract_zip(archive, archive.parent)
    if not prepared.exists():
        raise FileNotFoundError(f"archive did not contain {prepared.name}")


def prepare_datasets(names: Sequence[str], data_root: Path = RAW, max_gib: float = 5.0) -> Path:
    data_root.mkdir(parents=True, exist_ok=True)
    include_febrl = "all" in names or "febrl" in names
    selected = tuple(DATASETS) if "all" in names else tuple(name for name in names if name != "febrl")
    unknown = set(selected) - set(DATASETS)
    if unknown:
        raise ValueError(f"unknown datasets: {', '.join(sorted(unknown))}")
    max_bytes = int(max_gib * 1024**3)
    for name in selected:
        source = DATASETS[name]
        archive = data_root / source.archive
        prepared = data_root / source.prepared
        print(f"prepare {name}", flush=True)
        _download(source, archive, max_bytes)
        _prepare(source, archive, prepared)
    manifest = data_root / "download_manifest.json"
    rows = []
    for path in sorted(data_root.rglob("*")):
        if path.is_file() and path != manifest and not path.name.endswith(".part"):
            rows.append(
                {
                    "path": path.relative_to(data_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if include_febrl:
        subprocess.run(
            [
                sys.executable,
                "adapters/per2025/prepare_febrl.py",
                "--output",
                str(ROOT / "data" / "processed" / "febrl"),
            ],
            cwd=ROOT,
            check=True,
        )
    return manifest


def dataset_path(name: str, data_root: Path = RAW) -> Path:
    return data_root / DATASETS[name].prepared


def _base_learning_args(device: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "experiments.unitscope.run_learning",
        "--methods",
        *METHODS,
        "--n-silos",
        "6",
        "--H",
        "1",
        "--B",
        "6",
        "--lambda-value",
        "0.9",
        "--lambda-selection-source",
        "public-policy",
        "--lambda-policy-id",
        "fixed-allocation",
        "--clip-norm",
        "1",
        "--public-normalizer",
        "512",
        "--epsilon",
        "4",
        "--delta",
        "0.00001",
        "--rounds",
        "50",
        "--records-per-round",
        "512",
        "--grad-chunk",
        "8",
        "--learning-rate",
        "1",
        "--device",
        device,
    ]


def _dataset_args(dataset: str, replicate: int) -> list[str]:
    if dataset == "femnist":
        return [
            "--dataset",
            "femnist",
            "--data",
            str(dataset_path("femnist")),
            "--data-fingerprint",
            FEMNIST_SHA256,
            "--lambda-selection-record",
            str(ROOT / "configs" / "FIXED_ALLOCATION_FEMNIST.json"),
            "--scenario",
            "controlled-middle",
            "--overlap",
            "0.5",
            "--q-user",
            "0.5",
            "--q-record-given-handle",
            "0.5",
            "--evidence-view",
            "controlled",
            "--only-digits",
            "--maximum-writers",
            "1000",
            "--maximum-examples-per-writer",
            "32",
            "--writer-selection-seed",
            "31001",
            "--writer-selection-offset",
            "1000",
        ]
    if dataset == "synthea":
        return [
            "--dataset",
            "synthea",
            "--data",
            str(dataset_path("synthea")),
            "--data-fingerprint",
            DATASETS["synthea"].archive_sha256,
            "--lambda-selection-record",
            str(ROOT / "configs" / "FIXED_ALLOCATION_SYNTHEA.json"),
            "--scenario",
            "policy-grounded",
            "--evidence-view",
            "health-card",
            "--issuance-date",
            "2019-07-01",
            "--health-card-domains",
            "1",
            "--index-date",
            "2020-01-01",
            "--lookback-days",
            "365",
            "--horizon-days",
            "365",
        ]
    raise ValueError(dataset)


def main_cells(output: Path = REPRODUCED / "main", device: str = "cuda") -> list[dict]:
    cells = []
    for replicate in range(20):
        for dataset in ("femnist", "synthea"):
            directory = output / "runs" / f"{dataset}_r{replicate:02d}"
            command = _base_learning_args(device) + [
                "--seeds",
                str(100 + replicate),
                "--split-seed",
                str(51001 + replicate),
                "--topology-seed",
                str(61001 + replicate),
                "--evidence-seed",
                str(71001 + replicate),
                "--schedule-seed",
                "81001",
                "--noise-seed",
                "91001",
                "--output",
                str(directory),
                *_dataset_args(dataset, replicate),
            ]
            cells.append(
                {
                    "dataset": dataset,
                    "replicate": replicate,
                    "directory": directory,
                    "command": command,
                }
            )
    return cells


def _run(command: Sequence[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def write_registry(cells: Sequence[Mapping[str, object]], output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for cell in cells:
        directory = Path(cell["directory"])
        manifest = json.loads(
            (directory / "learning_manifest.json").read_text(encoding="utf-8")
        )
        entries.append(
            {
                "dataset": cell["dataset"],
                "replicate": cell["replicate"],
                "result_directory": os.path.relpath(directory, output).replace("\\", "/"),
                "config_hash": manifest["config_hash"],
            }
        )
    registry = {
        "profile": "fixed-allocation-lambda-0.9",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "noninferiority_margin": 0.01,
        "entries": entries,
    }
    path = output / "registry.json"
    path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return path


def run_main(output: Path = REPRODUCED / "main", device: str = "cuda") -> Path:
    missing = [name for name in ("femnist", "synthea") if not dataset_path(name).exists()]
    if missing:
        raise FileNotFoundError(f"prepare these datasets first: {', '.join(missing)}")
    cells = main_cells(output, device)
    for index, cell in enumerate(cells, 1):
        print(f"main {index}/{len(cells)}: {cell['dataset']} r{cell['replicate']:02d}")
        _run(cell["command"])
    registry = write_registry(cells, output)
    _run(
        [
            sys.executable,
            "-m",
            "experiments.unitscope.analyze_fixed_allocation",
            "--registry",
            str(registry),
            "--output",
            str(output / "analysis"),
        ]
    )
    return registry


def reference_cells(
    output: Path = REPRODUCED / "references", device: str = "cuda"
) -> list[dict]:
    cells = []
    for replicate in range(20):
        for dataset in ("femnist", "synthea"):
            directory = output / "runs" / f"{dataset}_r{replicate:02d}"
            command = _base_learning_args(device)
            start = command.index("--methods") + 1
            end = command.index("--n-silos")
            command[start:end] = REFERENCE_METHODS
            command.extend(
                [
                    "--seeds",
                    str(100 + replicate),
                    "--split-seed",
                    str(51001 + replicate),
                    "--topology-seed",
                    str(61001 + replicate),
                    "--evidence-seed",
                    str(71001 + replicate),
                    "--schedule-seed",
                    "81001",
                    "--noise-seed",
                    "91001",
                    "--output",
                    str(directory),
                    *_dataset_args(dataset, replicate),
                ]
            )
            cells.append(
                {
                    "dataset": dataset,
                    "replicate": replicate,
                    "directory": directory,
                    "command": command,
                }
            )
    return cells


def run_references(
    output: Path = REPRODUCED / "references",
    device: str = "cuda",
    main_registry: Path = REPRODUCED / "main" / "registry.json",
) -> Path:
    if not main_registry.exists():
        raise FileNotFoundError("run the main experiment before the reference methods")
    missing = [name for name in ("femnist", "synthea") if not dataset_path(name).exists()]
    if missing:
        raise FileNotFoundError(f"prepare these datasets first: {', '.join(missing)}")
    cells = reference_cells(output, device)
    for index, cell in enumerate(cells, 1):
        print(f"references {index}/{len(cells)}: {cell['dataset']} r{cell['replicate']:02d}")
        _run(cell["command"])
    entries = []
    for cell in cells:
        manifest = json.loads(
            (cell["directory"] / "learning_manifest.json").read_text(encoding="utf-8")
        )
        entries.append(
            {
                "dataset": cell["dataset"],
                "replicate": cell["replicate"],
                "result_directory": os.path.relpath(cell["directory"], output).replace("\\", "/"),
                "config_hash": manifest["config_hash"],
            }
        )
    registry = output / "registry.json"
    registry.write_text(json.dumps({"entries": entries}, indent=2) + "\n", encoding="utf-8")
    _run(
        [
            sys.executable,
            "-m",
            "experiments.unitscope.analyze_reference_methods",
            "--registry",
            str(registry),
            "--main-registry",
            str(main_registry),
            "--output",
            str(output / "analysis"),
        ]
    )
    return registry


def _replace_option(command: list[str], option: str, value: str) -> None:
    command[command.index(option) + 1] = value


def extension_cells(
    output: Path = REPRODUCED / "extensions", device: str = "cuda"
) -> list[dict]:
    cells = []
    families = []
    for replicate in range(10):
        for dataset in ("femnist", "synthea"):
            families.append(
                ("credible-accuracy-epsilon8-v1", dataset, replicate, None)
            )
    for alpha in (0.1, 0.5):
        for replicate in range(5):
            families.append(("user-noniid-dirichlet-v1", "femnist", replicate, alpha))
    for replicate in range(10):
        families.append(("sent140-account-task-v1", "sent140", replicate, None))
        families.append(("movielens-balanced-task", "movielens1m", replicate, None))

    for family, dataset, replicate, alpha in families:
        methods = list(METHODS)
        if dataset in {"sent140", "movielens1m"}:
            methods.append("full-identity-reference")
        command = _base_learning_args(device)
        start = command.index("--methods") + 1
        end = command.index("--n-silos")
        command[start:end] = methods
        if family == "credible-accuracy-epsilon8-v1":
            seed = 120 + replicate
            split, topology, evidence = 52001, 62001, 72001
            schedule, noise = 82001, 92001
            _replace_option(command, "--epsilon", "8")
        elif family == "user-noniid-dirichlet-v1":
            seed = 200 + replicate
            split, topology, evidence = 53001, 63001, 73001
            schedule, noise = 83001, 93001
        elif family == "sent140-account-task-v1":
            seed = 300 + replicate
            split, topology, evidence = 54001, 64001, 74001
            schedule, noise = 84001, 94001
        else:
            seed = 410 + replicate
            split, topology, evidence = 55001, 65001, 75001
            schedule, noise = 85001, 95001
        suffix = f"_alpha{str(alpha).replace('.', 'p')}" if alpha is not None else ""
        directory = output / "runs" / f"{family}_{dataset}{suffix}_r{replicate:02d}"
        command.extend(
            [
                "--seeds",
                str(seed),
                "--split-seed",
                str(split + replicate),
                "--topology-seed",
                str(topology + replicate),
                "--evidence-seed",
                str(evidence + replicate),
                "--schedule-seed",
                str(schedule),
                "--noise-seed",
                str(noise),
                "--output",
                str(directory),
            ]
        )
        if dataset in {"femnist", "synthea"}:
            command.extend(_dataset_args(dataset, replicate))
            if alpha is not None:
                command.extend(["--dirichlet-alpha", str(alpha)])
        elif dataset == "sent140":
            command.extend(
                [
                    "--dataset",
                    "sent140",
                    "--data",
                    str(dataset_path("sent140")),
                    "--data-fingerprint",
                    DATASETS["sent140"].archive_sha256,
                    "--lambda-selection-record",
                    str(ROOT / "configs" / "FIXED_ALLOCATION_SENT140.json"),
                    "--scenario",
                    "controlled-middle",
                    "--overlap",
                    "0.5",
                    "--q-user",
                    "0.5",
                    "--q-record-given-handle",
                    "0.5",
                    "--evidence-view",
                    "controlled",
                    "--minimum-records-per-account",
                    "2",
                    "--maximum-accounts",
                    "2000",
                    "--maximum-records-per-account",
                    "32",
                    "--account-selection-seed",
                    "31002",
                    "--text-hash-dimension",
                    "512",
                ]
            )
        else:
            command.extend(
                [
                    "--dataset",
                    "movielens1m",
                    "--data",
                    str(dataset_path("movielens1m")),
                    "--data-fingerprint",
                    DATASETS["movielens1m"].archive_sha256,
                    "--lambda-selection-record",
                    str(ROOT / "configs" / "FIXED_ALLOCATION_MOVIELENS.json"),
                    "--scenario",
                    "controlled-middle",
                    "--overlap",
                    "0.5",
                    "--q-user",
                    "0.5",
                    "--q-record-given-handle",
                    "0.5",
                    "--evidence-view",
                    "controlled",
                    "--maximum-accounts",
                    "3000",
                    "--maximum-records-per-account",
                    "24",
                    "--account-selection-seed",
                    "37102",
                    "--movielens-task-variant",
                    "balanced-v1",
                    "--balance-seed",
                    "37101",
                ]
            )
        cells.append(
            {
                "family": family,
                "dataset": dataset,
                "replicate": replicate,
                "dirichlet_alpha": alpha,
                "methods": methods,
                "directory": directory,
                "command": command,
            }
        )
    return cells


def run_extensions(
    output: Path = REPRODUCED / "extensions", device: str = "cuda"
) -> Path:
    missing = [name for name in DATASETS if not dataset_path(name).exists()]
    if missing:
        raise FileNotFoundError(f"prepare these datasets first: {', '.join(missing)}")
    cells = extension_cells(output, device)
    for index, cell in enumerate(cells, 1):
        print(f"extensions {index}/{len(cells)}: {cell['family']} r{cell['replicate']:02d}")
        _run(cell["command"])
    entries = []
    for cell in cells:
        manifest = json.loads(
            (cell["directory"] / "learning_manifest.json").read_text(encoding="utf-8")
        )
        entries.append(
            {
                "family": cell["family"],
                "dataset": cell["dataset"],
                "replicate": cell["replicate"],
                "dirichlet_alpha": cell["dirichlet_alpha"],
                "methods": cell["methods"],
                "result_directory": os.path.relpath(cell["directory"], output).replace("\\", "/"),
                "config_hash": manifest["config_hash"],
            }
        )
    registry = output / "registry.json"
    registry.write_text(json.dumps({"entries": entries}, indent=2) + "\n", encoding="utf-8")
    _run(
        [
            sys.executable,
            "-m",
            "experiments.unitscope.analyze_learning_extensions",
            "--registries",
            str(registry),
            "--output",
            str(output / "analysis"),
        ]
    )
    return registry


def _nonprivate_args(seed: int, rounds: int, learning_rate: float, output: Path) -> list[str]:
    replicate = seed - 750
    return [
        sys.executable,
        "-m",
        "experiments.unitscope.run_learning",
        "--dataset",
        "femnist",
        "--data",
        str(dataset_path("femnist")),
        "--data-fingerprint",
        FEMNIST_SHA256,
        "--methods",
        "non-private-reference",
        "--seeds",
        str(seed),
        "--split-seed",
        str(51001 + replicate),
        "--topology-seed",
        str(61001 + replicate),
        "--evidence-seed",
        str(71001 + replicate),
        "--schedule-seed",
        "88001",
        "--noise-seed",
        "98001",
        "--n-silos",
        "6",
        "--overlap",
        "0.5",
        "--q-user",
        "0.5",
        "--q-record-given-handle",
        "0.5",
        "--evidence-view",
        "controlled",
        "--H",
        "1",
        "--B",
        "6",
        "--lambda-value",
        "0.9",
        "--lambda-selection-source",
        "public-policy",
        "--lambda-policy-id",
        "femnist-nonprivate-capacity-reference",
        "--public-normalizer",
        "512",
        "--epsilon",
        "4",
        "--delta",
        "0.00001",
        "--rounds",
        str(rounds),
        "--records-per-round",
        "512",
        "--grad-chunk",
        "8",
        "--learning-rate",
        str(learning_rate),
        "--device",
        "cuda",
        "--scenario",
        "controlled-middle",
        "--only-digits",
        "--maximum-writers",
        "1000",
        "--maximum-examples-per-writer",
        "32",
        "--writer-selection-seed",
        "31001",
        "--writer-selection-offset",
        "1000",
        "--output",
        str(output),
    ]


def _read_one_result(directory: Path) -> dict:
    rows = [
        json.loads(line)
        for line in (directory / "learning_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one result in {directory}")
    return rows[0]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_nonprivate_results(source: Path, output: Path) -> tuple[float, int]:
    validation = []
    test = []
    for result_path in sorted(source.rglob("learning_results.jsonl")):
        row = _read_one_result(result_path.parent)
        parts = result_path.relative_to(source).parts
        tag = parts[1]
        learning_rate = float(tag.split("_rounds_")[0].removeprefix("lr_").replace("p", "."))
        rounds = int(tag.split("_rounds_")[1])
        selected = validation if parts[0] == "validation" else test
        selected.append(
            {
                "learning_rate": learning_rate,
                "rounds": rounds,
                "seed": row["seed"],
                "split_seed": row["split_seed"],
                "metric_value": row["metric_value"],
                "config_hash": row["config_hash"],
                "run_id": row["run_id"],
                "manifest_hash": row["manifest_hash"],
            }
        )
    if len(validation) != 60 or len(test) != 5:
        raise ValueError(f"expected 60 validation and 5 test runs, got {len(validation)} and {len(test)}")
    grid = []
    for learning_rate in (0.05, 0.1, 0.25, 0.5):
        for rounds in (50, 100, 200):
            values = [
                float(row["metric_value"])
                for row in validation
                if row["learning_rate"] == learning_rate and row["rounds"] == rounds
            ]
            grid.append(
                {
                    "learning_rate": learning_rate,
                    "rounds": rounds,
                    "mean_validation_user_macro_accuracy": sum(values) / len(values),
                    "replicates": len(values),
                }
            )
    best = min(grid, key=lambda row: (-row["mean_validation_user_macro_accuracy"], row["rounds"], row["learning_rate"]))
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "validation_runs.csv", validation)
    _write_csv(output / "validation_grid.csv", grid)
    _write_csv(output / "test_runs.csv", test)
    test_values = [float(row["metric_value"]) for row in test]
    interval = paired_seed_bootstrap(test_values, seed=20260717)
    manifest = {
        "dataset": "femnist",
        "method": "non-private-reference",
        "selection_split": "validation",
        "selection_rule": "highest mean; ties use fewer rounds, then lower learning rate",
        "selected_learning_rate": best["learning_rate"],
        "selected_rounds": best["rounds"],
        "test_mean_user_macro_accuracy": interval.estimate,
        "test_bootstrap_95_interval": [interval.lower, interval.upper],
        "validation_runs": len(validation),
        "test_runs": len(test),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return float(best["learning_rate"]), int(best["rounds"])


def run_nonprivate(output: Path = REPRODUCED / "femnist_nonprivate", device: str = "cuda") -> Path:
    if not dataset_path("femnist").exists():
        raise FileNotFoundError("prepare FEMNIST first")
    for learning_rate in (0.05, 0.1, 0.25, 0.5):
        for rounds in (50, 100, 200):
            for seed in range(750, 755):
                tag = f"lr_{str(learning_rate).replace('.', 'p')}_rounds_{rounds}"
                directory = output / "runs" / "validation" / tag / f"r{seed}"
                command = _nonprivate_args(seed, rounds, learning_rate, directory)
                command.extend(["--validation-only"])
                command[command.index("cuda")] = device
                _run(command)
    validation_source = output / "runs"
    grid_rows = []
    for path in sorted((validation_source / "validation").rglob("learning_results.jsonl")):
        row = _read_one_result(path.parent)
        tag = path.relative_to(validation_source).parts[1]
        lr = float(tag.split("_rounds_")[0].removeprefix("lr_").replace("p", "."))
        rounds = int(tag.split("_rounds_")[1])
        grid_rows.append((lr, rounds, float(row["metric_value"])))
    means = {
        (lr, rounds): sum(value for row_lr, row_rounds, value in grid_rows if row_lr == lr and row_rounds == rounds) / 5
        for lr in (0.05, 0.1, 0.25, 0.5)
        for rounds in (50, 100, 200)
    }
    learning_rate, rounds = min(means, key=lambda key: (-means[key], key[1], key[0]))
    for seed in range(750, 755):
        tag = f"lr_{str(learning_rate).replace('.', 'p')}_rounds_{rounds}"
        directory = output / "runs" / "test" / tag / f"r{seed}"
        command = _nonprivate_args(seed, rounds, learning_rate, directory)
        command[command.index("cuda")] = device
        _run(command)
    collect_nonprivate_results(output / "runs", output / "analysis")
    return output / "analysis" / "manifest.json"


def plan() -> dict:
    return {
        "datasets": [*DATASETS, "febrl"],
        "main_cells": len(main_cells()),
        "main_methods": list(METHODS),
        "reference_cells": len(reference_cells()),
        "reference_methods": list(REFERENCE_METHODS),
        "extension_cells": len(extension_cells()),
        "nonprivate_validation_cells": 60,
        "nonprivate_test_cells": 5,
        "outputs": {
            "main": "reproduced/main",
            "extensions": "reproduced/extensions",
            "references": "reproduced/references",
            "nonprivate": "reproduced/femnist_nonprivate",
            "figures": "figures",
        },
    }
