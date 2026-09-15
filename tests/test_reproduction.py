from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from experiments.unitscope.reproduction import (
    DATASETS,
    _safe_extract_zip,
    dataset_path,
    extension_cells,
    main_cells,
    plan,
    reference_cells,
    write_registry,
)


def test_dataset_catalog_paths(tmp_path: Path) -> None:
    assert set(DATASETS) == {"femnist", "synthea", "sent140", "movielens1m"}
    assert dataset_path("femnist", tmp_path) == tmp_path / "femnist" / "emnist_all.sqlite"
    assert dataset_path("movielens1m", tmp_path).name == "ml-1m"


def test_zip_extraction_rejects_parent_paths(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError, match="escapes destination"):
        _safe_extract_zip(archive, tmp_path / "output")
    assert not (tmp_path / "outside.txt").exists()


def test_zip_extraction_writes_regular_files(tmp_path: Path) -> None:
    archive = tmp_path / "good.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("folder/value.txt", "ok")
    output = tmp_path / "output"
    _safe_extract_zip(archive, output)
    assert output.joinpath("folder", "value.txt").read_text() == "ok"


def test_main_plan_has_frozen_grid(tmp_path: Path) -> None:
    cells = main_cells(tmp_path, "cpu")
    assert len(cells) == 40
    assert {(cell["dataset"], cell["replicate"]) for cell in cells} == {
        (dataset, replicate)
        for dataset in ("femnist", "synthea")
        for replicate in range(20)
    }
    assert plan()["main_cells"] == 40
    assert len(extension_cells(tmp_path, "cpu")) == 50
    assert len(reference_cells(tmp_path, "cpu")) == 40


def test_registry_uses_generated_manifests(tmp_path: Path) -> None:
    output = tmp_path / "main"
    cells = []
    for dataset, replicate in (("femnist", 0), ("synthea", 0)):
        directory = output / "runs" / f"{dataset}_r00"
        directory.mkdir(parents=True)
        (directory / "learning_manifest.json").write_text(
            json.dumps({"config_hash": f"hash-{dataset}"}), encoding="utf-8"
        )
        cells.append({"dataset": dataset, "replicate": replicate, "directory": directory})
    path = write_registry(cells, output)
    registry = json.loads(path.read_text(encoding="utf-8"))
    assert registry["entries"][0]["result_directory"] == "runs/femnist_r00"
    assert registry["entries"][1]["config_hash"] == "hash-synthea"


def test_plan_command_does_not_create_outputs(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "reproduce.py", "plan"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["main_cells"] == 40
    assert payload["extension_cells"] == 50
    assert payload["reference_cells"] == 40
    assert not tmp_path.joinpath("reproduced").exists()


def test_released_nonprivate_grid_matches_paper() -> None:
    path = Path(__file__).resolve().parents[1] / "results" / "femnist_nonprivate_tuning" / "validation_grid.csv"
    rows = path.read_text(encoding="utf-8").splitlines()
    values = [round(float(row.split(",")[2]), 4) for row in rows[1:]]
    assert values == [
        0.1474,
        0.3343,
        0.5469,
        0.3723,
        0.4010,
        0.7475,
        0.4788,
        0.6022,
        0.9137,
        0.2338,
        0.4322,
        0.6131,
    ]


def test_released_registries_resolve_all_run_directories() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "results/main/registry.json": 40,
        "results/reference_methods/registry.json": 40,
        "results/learning_extensions/registry.json": 40,
        "results/movielens_task/registry.json": 1,
    }
    for relative, count in expected.items():
        path = root / relative
        registry = json.loads(path.read_text(encoding="utf-8"))
        assert len(registry["entries"]) == count
        for entry in registry["entries"]:
            directory = path.parent / entry["result_directory"]
            assert directory.joinpath("learning_manifest.json").is_file()
            assert directory.joinpath("learning_results.jsonl").is_file()


def test_validity_command_reports_missing_data(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "reproduce.py",
            "validity",
            "--paper",
            "README.md",
            "--data-root",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "validity requires prepared FEMNIST and Synthea data" in completed.stderr
    assert "python reproduce.py data --dataset femnist" in completed.stderr
    assert "python reproduce.py data --dataset synthea" in completed.stderr
