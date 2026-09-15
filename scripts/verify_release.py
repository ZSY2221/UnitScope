from __future__ import annotations

import os
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", ".pytest_cache", ".venv", "figures", "raw", "processed", "reproduced"}


def json_strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from json_strings(key)
            yield from json_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from json_strings(item)


def run(*args: str, temporary: Path | None = None) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if temporary is not None:
        environment["TMP"] = str(temporary)
        environment["TEMP"] = str(temporary)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True, env=environment)


def scan() -> None:
    forbidden_suffixes = {
        ".7z",
        ".bz2",
        ".dat",
        ".db",
        ".feather",
        ".gz",
        ".h5",
        ".hdf5",
        ".lzma",
        ".npy",
        ".npz",
        ".parquet",
        ".pickle",
        ".pkl",
        ".pyc",
        ".pt",
        ".pth",
        ".sqlite",
        ".tar",
        ".xz",
        ".zip",
    }
    secret = re.compile(
        r"(?i)(api[_-]?key\s*[:=]|secret\s*[:=]|password\s*[:=]|AKIA[0-9A-Z]{16})"
    )
    absolute = re.compile(
        r"(?i)([A-Z]:\\(?:Users|Program Files)\\|/home/[^/]+/|/Users/[^/]+/)"
    )
    errors: list[str] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if path.name == "__pycache__" and path.is_dir():
            errors.append(f"forbidden cache directory: {relative}")
            continue
        if path.is_file() and path.suffix.lower() == ".pyc":
            errors.append(f"forbidden data/archive file: {relative}")
            continue
        if not path.is_file() or IGNORED_PARTS.intersection(relative.parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() in forbidden_suffixes:
            errors.append(f"forbidden data/archive file: {relative}")
            continue
        if path.suffix.lower() not in {
            ".py",
            ".ps1",
            ".md",
            ".json",
            ".txt",
            ".csv",
            ".log",
            "",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if secret.search(text):
            errors.append(f"possible secret: {relative}")
        if path.suffix.lower() == ".json":
            try:
                values = json_strings(json.loads(text))
                has_absolute_path = any(absolute.search(value) for value in values)
            except json.JSONDecodeError:
                has_absolute_path = bool(absolute.search(text))
        else:
            has_absolute_path = bool(absolute.search(text))
        if has_absolute_path:
            errors.append(f"absolute host path: {relative}")
    if errors:
        raise SystemExit("\n".join(errors))


def main() -> None:
    scan()
    run("scripts/generate_hashes.py", "--check")
    with tempfile.TemporaryDirectory(prefix=".unitscope-release-", dir=ROOT) as temporary:
        temporary_path = Path(temporary)
        run(
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(temporary_path / "pytest"),
            temporary=temporary_path,
        )
        run("examples/minimal_compile.py", temporary=temporary_path)
        run(
            "-m",
            "experiments.unitscope.run_vector_grid",
            "--output",
            str(temporary_path / "vector_grid_smoke"),
            "--n-users",
            "24",
            "--dimension",
            "8",
            "--n-silos",
            "3",
            "--B",
            "3",
            "--records-per-silo",
            "1",
            "--seeds",
            "0",
            temporary=temporary_path,
        )
    print("release verification: PASS")


if __name__ == "__main__":
    main()
