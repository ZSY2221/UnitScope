from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "HASHES.md"
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "figures",
    "raw",
    "processed",
    "reproduced",
}
RELEASED_LOGS = {
    Path("results/mechanism_studies/lambda_frontier/endpoints/femnist/stderr.log"),
    Path("results/mechanism_studies/lambda_frontier/endpoints/femnist/stdout.log"),
    Path("results/mechanism_studies/lambda_frontier/endpoints/synthea/stderr.log"),
    Path("results/mechanism_studies/lambda_frontier/endpoints/synthea/stdout.log"),
}


def release_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != OUTPUT
        and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
        and (
            path.suffix.lower() != ".log"
            or path.relative_to(ROOT) in RELEASED_LOGS
        )
    )


def render() -> str:
    lines = [
        "# Release file hashes",
        "",
        "Every file in the clean public tree is listed below except this registry itself.",
        "Values are SHA-256 over the complete file bytes; paths are repository-relative.",
        "",
        "| Path | SHA-256 |",
        "|---|---|",
    ]
    for path in release_files():
        relative = path.relative_to(ROOT).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"| `{relative}` | `{digest}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or verify HASHES.md.")
    parser.add_argument(
        "--check", action="store_true", help="fail if HASHES.md is stale"
    )
    args = parser.parse_args()
    expected = render()
    if args.check:
        actual = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if actual != expected:
            raise SystemExit("HASHES.md is stale; run scripts/generate_hashes.py")
        print(f"hash registry: PASS ({len(release_files())} files)")
        return
    OUTPUT.write_text(expected, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT.name} for {len(release_files())} files")


if __name__ == "__main__":
    main()
