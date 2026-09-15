from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from experiments.unitscope.reproduction import (
    DATASETS,
    REPRODUCED,
    ROOT,
    collect_nonprivate_results,
    plan,
    prepare_datasets,
    run_extensions,
    run_main,
    run_nonprivate,
    run_references,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the UnitScope experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data = subparsers.add_parser("data", help="download and prepare the public datasets")
    data.add_argument("--dataset", choices=(*DATASETS, "febrl", "all"), default="all")
    data.add_argument("--max-gib", type=float, default=5.0)

    subparsers.add_parser("plan", help="show the experiment counts without running them")
    subparsers.add_parser("smoke", help="run the release checks and CPU smoke experiment")

    analyze = subparsers.add_parser("analyze", help="recompute tables from the released run directories")
    analyze.add_argument("--output", type=Path, default=REPRODUCED / "released_analysis")

    main = subparsers.add_parser("main", help="run and analyze the 40-cell main experiment")
    main.add_argument("--device", default="cuda")
    main.add_argument("--output", type=Path, default=REPRODUCED / "main")

    extensions = subparsers.add_parser("extensions", help="run the supplementary learning experiments")
    extensions.add_argument("--device", default="cuda")
    extensions.add_argument("--output", type=Path, default=REPRODUCED / "extensions")

    references = subparsers.add_parser("references", help="run the four paper reference methods")
    references.add_argument("--device", default="cuda")
    references.add_argument("--output", type=Path, default=REPRODUCED / "references")
    references.add_argument("--main-registry", type=Path, default=REPRODUCED / "main" / "registry.json")

    nonprivate = subparsers.add_parser("nonprivate", help="run the FEMNIST capacity grid")
    nonprivate.add_argument("--device", default="cuda")
    nonprivate.add_argument("--output", type=Path, default=REPRODUCED / "femnist_nonprivate")

    lambda_grid = subparsers.add_parser(
        "lambda-grid", help="recompute the lambda frontier from released runs"
    )
    lambda_grid.add_argument("--output", type=Path, default=REPRODUCED / "lambda_frontier")

    evidence_grid = subparsers.add_parser(
        "evidence-grid", help="recompute the evidence grid from released runs"
    )
    evidence_grid.add_argument("--output", type=Path, default=REPRODUCED / "evidence_grid")

    adaptive = subparsers.add_parser(
        "adaptive", help="recompute the adaptive allocation confirmation"
    )
    adaptive.add_argument("--output", type=Path, default=REPRODUCED / "adaptive_allocation")

    validity = subparsers.add_parser(
        "validity", help="optionally check released results against supplied paper text"
    )
    validity.add_argument(
        "--paper",
        type=Path,
        required=True,
        help="path to manuscript text already available to the person running the check",
    )
    validity.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw")
    validity.add_argument("--main-registry", type=Path, default=ROOT / "results" / "main" / "registry.json")
    validity.add_argument("--reference-registry", type=Path, default=ROOT / "results" / "reference_methods" / "registry.json")
    validity.add_argument("--output", type=Path, default=REPRODUCED / "validity")

    figures = subparsers.add_parser("figures", help="rebuild figures from result tables")
    figures.add_argument("--results-root", type=Path, default=ROOT / "results")
    figures.add_argument("--fallback-results-root", type=Path)
    figures.add_argument("--output", type=Path, default=ROOT / "figures")
    all_parser = subparsers.add_parser(
        "all", help="prepare data, run the learning experiments, and plot"
    )
    all_parser.add_argument("--device", default="cuda")
    all_parser.add_argument("--max-gib", type=float, default=5.0)
    all_parser.add_argument(
        "--paper",
        type=Path,
        help="optionally run the paper-code consistency check after training",
    )
    return parser.parse_args()


def run_script(path: str, *arguments: str) -> None:
    subprocess.run([sys.executable, path, *arguments], cwd=ROOT, check=True)


def run_figures(
    results_root: Path,
    output: Path,
    fallback_results_root: Path | None = None,
) -> None:
    arguments = [
        "--results-root",
        str(results_root.resolve()),
        "--figures-root",
        str(output.resolve()),
    ]
    if fallback_results_root is not None:
        arguments.extend(
            ["--fallback-results-root", str(fallback_results_root.resolve())]
        )
    run_script("plotting/plot_paper_figures.py", *arguments)


def released_inputs(pattern: str) -> list[Path]:
    paths = sorted(ROOT.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no released inputs match {pattern}")
    return paths


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        print(json.dumps(plan(), indent=2))
    elif args.command == "data":
        print(prepare_datasets([args.dataset], max_gib=args.max_gib))
    elif args.command == "smoke":
        run_script("scripts/verify_release.py")
    elif args.command == "analyze":
        output = args.output.resolve()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.unitscope.analyze_fixed_allocation",
                "--registry",
                "results/main/registry.json",
                "--output",
                str(output / "main"),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.unitscope.analyze_reference_methods",
                "--registry",
                "results/reference_methods/registry.json",
                "--main-registry",
                "results/main/registry.json",
                "--output",
                str(output / "references"),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.unitscope.analyze_learning_extensions",
                "--registries",
                "results/learning_extensions/registry.json",
                "results/movielens_task/registry.json",
                "--output",
                str(output / "extensions"),
            ],
            cwd=ROOT,
            check=True,
        )
        collect_nonprivate_results(
            ROOT / "results" / "femnist_nonprivate_tuning" / "runs",
            output / "femnist_nonprivate_tuning",
        )
    elif args.command == "main":
        print(run_main(args.output.resolve(), args.device))
    elif args.command == "extensions":
        print(run_extensions(args.output.resolve(), args.device))
    elif args.command == "references":
        print(run_references(args.output.resolve(), args.device, args.main_registry.resolve()))
    elif args.command == "nonprivate":
        print(run_nonprivate(args.output.resolve(), args.device))
    elif args.command == "lambda-grid":
        main_inputs = released_inputs(
            "results/mechanism_studies/lambda_frontier/endpoints/*/learning_results.jsonl"
        )
        sweep_inputs = released_inputs(
            "results/mechanism_studies/lambda_frontier/runs/*/learning_results.jsonl"
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.unitscope.analyze_lambda_frontier",
                "--main-inputs",
                *(str(path) for path in main_inputs),
                "--sweep-inputs",
                *(str(path) for path in sweep_inputs),
                "--output",
                str(args.output.resolve()),
            ],
            cwd=ROOT,
            check=True,
        )
    elif args.command == "evidence-grid":
        main_inputs = released_inputs(
            "results/mechanism_studies/lambda_frontier/endpoints/*/learning_results.jsonl"
        )
        grid_inputs = released_inputs(
            "results/mechanism_studies/evidence_grid/runs/*/learning_results.jsonl"
        )
        central_sweep = (
            ROOT
            / "results"
            / "mechanism_studies"
            / "lambda_frontier"
            / "runs"
            / "femnist_lambda_090"
            / "learning_results.jsonl"
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.unitscope.analyze_evidence_grid",
                "--grid-inputs",
                *(str(path) for path in grid_inputs),
                "--main-inputs",
                *(str(path) for path in main_inputs),
                "--central-sweep-input",
                str(central_sweep),
                "--output",
                str(args.output.resolve()),
            ],
            cwd=ROOT,
            check=True,
        )
    elif args.command == "adaptive":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.unitscope.analyze_adaptive_allocation",
                "--runs",
                str(
                    ROOT
                    / "results"
                    / "adaptive_allocation"
                    / "runs"
                    / "confirmation_cuda_v1"
                ),
                "--output",
                str(args.output.resolve()),
            ],
            cwd=ROOT,
            check=True,
        )
    elif args.command == "validity":
        data_root = args.data_root.resolve()
        required = (
            data_root / "femnist" / "emnist_all.sqlite",
            data_root / "synthea" / "10k_synthea_covid19_csv",
        )
        missing = [path for path in required if not path.exists()]
        if missing:
            paths = "\n".join(f"  {path}" for path in missing)
            raise SystemExit(
                "validity requires prepared FEMNIST and Synthea data.\n"
                "Run: python reproduce.py data --dataset femnist\n"
                "     python reproduce.py data --dataset synthea\n"
                f"Missing:\n{paths}"
            )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.unitscope.audit_experimental_validity",
                "--main-registry",
                str(args.main_registry.resolve()),
                "--reference-registry",
                str(args.reference_registry.resolve()),
                "--paper",
                str(args.paper.resolve()),
                "--data-root",
                str(data_root),
                "--output",
                str(args.output.resolve()),
            ],
            cwd=ROOT,
            check=True,
        )
    elif args.command == "figures":
        run_figures(args.results_root, args.output, args.fallback_results_root)
    elif args.command == "all":
        print(
            "all runs the main, reference, extension, MovieLens, and non-private "
            "learning experiments. Mechanism, linkage, allocation, and system studies "
            "are not rerun by this command; their released tables are used when plotting."
        )
        prepare_datasets(["all"], max_gib=args.max_gib)
        run_main(device=args.device)
        run_references(device=args.device)
        run_extensions(device=args.device)
        run_nonprivate(device=args.device)
        if args.paper:
            subprocess.run(
                [
                    sys.executable,
                    "reproduce.py",
                    "validity",
                    "--paper",
                    str(args.paper.resolve()),
                    "--main-registry",
                    str(REPRODUCED / "main" / "registry.json"),
                    "--reference-registry",
                    str(REPRODUCED / "references" / "registry.json"),
                ],
                cwd=ROOT,
                check=True,
            )
        run_figures(
            REPRODUCED,
            REPRODUCED / "figures",
            fallback_results_root=ROOT / "results",
        )


if __name__ == "__main__":
    main()
