from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from ssl_experiments.analysis import analyze_study
from ssl_experiments.bootstrap import paired_bootstrap, write_bootstrap
from ssl_experiments.commands import (
    build_evaluation_commands,
    build_finetune_command,
    build_pretrain_command,
)
from ssl_experiments.config import ConfigError, load_study
from ssl_experiments.matrix import (
    analyze_matrix,
    load_matrix,
    render_matrix,
    run_matrix,
)
from ssl_experiments.pipeline import Pipeline, STAGES


def comma_values(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sslx")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("validate", "plan", "status", "analyze"):
        command = subparsers.add_parser(name)
        command.add_argument("study", type=Path)
        if name == "plan":
            command.add_argument("--methods", type=comma_values)
            command.add_argument("--stages", type=comma_values, default=STAGES)

    run = subparsers.add_parser("run")
    run.add_argument("study", type=Path)
    run.add_argument("--methods", type=comma_values)
    run.add_argument("--stages", type=comma_values, default=STAGES)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--analyze", action="store_true")

    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument("left", type=Path)
    bootstrap.add_argument("right", type=Path)
    bootstrap.add_argument("--left-name", required=True)
    bootstrap.add_argument("--right-name", required=True)
    bootstrap.add_argument("--output", type=Path, required=True)
    bootstrap.add_argument("--samples", type=int, default=5000)
    bootstrap.add_argument("--seed", type=int, default=0)

    thesis = subparsers.add_parser("thesis-evaluate")
    thesis.add_argument("study", type=Path)
    thesis.add_argument("--phase", choices=("all", "calibration", "objectness"), default="all")
    thesis.add_argument("--overwrite", action="store_true")

    for name in ("matrix-plan", "matrix-status", "matrix-analyze"):
        command = subparsers.add_parser(name)
        command.add_argument("matrix", type=Path)
        command.add_argument("--regenerate-configs", action="store_true")
        if name == "matrix-plan":
            command.add_argument("--methods", type=comma_values)
            command.add_argument("--stages", type=comma_values, default=STAGES)
        if name == "matrix-analyze":
            command.add_argument("--allow-incomplete", action="store_true")

    matrix_run = subparsers.add_parser("matrix-run")
    matrix_run.add_argument("matrix", type=Path)
    matrix_run.add_argument("--methods", type=comma_values)
    matrix_run.add_argument("--stages", type=comma_values, default=STAGES)
    matrix_run.add_argument("--dry-run", action="store_true")
    matrix_run.add_argument("--analyze", action="store_true")
    matrix_run.add_argument("--regenerate-configs", action="store_true")
    return parser


def print_status(rows: list[dict[str, str]]) -> None:
    headers = ("method", "pretrain", "finetune", "evaluate")
    widths = {
        key: max(len(key), *(len(row[key]) for row in rows)) for key in headers
    }
    print("  ".join(key.ljust(widths[key]) for key in headers))
    print("  ".join("-" * widths[key] for key in headers))
    for row in rows:
        print("  ".join(row[key].ljust(widths[key]) for key in headers))


def plan(study: Path, methods: tuple[str, ...] | None, stages: tuple[str, ...]) -> None:
    config = load_study(study)
    pipeline = Pipeline(config, dry_run=True)
    selected = pipeline.selected_methods(methods)
    for method in selected:
        commands = []
        if "pretrain" in stages:
            command = build_pretrain_command(config, method)
            if command is not None:
                commands.append(command)
        if "finetune" in stages:
            commands.append(build_finetune_command(config, method))
        if "evaluate" in stages:
            commands.extend(build_evaluation_commands(config, method))
        for command in commands:
            print(f"\n[{command.method}/{command.name}]")
            print(shlex.join(command.command))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "thesis-evaluate":
            from ssl_experiments.evaluation.thesis import run_study
            print(run_study(args.study, args.phase, args.overwrite))
            return
        if args.command == "bootstrap":
            result = paired_bootstrap(
                args.left,
                args.right,
                left_name=args.left_name,
                right_name=args.right_name,
                samples=args.samples,
                seed=args.seed,
            )
            write_bootstrap(args.output, result)
            print(args.output / "bootstrap.md")
            return

        if args.command.startswith("matrix-"):
            matrix = load_matrix(args.matrix)
            if args.command == "matrix-run":
                run_matrix(
                    matrix,
                    method_names=args.methods,
                    stages=args.stages,
                    dry_run=args.dry_run,
                    regenerate=args.regenerate_configs,
                )
                if args.analyze and not args.dry_run:
                    analyze_matrix(matrix)
            elif args.command == "matrix-plan":
                for cell in render_matrix(
                    matrix,
                    regenerate=args.regenerate_configs,
                ):
                    print(f"\n=== matrix cell: {cell.name} {cell.axes} ===")
                    plan(cell.config.source, args.methods, args.stages)
            elif args.command == "matrix-status":
                for cell in render_matrix(
                    matrix,
                    regenerate=args.regenerate_configs,
                ):
                    print(f"\n[{cell.name}] {cell.axes}")
                    print_status(Pipeline(cell.config).status())
            elif args.command == "matrix-analyze":
                result = analyze_matrix(
                    matrix,
                    allow_incomplete=args.allow_incomplete,
                )
                print(f"matrix={result['matrix']} runs={len(result['rows'])}")
            return

        config = load_study(args.study)
        pipeline = Pipeline(config, dry_run=getattr(args, "dry_run", False))
        if args.command == "validate":
            pipeline.validate_paths()
            print(
                json.dumps(
                    {
                        "valid": True,
                        "study": config.name,
                        "config_hash": config.config_hash,
                        "artifact_root": str(config.artifact_root),
                        "gpu": config.gpu,
                        "methods": [method.name for method in config.methods],
                    },
                    indent=2,
                )
            )
        elif args.command == "plan":
            plan(args.study, args.methods, args.stages)
        elif args.command == "status":
            print_status(pipeline.status())
        elif args.command == "run":
            pipeline.run(method_names=args.methods, stages=args.stages)
            if args.analyze and not args.dry_run:
                analyze_study(config)
        elif args.command == "analyze":
            result = analyze_study(config)
            print(config.artifact_root / "analysis/summary.md")
            print(f"overall_success={result['overall_success']}")
    except (ConfigError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
