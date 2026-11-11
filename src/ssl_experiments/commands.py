from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from ssl_experiments.config import MethodSpec, StudyConfig, deep_merge


SHAPE_DEFAULTS: dict[str, Any] = {
    "auxiliary_shape_probability": 0.25,
    "shape_loss_weight": 0.1,
    "shape_spatial_levels": [2, 4],
}


@dataclass(frozen=True)
class StageCommand:
    method: str
    stage: str
    name: str
    command: tuple[str, ...]
    cwd: Path
    completion: Path


def script_path(config: StudyConfig, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else config.project_root / path


def entrypoint_command(config: StudyConfig, section: Mapping[str, Any]) -> tuple[str, ...]:
    if section.get("module"):
        return str(config.python), "-m", str(section["module"])
    return str(config.python), str(script_path(config, str(section["script"])))


def resolve_project_argument_paths(
    config: StudyConfig,
    args: Mapping[str, Any],
    keys: Iterable[str],
) -> dict[str, Any]:
    resolved = dict(args)
    for key in keys:
        if key in resolved and resolved[key] is not None:
            value = Path(os.path.expandvars(str(resolved[key]))).expanduser()
            resolved[key] = str(
                value if value.is_absolute() else (config.project_root / value).resolve()
            )
    return resolved


def args_to_cli(args: Mapping[str, Any]) -> list[str]:
    command: list[str] = []
    for key, value in args.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            command.append(flag if value else "--no-" + key.replace("_", "-"))
        elif isinstance(value, (list, tuple)):
            command.append(flag)
            command.extend(str(item) for item in value)
        else:
            command.extend((flag, str(value)))
    return command


def pretrain_dir(config: StudyConfig, method: MethodSpec) -> Path:
    return config.pretraining_root / method.name


def pretrain_checkpoint(config: StudyConfig, method: MethodSpec) -> Path | None:
    if method.kind == "supervised":
        return None
    if method.kind == "custom":
        value = method.pretraining.get("checkpoint", "{pretrain_dir}/latest.pt")
        return Path(format_token(str(value), template_context(config, method)))
    section = config.merged_pretraining(method)
    return pretrain_dir(config, method) / str(section.get("checkpoint", "latest.pt"))


def finetune_dir(config: StudyConfig, method: MethodSpec) -> Path:
    return config.artifact_root / "finetuning" / method.name


def finetune_results(config: StudyConfig, method: MethodSpec) -> Path:
    return finetune_dir(config, method) / "results.csv"


def finetune_weights(config: StudyConfig, method: MethodSpec) -> Path:
    return finetune_dir(config, method) / "weights/best.pt"


def shape_evaluation_dir(config: StudyConfig, method: MethodSpec) -> Path:
    return config.artifact_root / "evaluations/shape_reliance" / method.name


def clean_evaluation_dir(config: StudyConfig, method: MethodSpec) -> Path:
    return config.artifact_root / "evaluations/clean_validation" / method.name


def template_context(config: StudyConfig, method: MethodSpec) -> dict[str, str]:
    checkpoint = pretrain_dir(config, method) / "latest.pt"
    return {
        "python": str(config.python),
        "project_root": str(config.project_root),
        "repo": str(config.project_root),
        "artifact_root": str(config.artifact_root),
        "method": method.name,
        "pretrain_dir": str(pretrain_dir(config, method)),
        "checkpoint": str(checkpoint),
        "finetune_dir": str(finetune_dir(config, method)),
        "weights": str(finetune_weights(config, method)),
        "evaluation_dir": str(shape_evaluation_dir(config, method)),
    }


def format_token(value: str, context: Mapping[str, str]) -> str:
    try:
        return value.format_map(context)
    except KeyError as error:
        raise ValueError(f"Unknown command template field: {error.args[0]}") from error


def format_command(tokens: Iterable[Any], context: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(format_token(str(token), context) for token in tokens)


def _builtin_ssl_args(kind: str) -> dict[str, Any]:
    if kind == "vicreg_global":
        return {"sampling": "global"}
    if kind == "vicreg_object":
        return {"sampling": "object"}
    if kind == "vicreg_mixed":
        return {"sampling": "mixed", "object_probability": 0.75}
    if kind == "vicreg_opposed":
        return {"sampling": "global", "view_pairing": "opposed"}
    result = {"sampling": "global", **SHAPE_DEFAULTS}
    if kind == "vicreg_shape_uniform":
        result.update(shape_region_weighting="uniform", shape_pairing="aligned")
    elif kind == "vicreg_shape_region":
        result.update(
            shape_region_weighting="contour",
            shape_region_fraction=0.25,
            shape_pairing="aligned",
        )
    elif kind == "vicreg_shape_mismatched":
        result.update(
            shape_region_weighting="contour",
            shape_region_fraction=0.25,
            shape_pairing="mismatched",
        )
    else:
        raise ValueError(f"Unsupported built-in SSL method: {kind}")
    return result


def build_pretrain_command(
    config: StudyConfig,
    method: MethodSpec,
    *,
    resume: bool = False,
) -> StageCommand | None:
    if method.kind == "supervised":
        return None
    checkpoint = pretrain_checkpoint(config, method)
    assert checkpoint is not None
    if method.kind == "custom":
        context = template_context(config, method)
        command = format_command(method.pretraining["command"], context)
        if resume and "resume_command" in method.pretraining:
            command = format_command(method.pretraining["resume_command"], context)
        return StageCommand(
            method.name, "pretrain", "pretrain", command, config.project_root, checkpoint
        )

    section = config.merged_pretraining(method)
    args = deep_merge(_builtin_ssl_args(method.kind), section.get("args", {}))
    args = resolve_project_argument_paths(
        config,
        args,
        ("images", "image_list", "labels"),
    )
    args.update(output=str(config.pretraining_root), name=method.name)
    args.setdefault("device", 0)
    if resume:
        args["resume"] = str(checkpoint)
    command = (*entrypoint_command(config, section), *args_to_cli(args))
    return StageCommand(
        method.name, "pretrain", "pretrain", tuple(command), config.project_root, checkpoint
    )


def build_finetune_command(
    config: StudyConfig,
    method: MethodSpec,
    *,
    resume: bool = False,
) -> StageCommand:
    section = config.merged_finetuning(method)
    entrypoint = entrypoint_command(config, section)
    run_dir = finetune_dir(config, method)
    if resume:
        last = run_dir / "weights/last.pt"
        args = dict(section.get("resume_args", {}))
        args.setdefault("device", 0)
        command = (*entrypoint, "--resume", str(last), *args_to_cli(args))
    else:
        args = resolve_project_argument_paths(config, section.get("args", {}), ("data",))
        args.update(project=str(config.artifact_root / "finetuning"), name=method.name)
        args.setdefault("device", 0)
        checkpoint = pretrain_checkpoint(config, method)
        if checkpoint is not None:
            args["ssl_checkpoint"] = str(checkpoint)
        command = (*entrypoint, *args_to_cli(args))
    return StageCommand(
        method.name,
        "finetune",
        "finetune",
        tuple(command),
        config.project_root,
        finetune_weights(config, method),
    )


def build_evaluation_commands(
    config: StudyConfig,
    method: MethodSpec,
) -> tuple[StageCommand, ...]:
    commands: list[StageCommand] = []
    clean = config.evaluations.get("clean_validation")
    if clean and clean.get("enabled", True):
        output = clean_evaluation_dir(config, method)
        args = resolve_project_argument_paths(
            config,
            clean.get("args", {}),
            (),
        )
        args.update(
            data=str(config.resolve_path(clean["data"])),
            output=str(output),
            checkpoint=f"{method.name}={finetune_weights(config, method)}",
        )
        args.setdefault("device", 0)
        commands.append(
            StageCommand(
                method.name,
                "evaluate",
                "clean_validation",
                (*entrypoint_command(config, clean), *args_to_cli(args)),
                config.project_root,
                output / "clean_validation.json",
            )
        )
    shape = config.evaluations.get("shape_reliance")
    if shape and shape.get("enabled", True):
        output = config.artifact_root / "evaluations/shape_reliance"
        args = dict(shape.get("args", {}))
        args.update(
            weights=str(finetune_weights(config, method)),
            labels=str(config.resolve_path(shape["labels"])),
            output=str(output),
            run_name=method.name,
        )
        args.setdefault("device", 0)
        command = (
            *entrypoint_command(config, shape),
            str(config.resolve_path(shape["images"])),
            *args_to_cli(args),
        )
        commands.append(
            StageCommand(
                method.name,
                "evaluate",
                "shape_reliance",
                tuple(command),
                config.project_root,
                shape_evaluation_dir(config, method) / "report.json",
            )
        )

    for row in config.evaluations.get("custom", []):
        context = template_context(config, method)
        name = str(row["name"])
        custom_dir = config.artifact_root / "evaluations" / name / method.name
        context = {**context, "evaluation_dir": str(custom_dir)}
        completion = Path(format_token(str(row["completion"]), context))
        commands.append(
            StageCommand(
                method.name,
                "evaluate",
                name,
                format_command(row["command"], context),
                config.project_root,
                completion,
            )
        )
    return tuple(commands)
