from __future__ import annotations

import csv
import json
import os
import subprocess
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterable, Iterator

from ssl_experiments.commands import (
    StageCommand,
    build_evaluation_commands,
    build_finetune_command,
    build_pretrain_command,
    finetune_dir,
    finetune_results,
    finetune_weights,
    pretrain_checkpoint,
    pretrain_dir,
    resolve_project_argument_paths,
    script_path,
)
from ssl_experiments.config import MethodSpec, StudyConfig
from ssl_experiments.runner import CommandRunner, atomic_json, utc_now


STAGES = ("pretrain", "finetune", "evaluate")


def jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


class Pipeline:
    def __init__(self, config: StudyConfig, *, dry_run: bool = False) -> None:
        self.config = config
        self.runner = CommandRunner(gpu=config.gpu, dry_run=dry_run)
        self.dry_run = dry_run

    def state_path(self, command: StageCommand) -> Path:
        return (
            self.config.artifact_root
            / "state"
            / command.method
            / f"{command.name}.json"
        )

    def log_path(self, command: StageCommand) -> Path:
        return (
            self.config.artifact_root
            / "logs"
            / command.method
            / f"{command.name}.log"
        )

    def marker_complete(self, command: StageCommand) -> bool:
        path = self.state_path(command)
        if not path.is_file() or not command.completion.exists():
            return False
        state = json.loads(path.read_text(encoding="utf-8"))
        return state.get("status") == "complete" and state.get("config_hash") == self.config.config_hash

    def pretrain_complete(self, method: MethodSpec) -> bool:
        if method.kind == "supervised":
            return True
        command = build_pretrain_command(self.config, method)
        assert command is not None
        if self.marker_complete(command):
            return True
        checkpoint = pretrain_checkpoint(self.config, method)
        if checkpoint is None or not checkpoint.is_file():
            return False
        if method.kind == "custom":
            return False
        epochs = int(self.config.merged_pretraining(method).get("args", {}).get("epochs", 0))
        return epochs > 0 and jsonl_rows(pretrain_dir(self.config, method) / "metrics.jsonl") >= epochs

    def finetune_complete(self, method: MethodSpec) -> bool:
        command = build_finetune_command(self.config, method)
        if self.marker_complete(command):
            return True
        if not finetune_weights(self.config, method).is_file():
            return False
        epochs = int(self.config.merged_finetuning(method).get("args", {}).get("epochs", 0))
        return epochs > 0 and csv_rows(finetune_results(self.config, method)) >= epochs

    def evaluation_complete(self, method: MethodSpec) -> bool:
        commands = build_evaluation_commands(self.config, method)
        return all(
            self.marker_complete(command) or command.completion.is_file()
            for command in commands
        )

    def validate_paths(self) -> None:
        required = [self.config.project_root, self.config.python]
        required.extend(
            script_path(self.config, str(section["script"]))
            for section in (self.config.pretraining, self.config.finetuning)
            if section.get("script")
        )
        shape = self.config.evaluations.get("shape_reliance")
        if shape and shape.get("enabled", True):
            required.extend(
                [
                    self.config.resolve_path(shape["images"]),
                    self.config.resolve_path(shape["labels"]),
                ]
            )
            if shape.get("script"):
                required.append(script_path(self.config, str(shape["script"])))
        clean = self.config.evaluations.get("clean_validation")
        if clean and clean.get("enabled", True):
            required.append(self.config.resolve_path(clean["data"]))
            if clean.get("script"):
                required.append(script_path(self.config, str(clean["script"])))
        for method in self.config.methods:
            if method.kind not in {"supervised", "custom"}:
                pretrain_args = resolve_project_argument_paths(
                    self.config, self.config.merged_pretraining(method).get("args", {}),
                    ("images", "image_list", "labels"),
                )
                required.extend(Path(pretrain_args[key]) for key in ("images", "image_list", "labels")
                                if pretrain_args.get(key))
            finetune_args = resolve_project_argument_paths(
                self.config, self.config.merged_finetuning(method).get("args", {}), ("data",),
            )
            if finetune_args.get("data"):
                required.append(Path(finetune_args["data"]))
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required paths:\n" + "\n".join(map(str, missing)))

    def selected_methods(self, names: Iterable[str] | None) -> tuple[MethodSpec, ...]:
        if not names:
            return self.config.methods
        requested = tuple(names)
        unknown = sorted(set(requested) - {method.name for method in self.config.methods})
        if unknown:
            raise ValueError(f"Unknown methods: {', '.join(unknown)}")
        return tuple(self.config.method(name) for name in requested)

    def snapshot(self) -> None:
        root = self.config.artifact_root
        root.mkdir(parents=True, exist_ok=True)
        metadata_path = root / "study.json"
        if metadata_path.exists():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing.get("config_hash") != self.config.config_hash:
                raise RuntimeError(
                    f"Study configuration changed after launch: {metadata_path}"
                )
            return
        git = subprocess.run(
            ["git", "-C", str(self.config.project_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        metadata = {
            "schema_version": 1,
            "created_at": utc_now(),
            "config_hash": self.config.config_hash,
            "config_source": str(self.config.source),
            "project_root": str(self.config.project_root),
            "project_git_commit": git.stdout.strip() if git.returncode == 0 else None,
            "physical_gpu": self.config.gpu,
            "config": self.config.raw,
        }
        atomic_json(metadata_path, metadata)
        (root / "study.yaml").write_text(
            self.config.source.read_text(encoding="utf-8"), encoding="utf-8"
        )

    @contextmanager
    def lock(self) -> Iterator[None]:
        path = self.config.artifact_root / "study.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as stream:
            try:
                if os.name == "nt":
                    import msvcrt
                    if stream.seek(0, 2) == 0:
                        stream.write(b"\0")
                        stream.flush()
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError(f"Study is already locked: {path}") from error
            try:
                yield
            finally:
                if os.name == "nt":
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream, fcntl.LOCK_UN)

    def record_complete(self, command: StageCommand, started: str, finished: str) -> None:
        if self.dry_run:
            return
        if not command.completion.exists():
            raise RuntimeError(
                f"{command.method}/{command.name} returned successfully but did not create "
                f"{command.completion}"
            )
        atomic_json(
            self.state_path(command),
            {
                "status": "complete",
                "method": command.method,
                "stage": command.stage,
                "name": command.name,
                "started_at": started,
                "finished_at": finished,
                "config_hash": self.config.config_hash,
                "completion": str(command.completion),
                "command": list(command.command),
            },
        )

    def execute(self, command: StageCommand) -> None:
        if self.marker_complete(command) or (
            command.stage == "evaluate" and command.completion.is_file()
        ):
            print(f"SKIP complete: {command.method}/{command.name}")
            return
        result = self.runner.run(command, self.log_path(command))
        self.record_complete(command, result.started_at, result.finished_at)

    def run_pretrain(self, method: MethodSpec) -> None:
        if method.kind == "supervised":
            print(f"SKIP not applicable: {method.name}/pretrain")
            return
        if self.pretrain_complete(method):
            print(f"SKIP complete: {method.name}/pretrain")
            return
        checkpoint = pretrain_checkpoint(self.config, method)
        resume = bool(checkpoint and checkpoint.is_file())
        directory = pretrain_dir(self.config, method)
        if directory.exists() and any(directory.iterdir()) and not resume:
            raise RuntimeError(f"Partial pretraining has no checkpoint: {directory}")
        command = build_pretrain_command(self.config, method, resume=resume)
        assert command is not None
        self.execute(command)

    def run_finetune(self, method: MethodSpec) -> None:
        if self.finetune_complete(method):
            print(f"SKIP complete: {method.name}/finetune")
            return
        checkpoint = pretrain_checkpoint(self.config, method)
        if checkpoint is not None and not checkpoint.is_file() and not self.dry_run:
            raise FileNotFoundError(checkpoint)
        run_dir = finetune_dir(self.config, method)
        last = run_dir / "weights/last.pt"
        resume = last.is_file()
        if run_dir.exists() and any(run_dir.iterdir()) and not resume:
            raise RuntimeError(f"Partial fine-tuning has no last checkpoint: {run_dir}")
        self.execute(build_finetune_command(self.config, method, resume=resume))

    def run_evaluate(self, method: MethodSpec) -> None:
        weights = finetune_weights(self.config, method)
        if not weights.is_file() and not self.dry_run:
            raise FileNotFoundError(weights)
        for command in build_evaluation_commands(self.config, method):
            self.execute(command)

    def run(
        self,
        *,
        method_names: Iterable[str] | None = None,
        stages: Iterable[str] = STAGES,
    ) -> None:
        selected_stages = tuple(stages)
        unknown = sorted(set(selected_stages) - set(STAGES))
        if unknown:
            raise ValueError(f"Unknown stages: {', '.join(unknown)}")
        self.validate_paths()
        if not self.dry_run:
            self.snapshot()
        lock = nullcontext() if self.dry_run else self.lock()
        with lock:
            for method in self.selected_methods(method_names):
                if "pretrain" in selected_stages:
                    self.run_pretrain(method)
                if "finetune" in selected_stages:
                    self.run_finetune(method)
                if "evaluate" in selected_stages:
                    self.run_evaluate(method)

    def status(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for method in self.config.methods:
            evaluations = build_evaluation_commands(self.config, method)
            rows.append(
                {
                    "method": method.name,
                    "pretrain": (
                        "not_applicable"
                        if method.kind == "supervised"
                        else "complete" if self.pretrain_complete(method) else "pending"
                    ),
                    "finetune": "complete" if self.finetune_complete(method) else "pending",
                    "evaluate": (
                        "not_applicable"
                        if not evaluations
                        else "complete" if self.evaluation_complete(method) else "pending"
                    ),
                }
            )
        return rows
