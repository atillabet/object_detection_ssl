from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ssl_experiments.commands import StageCommand


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class CommandResult:
    return_code: int
    started_at: str
    finished_at: str


class CommandRunner:
    def __init__(self, *, gpu: int, dry_run: bool = False) -> None:
        self.gpu = gpu
        self.dry_run = dry_run

    def environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        env["NVIDIA_VISIBLE_DEVICES"] = str(self.gpu)
        return env

    def run(self, stage: StageCommand, log_path: Path) -> CommandResult:
        rendered = shlex.join(stage.command)
        print(f"\n[{stage.method}/{stage.name}] $ {rendered}", flush=True)
        started = utc_now()
        if self.dry_run:
            return CommandResult(0, started, utc_now())

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", buffering=1, encoding="utf-8") as log:
            log.write(f"\nSTARTED={started}\nCOMMAND={rendered}\n")
            process = subprocess.Popen(
                stage.command,
                cwd=stage.cwd,
                env=self.environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    sys.stdout.write(line)
                    log.write(line)
            finally:
                process.stdout.close()
            return_code = process.wait()
            finished = utc_now()
            log.write(f"EXIT_CODE={return_code}\nFINISHED={finished}\n")
        if return_code != 0:
            raise RuntimeError(
                f"{stage.method}/{stage.name} failed with exit code {return_code}"
            )
        return CommandResult(return_code, started, finished)
