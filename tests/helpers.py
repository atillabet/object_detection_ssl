from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def write_study(
    root: Path,
    *,
    methods: list[dict[str, Any]] | None = None,
    with_shape: bool = True,
) -> Path:
    repo = root / "project"
    repo.mkdir()
    for script in ("pretrain.py", "train.py", "shape.py"):
        (repo / script).write_text("# test\n", encoding="utf-8")
    (repo / "data.yaml").write_text("path: .\n", encoding="utf-8")
    images = root / "images"
    labels = root / "labels"
    images.mkdir()
    labels.mkdir()
    method_rows = methods or [
        {"name": "supervised", "kind": "supervised"},
        {"name": "region", "kind": "vicreg_shape_region"},
    ]
    raw: dict[str, Any] = {
        "schema_version": 1,
        "name": "test_study",
        "paths": {
            "project_root": str(repo),
            "python": sys.executable,
            "output_root": str(root / "artifacts"),
        },
        "runtime": {"gpu": 0},
        "pretraining": {
            "script": "pretrain.py",
            "args": {"images": str(images), "epochs": 2, "batch": 8},
        },
        "finetuning": {
            "script": "train.py",
            "args": {"data": "data.yaml", "epochs": 2},
            "resume_args": {"batch": 8},
        },
        "methods": method_rows,
        "analysis": {"baseline": method_rows[0]["name"]},
    }
    if with_shape:
        raw["evaluations"] = {
            "shape_reliance": {
                "enabled": True,
                "script": "shape.py",
                "images": str(images),
                "labels": str(labels),
                "args": {"max_images": 10, "random_sample": True},
            }
        }
    path = root / "study.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def write_results(path: Path, values: list[tuple[int, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)"))
        writer.writerows(values)


def write_report(path: Path, *, clean: float, shape: float, color: float, texture: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "conditions": [
            {
                "key": "clean",
                "map50": clean * 2,
                "map50_95": clean,
                "recall": 0.7,
            }
        ],
        "cue_reliance_auc": [
            {"category": "shape", "cue_reliance": shape},
            {"category": "color", "cue_reliance": color},
            {"category": "texture", "cue_reliance": texture},
            {"category": "context", "cue_reliance": None},
        ],
    }
    path.write_text(json.dumps(report), encoding="utf-8")
