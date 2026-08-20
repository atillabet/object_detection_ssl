#!/usr/bin/env python3
"""Run one reproducible Ultralytics validation pass for several checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO


METRIC_KEYS = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "map50",
    "metrics/mAP50-95(B)": "map50_95",
    "fitness": "fitness",
}


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoints must use NAME=/path/to/model.pt")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("checkpoint name cannot be empty")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {path}")
    return name, path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value: object) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for name, checkpoint in args.checkpoint:
        print(f"\n=== Clean validation: {name} ===", flush=True)
        model = YOLO(str(checkpoint))
        metrics = model.val(
            data=str(args.data.resolve()),
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            seed=args.seed,
            deterministic=True,
            conf=0.001,
            iou=0.7,
            max_det=300,
            plots=False,
            project=str((args.output / "runs").resolve()),
            name=name,
            exist_ok=True,
            verbose=True,
        )
        result_values = metrics.results_dict
        row: dict[str, object] = {
            "model": name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
        }
        for source_key, output_key in METRIC_KEYS.items():
            if source_key in result_values:
                row[output_key] = scalar(result_values[source_key])
        for speed_key, value in metrics.speed.items():
            row[f"speed_{speed_key}_ms"] = scalar(value)
        rows.append(row)
        del metrics, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output / "clean_validation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "data": str(args.data.resolve()),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "seed": args.seed,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
        },
        "results": rows,
    }
    (args.output / "clean_validation.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
