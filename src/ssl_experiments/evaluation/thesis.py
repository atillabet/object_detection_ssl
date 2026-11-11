"""Joint calibration and common-target objectness evaluation for a fixed study.

This is a new implementation of the documented protocol. It neither recovers
historical hyperparameters nor supplies evidence about previous checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from ssl_experiments.commands import finetune_weights
from ssl_experiments.config import load_study
from ssl_experiments.data.yolo import IMAGE_EXTENSIONS, load_yolo_labels
from ssl_experiments.evaluation.operating_points import (
    clean_matches,
    common_retention,
    common_targets,
    select_f1_threshold,
)
from ssl_experiments.evaluation.metrics import compute_map_50_95_per_class
from ssl_experiments.evaluation.sahi import sahi_inference_batched
from ssl_experiments.evaluation.shape_reliance import (
    Condition,
    apply_condition,
    build_v2_conditions,
    erase_labeled_regions,
    replace_background,
    stable_rng,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def find_samples(images: Path, labels: Path, expected: int | None = None):
    files = sorted(
        p
        for p in images.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files or (expected is not None and len(files) != expected):
        raise ValueError(
            f"Expected {expected or 'a nonempty set of'} images, found {len(files)} in {images}"
        )
    samples = []
    for path in files:
        identity = path.relative_to(images).as_posix()
        label = (labels / path.relative_to(images)).with_suffix(".txt")
        if not label.is_file():
            raise FileNotFoundError(
                f"Missing annotation (empty files are allowed): {label}"
            )
        samples.append((identity, path, label))
    return samples


def predict(model, image, settings):
    boxes, scores, classes = sahi_inference_batched(
        model,
        image,
        (settings["tile_size"],) * 2,
        (settings["overlap"],) * 2,
        settings["batch_size"],
        model_input_size=settings["imgsz"],
        conf=settings["confidence_floor"],
        large_conf=settings["confidence_floor"],
        iou_threshold=settings["nms_iou"],
        half=settings["half"],
        device=settings["device"],
    )
    return [
        (int(cls), float(score), box)
        for box, score, cls in zip(boxes, scores, classes, strict=True)
    ]


def infer_records(model, samples, settings, transform=None):
    records = {}
    for identity, path, label in samples:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot decode image: {path}")
        boxes, classes = load_yolo_labels(label, image.shape[1], image.shape[0])
        transformed = (
            transform(identity, image, boxes) if transform is not None else image
        )
        records[identity] = {
            "boxes": boxes,
            "classes": classes,
            "predictions": predict(model, transformed, settings),
        }
    return records


def cache_records(path: Path, records) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for identity, record in records.items():
            stream.write(json.dumps({"image_id": identity, **record}) + "\n")


def manifest(samples):
    return [
        {
            "image_id": identity,
            "image_sha256": sha256(path),
            "label_sha256": sha256(label),
        }
        for identity, path, label in samples
    ]


def validate_calibration(calibration, weights, settings) -> None:
    if calibration.get("inference") != settings:
        raise ValueError(
            "Calibration inference settings differ; recalibrate instead of reusing thresholds"
        )
    if set(calibration["models"]) != set(weights):
        raise ValueError("Calibration model names differ from this study")
    for name, path in weights.items():
        entry = calibration["models"][name]
        if entry["checkpoint_sha256"] != sha256(path):
            raise ValueError(f"Checkpoint changed since confidence calibration: {name}")
        if not settings["confidence_floor"] <= entry["threshold"] <= 1.0:
            raise ValueError(f"Invalid calibrated threshold: {name}")


def release_model(model) -> None:
    import torch

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def calibrate(weights, samples, settings, output):
    from ultralytics import YOLO

    payload = {
        "selection": "maximum_micro_f1_on_general_validation",
        "inference": settings,
        "images": len(samples),
        "models": {},
    }
    write_json(output / "validation_manifest.json", manifest(samples))
    for name, path in weights.items():
        model = YOLO(str(path))
        records = infer_records(model, samples, settings)
        point = select_f1_threshold(records, match_iou=settings["match_iou"])
        predictions = [
            (identity, cls, score, box)
            for identity, record in records.items()
            for cls, score, box in record["predictions"]
        ]
        targets = [
            (identity, cls, box)
            for identity, record in records.items()
            for cls, box in zip(record["classes"], record["boxes"], strict=True)
        ]
        _, _, per_class, _ = compute_map_50_95_per_class(predictions, targets)
        classes = sorted({cls for _, cls, _ in targets})
        point["map50"] = float(np.mean([per_class[cls]["mAP50"] for cls in classes]))
        point["map50_95"] = float(
            np.mean([per_class[cls]["mAP50-95"] for cls in classes])
        )
        payload["models"][name] = {
            **point,
            "checkpoint": str(path),
            "checkpoint_sha256": sha256(path),
        }
        cache_records(output / f"{name}_validation_predictions.jsonl", records)
        release_model(model)
        del model, records
    write_json(output / "calibration.json", payload)
    return payload


def objectness(weights, samples, settings, calibration, output, severities, seed):
    from ultralytics import YOLO

    validate_calibration(calibration, weights, settings)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "sample_manifest.json", manifest(samples))
    matches = {}
    for name, path in weights.items():
        model = YOLO(str(path))
        records = infer_records(model, samples, settings)
        matches[name] = clean_matches(
            records,
            threshold=calibration["models"][name]["threshold"],
            match_iou=settings["match_iou"],
        )
        cache_records(output / f"{name}_clean_predictions.jsonl", records)
        release_model(model)
        del model, records
    common = common_targets(matches)
    write_json(output / "common_targets.json", [list(key) for key in sorted(common)])
    selected = {}
    for identity, index in common:
        selected.setdefault(identity, set()).add(index)
    # No silhouette/edge diagnostic is included in the thesis protocol.
    conditions = [
        condition
        for condition in build_v2_conditions(severities, include_edge=False)
        if condition.transform not in {"none", "object_patch_rotate"}
    ]
    conditions.extend(
        (
            Condition("object_only", "context", "object_only", 1.0),
            Condition("object_removed", "diagnostic", "object_removed", 1.0),
        )
    )
    sample_by_id = {identity: (path, label) for identity, path, label in samples}
    identities = [identity for identity, _, _ in samples]

    def intervention(condition):
        def transform(identity, image, boxes):
            target_boxes = [boxes[i] for i in sorted(selected.get(identity, ()))]
            if condition.transform == "background_swap":
                candidates = [value for value in identities if value != identity]
                if not candidates:
                    raise ValueError("Background swap requires two sample images")
                rng = stable_rng(seed, identity, "background_donor")
                donor_path = sample_by_id[
                    candidates[int(rng.integers(len(candidates)))]
                ][0]
                donor = cv2.imread(str(donor_path))
                if donor is None:
                    raise ValueError(f"Cannot decode background donor: {donor_path}")
                return replace_background(image, donor, target_boxes, severity=1.0)
            if condition.transform == "object_only":
                return replace_background(
                    image, np.zeros_like(image), target_boxes, severity=1.0
                )
            if condition.transform == "object_removed":
                return erase_labeled_regions(image, target_boxes)
            return apply_condition(
                image, target_boxes, condition, seed=seed, image_id=identity
            )[0]

        return transform

    rows = []
    for name, path in weights.items():
        model = YOLO(str(path))
        threshold = calibration["models"][name]["threshold"]
        for condition in conditions:
            records = infer_records(model, samples, settings, intervention(condition))
            result = common_retention(
                records,
                matches[name],
                common,
                threshold=threshold,
                match_iou=settings["match_iou"],
            )
            if condition.transform == "object_removed":
                result["suppression_rate"] = 1.0 - result["retention"]
                result["diagnostic_only"] = True
            rows.append(
                {"model": name, **condition.to_dict(), "threshold": threshold, **result}
            )
            cache_records(output / f"{name}_{condition.key}_predictions.jsonl", records)
        release_model(model)
        del model
    write_json(
        output / "report.json",
        {
            "protocol": "shared_clean_targets_all_models",
            "images": len(samples),
            "common_target_count": len(common),
            "inference": settings,
            "calibration": calibration,
            "conditions": rows,
            "limitations": [
                "No confidence intervals or repeated-training uncertainty are computed.",
                "Bounding-box interventions may retain nearby background and introduce seams.",
            ],
        },
    )
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "retention.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_study(study: Path, phase: str = "all", overwrite: bool = False):
    config = load_study(study)
    protocol = config.raw.get("thesis_evaluation")
    if (
        not isinstance(protocol, dict)
        or not {"validation", "objectness"} <= protocol.keys()
    ):
        raise ValueError(
            "Study must define thesis_evaluation.validation and thesis_evaluation.objectness"
        )
    settings = {
        "tile_size": 320,
        "overlap": 64,
        "imgsz": 320,
        "batch_size": 16,
        "confidence_floor": 0.001,
        "nms_iou": 0.7,
        "match_iou": 0.5,
        "device": "0",
        "half": False,
        **protocol.get("inference", {}),
    }
    if not 0 < settings["confidence_floor"] < 1 or not 0 < settings["match_iou"] <= 1:
        raise ValueError("Invalid confidence floor or matching IoU")
    weights = {
        method.name: finetune_weights(config, method) for method in config.methods
    }
    for path in weights.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output = config.artifact_root / "thesis_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    calibration_path = output / "calibration.json"
    if phase in {"all", "calibration"}:
        if calibration_path.exists() and not overwrite:
            raise FileExistsError(
                "Calibration already exists; use --phase objectness or --overwrite"
            )
        validation = protocol["validation"]
        samples = find_samples(
            config.resolve_path(validation["images"]),
            config.resolve_path(validation["labels"]),
        )
        calibration = calibrate(weights, samples, settings, output)
    else:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if phase in {"all", "objectness"}:
        destination = output / "objectness"
        if (destination / "report.json").exists() and not overwrite:
            raise FileExistsError(
                "Objectness report already exists; use --overwrite to replace it"
            )
        behavior = protocol["objectness"]
        samples = find_samples(
            config.resolve_path(behavior["images"]),
            config.resolve_path(behavior["labels"]),
            expected=behavior.get("expected_images", 2000),
        )
        objectness(
            weights,
            samples,
            settings,
            calibration,
            destination,
            protocol.get("severities", [0.25, 0.5, 0.75]),
            protocol.get("seed", 0),
        )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument(
        "--phase", choices=("all", "calibration", "objectness"), default="all"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print(run_study(args.study, args.phase, args.overwrite))


if __name__ == "__main__":
    main()
