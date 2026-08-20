#!/usr/bin/env python3
"""Measure how strongly a YOLO detector relies on object shape."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table
from ultralytics import YOLO

from ssl_experiments.data.yolo import IMAGE_EXTENSIONS, load_yolo_labels
from ssl_experiments.evaluation.metrics import (
    compute_map_50_95_per_class,
    evaluate_image,
)
from ssl_experiments.evaluation.sahi import sahi_inference_batched
from ssl_experiments.evaluation.iou import iou_one_vs_many
from ssl_experiments.evaluation.shape_reliance import (
    Condition,
    aggregate_category_retention,
    aggregate_reliance_auc,
    aggregate_shape_scores,
    apply_condition,
    build_conditions,
    build_v2_conditions,
    erase_labeled_regions,
    stable_rng,
)

app = typer.Typer(
    help=(
        "Evaluate a frozen detector on geometry-preserving appearance changes and "
        "appearance-preserving spatial disruptions."
    )
)
console = Console()


@dataclass
class PendingInference:
    condition: Condition
    image_id: str
    image_bgr: np.ndarray
    gt_boxes: list[list[float]]
    gt_classes: list[int]


@dataclass
class ConditionAccumulator:
    condition: Condition
    predictions: list[tuple[str, int, float, list[float]]] = field(
        default_factory=list
    )
    ground_truths: list[tuple[str, int, list[float]]] = field(default_factory=list)
    per_image: list[dict[str, Any]] = field(default_factory=list)
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    transformed_objects: int = 0
    skipped_objects: int = 0


def _parse_severities(value: str) -> list[float]:
    try:
        severities = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise typer.BadParameter(
            "Severities must be comma-separated numbers, for example 0.25,0.5,0.75"
        ) from exc
    if not severities:
        raise typer.BadParameter("At least one severity is required")
    if any(severity <= 0.0 or severity > 1.0 for severity in severities):
        raise typer.BadParameter("Every severity must be in the interval (0, 1]")
    return severities


def _find_images(
    source: Path,
    max_images: int | None,
    *,
    random_sample: bool = False,
    seed: int = 0,
) -> tuple[list[Path], Path]:
    source = source.resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise typer.BadParameter(f"Unsupported image type: {source.suffix}")
        return [source], source.parent
    if not source.is_dir():
        raise typer.BadParameter(f"Image source does not exist: {source}")

    images = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if max_images is not None and max_images < len(images):
        if random_sample:
            images = sorted(random.Random(seed).sample(images, max_images))
        else:
            images = images[:max_images]
    if not images:
        raise typer.BadParameter(f"No supported images found under {source}")
    return images, source


def _infer_labels_dir(source: Path) -> Path | None:
    source = source.resolve()
    directory = source.parent if source.is_file() else source
    candidates: list[Path] = []

    if directory.name == "images":
        candidates.append(directory.parent / "labels")
    if directory.parent.name == "images":
        candidates.append(directory.parent.parent / "labels" / directory.name)
    candidates.extend(
        [
            directory.parent / "labels",
            directory / "labels",
        ]
    )
    return next((candidate for candidate in candidates if candidate.is_dir()), None)


def _label_path(
    image_path: Path,
    image_root: Path,
    labels_dir: Path,
) -> Path:
    try:
        relative = image_path.relative_to(image_root).with_suffix(".txt")
    except ValueError:
        relative = Path(f"{image_path.stem}.txt")
    nested = labels_dir / relative
    if nested.exists():
        return nested
    return labels_dir / f"{image_path.stem}.txt"


def _load_yolo_segment_masks(
    path: Path,
    image_width: int,
    image_height: int,
) -> tuple[list[np.ndarray], list[int]]:
    """Load standard YOLO segmentation polygons as full-resolution masks."""
    if not path.exists():
        raise FileNotFoundError(f"Segmentation label does not exist: {path}")
    masks: list[np.ndarray] = []
    classes: list[int] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        values = line.split()
        if not values:
            continue
        if len(values) < 7 or (len(values) - 1) % 2:
            raise ValueError(
                f"{path}:{line_number} is not a YOLO segmentation polygon"
            )
        class_id = int(float(values[0]))
        coordinates = np.asarray(values[1:], dtype=np.float32).reshape(-1, 2)
        coordinates[:, 0] = np.clip(coordinates[:, 0] * image_width, 0, image_width - 1)
        coordinates[:, 1] = np.clip(coordinates[:, 1] * image_height, 0, image_height - 1)
        polygon = np.rint(coordinates).astype(np.int32)
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        if not np.any(mask):
            raise ValueError(f"{path}:{line_number} produced an empty mask")
        masks.append(mask.astype(bool))
        classes.append(class_id)
    return masks, classes


def _select_background_donor_index(
    image_count: int,
    image_index: int,
    *,
    seed: int,
    image_id: str,
) -> int:
    """Select a deterministic different image as the real-background donor."""
    if image_count < 2:
        raise ValueError("V2 context swaps require at least two source images")
    rng = stable_rng(seed, image_id, "background_donor")
    donor_index = int(rng.integers(0, image_count - 1))
    return donor_index + 1 if donor_index >= image_index else donor_index


def _model_class_names(model: YOLO) -> dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {class_id: str(name) for class_id, name in enumerate(names)}


def _prediction_arrays(result: Any) -> tuple[list[list[float]], list[int], list[float]]:
    if result.boxes is None or len(result.boxes) == 0:
        return [], [], []
    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(float).tolist()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int).tolist()
    confidences = result.boxes.conf.detach().cpu().numpy().astype(float).tolist()
    return boxes, classes, confidences


def _record_predictions(
    accumulator: ConditionAccumulator,
    item: PendingInference,
    pred_boxes: list[list[float]],
    pred_classes: list[int],
    pred_confidences: list[float],
    *,
    match_iou: float,
) -> None:
    for box, class_id, confidence in zip(
        pred_boxes,
        pred_classes,
        pred_confidences,
        strict=True,
    ):
        accumulator.predictions.append(
            (item.image_id, class_id, confidence, box)
        )

    tp, fp, fn, precision, recall, f1 = evaluate_image(
        item.gt_boxes,
        item.gt_classes,
        pred_boxes,
        pred_classes,
        match_iou,
    )
    accumulator.total_tp += tp
    accumulator.total_fp += fp
    accumulator.total_fn += fn
    accumulator.per_image.append(
        {
            "condition": item.condition.key,
            "category": item.condition.category,
            "transform": item.condition.transform,
            "severity": item.condition.severity,
            "image_id": item.image_id,
            "num_gt": len(item.gt_boxes),
            "num_predictions": len(pred_boxes),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def _group_ground_truths(
    accumulator: ConditionAccumulator,
) -> dict[str, tuple[list[list[float]], list[int]]]:
    grouped: dict[str, tuple[list[list[float]], list[int]]] = {}
    for image_id, class_id, box in accumulator.ground_truths:
        boxes, classes = grouped.setdefault(image_id, ([], []))
        boxes.append(box)
        classes.append(class_id)
    return grouped


def _group_predictions(
    accumulator: ConditionAccumulator,
) -> dict[str, list[tuple[int, float, list[float]]]]:
    grouped: dict[str, list[tuple[int, float, list[float]]]] = {}
    for image_id, class_id, confidence, box in accumulator.predictions:
        grouped.setdefault(image_id, []).append((class_id, confidence, box))
    return grouped


def _match_correct_detections(
    gt_boxes: list[list[float]],
    gt_classes: list[int],
    predictions: list[tuple[int, float, list[float]]],
    *,
    confidence: float,
    match_iou: float,
) -> dict[int, tuple[float, float]]:
    """Greedily map confident correct-class predictions to GT indices."""
    if not gt_boxes:
        return {}
    gt_array = np.asarray(gt_boxes, dtype=np.float32)
    gt_class_array = np.asarray(gt_classes, dtype=np.int64)
    matched: dict[int, tuple[float, float]] = {}
    for class_id, score, box in sorted(
        predictions,
        key=lambda item: item[1],
        reverse=True,
    ):
        if score < confidence:
            continue
        ious = iou_one_vs_many(box, gt_array)
        ious[gt_class_array != class_id] = 0.0
        for gt_index in matched:
            ious[gt_index] = 0.0
        best_index = int(np.argmax(ious))
        best_iou = float(ious[best_index])
        if best_iou >= match_iou:
            matched[best_index] = (score, best_iou)
    return matched


def _has_class_switch(
    gt_box: list[float],
    gt_class: int,
    predictions: list[tuple[int, float, list[float]]],
    *,
    confidence: float,
    match_iou: float,
) -> bool:
    for class_id, score, box in predictions:
        if class_id == gt_class or score < confidence:
            continue
        if float(iou_one_vs_many(box, np.asarray([gt_box], dtype=np.float32))[0]) >= match_iou:
            return True
    return False


def _compute_paired_detection_rows(
    accumulators: dict[str, ConditionAccumulator],
    conditions: list[Condition],
    *,
    confidence: float,
    match_iou: float,
) -> list[dict[str, Any]]:
    """Measure whether each clean true positive survives a counterfactual."""
    clean = accumulators["clean"]
    ground_truths = _group_ground_truths(clean)
    clean_predictions = _group_predictions(clean)
    clean_matches = {
        image_id: _match_correct_detections(
            boxes,
            classes,
            clean_predictions.get(image_id, []),
            confidence=confidence,
            match_iou=match_iou,
        )
        for image_id, (boxes, classes) in ground_truths.items()
    }

    rows: list[dict[str, Any]] = []
    for condition in conditions:
        condition_predictions = _group_predictions(accumulators[condition.key])
        baseline_count = 0
        retained_count = 0
        switched_count = 0
        confidence_deltas: list[float] = []
        localization_deltas: list[float] = []
        for image_id, (boxes, classes) in ground_truths.items():
            baseline = clean_matches[image_id]
            predictions = condition_predictions.get(image_id, [])
            counterfactual = _match_correct_detections(
                boxes,
                classes,
                predictions,
                confidence=confidence,
                match_iou=match_iou,
            )
            for gt_index, (clean_score, clean_iou) in baseline.items():
                baseline_count += 1
                condition_score, condition_iou = counterfactual.get(
                    gt_index,
                    (0.0, 0.0),
                )
                if gt_index in counterfactual:
                    retained_count += 1
                elif _has_class_switch(
                    boxes[gt_index],
                    classes[gt_index],
                    predictions,
                    confidence=confidence,
                    match_iou=match_iou,
                ):
                    switched_count += 1
                confidence_deltas.append(condition_score - clean_score)
                localization_deltas.append(condition_iou - clean_iou)
        rows.append(
            {
                **condition.to_dict(),
                "paired_confidence": confidence,
                "clean_true_positives": baseline_count,
                "retained_true_positives": retained_count,
                "paired_retention": (
                    retained_count / baseline_count if baseline_count else None
                ),
                "mean_confidence_delta": (
                    float(np.mean(confidence_deltas))
                    if confidence_deltas
                    else None
                ),
                "mean_localization_iou_delta": (
                    float(np.mean(localization_deltas))
                    if localization_deltas
                    else None
                ),
                "class_switch_rate": (
                    switched_count / baseline_count if baseline_count else None
                ),
            }
        )
    return rows


def _flush_inference(
    model: YOLO,
    pending: list[PendingInference],
    accumulators: dict[str, ConditionAccumulator],
    *,
    imgsz: int,
    confidence: float,
    nms_iou: float,
    match_iou: float,
    device: str | None,
    max_det: int,
    use_sahi: bool,
    slice_size: int,
    overlap: int,
    sahi_batch_size: int,
    large_confidence: float,
    size_threshold: int,
) -> None:
    if not pending:
        return
    if use_sahi:
        for item in pending:
            pred_boxes, pred_confidences, pred_classes = sahi_inference_batched(
                model,
                item.image_bgr,
                (slice_size, slice_size),
                (overlap, overlap),
                sahi_batch_size,
                model_input_size=imgsz,
                conf=confidence,
                iou_threshold=nms_iou,
                large_conf=large_confidence,
                size_threshold=size_threshold,
                half=device not in {None, "cpu"},
                device=device,
            )
            if len(pred_confidences) > max_det:
                order = np.argsort(pred_confidences)[::-1][:max_det]
                pred_boxes = [pred_boxes[index] for index in order]
                pred_classes = [pred_classes[index] for index in order]
                pred_confidences = [pred_confidences[index] for index in order]
            _record_predictions(
                accumulators[item.condition.key],
                item,
                pred_boxes,
                pred_classes,
                pred_confidences,
                match_iou=match_iou,
            )
    else:
        results = model.predict(
            [item.image_bgr for item in pending],
            imgsz=imgsz,
            conf=confidence,
            iou=nms_iou,
            device=device,
            max_det=max_det,
            verbose=False,
        )
        if len(results) != len(pending):
            raise RuntimeError(
                f"Expected {len(pending)} inference results, received {len(results)}"
            )
        for item, result in zip(pending, results, strict=True):
            pred_boxes, pred_classes, pred_confidences = _prediction_arrays(result)
            _record_predictions(
                accumulators[item.condition.key],
                item,
                pred_boxes,
                pred_classes,
                pred_confidences,
                match_iou=match_iou,
            )
    pending.clear()


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _compute_condition_metrics(
    accumulator: ConditionAccumulator,
    ground_truth_class_ids: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _, _, per_class, _ = compute_map_50_95_per_class(
        accumulator.predictions,
        accumulator.ground_truths,
    )
    class_map50 = [
        per_class.get(class_id, {}).get("mAP50", 0.0)
        for class_id in ground_truth_class_ids
    ]
    class_map50_95 = [
        per_class.get(class_id, {}).get("mAP50-95", 0.0)
        for class_id in ground_truth_class_ids
    ]
    map50 = float(np.mean(class_map50)) if class_map50 else 0.0
    map50_95 = float(np.mean(class_map50_95)) if class_map50_95 else 0.0

    precision = _safe_divide(
        accumulator.total_tp,
        accumulator.total_tp + accumulator.total_fp,
    )
    recall = _safe_divide(
        accumulator.total_tp,
        accumulator.total_tp + accumulator.total_fn,
    )
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    summary = {
        **accumulator.condition.to_dict(),
        "num_images": len(accumulator.per_image),
        "num_ground_truths": len(accumulator.ground_truths),
        "num_predictions": len(accumulator.predictions),
        "transformed_objects": accumulator.transformed_objects,
        "skipped_objects": accumulator.skipped_objects,
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": accumulator.total_tp,
        "fp": accumulator.total_fp,
        "fn": accumulator.total_fn,
    }
    class_rows = [
        {
            **accumulator.condition.to_dict(),
            "class_id": class_id,
            "map50": per_class.get(class_id, {}).get("mAP50", 0.0),
            "map50_95": per_class.get(class_id, {}).get("mAP50-95", 0.0),
            "gt_count": sum(
                1
                for _, gt_class_id, _ in accumulator.ground_truths
                if gt_class_id == class_id
            ),
        }
        for class_id in ground_truth_class_ids
    ]
    return summary, class_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _write_markdown_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    category_rows: list[dict[str, Any]],
    reliance_auc_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    *,
    weights: Path,
    source: Path,
    benchmark_version: str,
    region_source: str,
) -> None:
    title = (
        "Feature Reliance Benchmark V2"
        if benchmark_version == "v2"
        else "Shape Reliance Benchmark"
    )
    lines = [
        f"# {title}",
        "",
        f"- Weights: `{weights}`",
        f"- Images: `{source}`",
        f"- Benchmark version: `{benchmark_version}`",
    ]
    if benchmark_version == "v2":
        lines.append(f"- Object-region source: `{region_source}`")
    lines.extend(
        [
            "",
            "## Condition Results",
            "",
            "| Condition | Category | Severity | mAP50 | mAP50-95 | Precision | Recall |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['key']} | {row['category']} | {row['severity']:.2f} | "
            f"{row['map50']:.4f} | {row['map50_95']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} |"
        )

    if score_rows:
        lines.extend(
            [
                "",
                "## Shape Reliance Scores",
                "",
                "| Severity | Appearance retention | Shape retention | Shape impact | Shape reliance gap | Absolute mAP gap |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in score_rows:
            lines.append(
                f"| {row['severity']:.2f} | "
                f"{_format_optional(row['appearance_retention'])} | "
                f"{_format_optional(row['shape_retention'])} | "
                f"{_format_optional(row['shape_impact'])} | "
                f"{_format_optional(row['shape_reliance_gap'])} | "
                f"{row['absolute_shape_gap']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Category Retention",
            "",
            "| Severity | Category | mAP50-95 | Retention |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in category_rows:
        lines.append(
            f"| {row['severity']:.2f} | {row['category']} | "
            f"{row['metric']:.4f} | {_format_optional(row['retention'])} |"
        )

    if reliance_auc_rows:
        lines.extend(
            [
                "",
                "## Cue Reliance Curves",
                "",
                "| Category | Max severity | Retention AUC | Cue reliance |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in reliance_auc_rows:
            lines.append(
                f"| {row['category']} | "
                f"{_format_optional(row['max_severity'])} | "
                f"{_format_optional(row['retention_auc'])} | "
                f"{_format_optional(row['cue_reliance'])} |"
            )
        by_category = {row["category"]: row for row in reliance_auc_rows}
        shape = by_category.get("shape", {}).get("cue_reliance")
        texture = by_category.get("texture", {}).get("cue_reliance")
        if shape is not None and texture is not None:
            lines.extend(
                [
                    "",
                    "Relative shape-minus-texture reliance: "
                    f"`{float(shape) - float(texture):.4f}`.",
                ]
            )

    if paired_rows:
        lines.extend(
            [
                "",
                "## Paired Clean-Detection Retention",
                "",
                "| Condition | Clean TPs | Retained | Retention | Confidence delta | IoU delta | Class-switch rate |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in paired_rows:
            lines.append(
                f"| {row['key']} | {row['clean_true_positives']} | "
                f"{row['retained_true_positives']} | "
                f"{_format_optional(row['paired_retention'])} | "
                f"{_format_optional(row['mean_confidence_delta'])} | "
                f"{_format_optional(row['mean_localization_iou_delta'])} | "
                f"{_format_optional(row['class_switch_rate'])} |"
            )

    if benchmark_version == "v2":
        lines.extend(
            [
                "",
                "Cue reliance is `1 - normalized retention AUC`; larger values "
                "mean that suppressing the cue causes greater performance loss. "
                "Paired retention follows only detections that are correct on "
                "the clean image. Binary context swaps are reported as endpoint "
                "retention and are not assigned a curve AUC.",
                "",
                "## Interpretation Limits",
                "",
                "- Suppression transformations reduce targeted information but do not mathematically isolate a single visual feature.",
                "- A bounding-box fallback also modifies object-adjacent context; segmentation masks provide stronger isolation.",
                "- Real background donors have labeled regions inpainted, which can leave local artifacts.",
                "- Patch operations measure global and local spatial structure, not silhouette use alone.",
                "- The edge-only condition remains diagnostic and is excluded from cue-reliance curves.",
                "- Preview images must be inspected before using the results in a thesis.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "The score uses mAP50-95. Appearance and shape metrics are the means "
                "of every condition in their respective category. The normalized "
                "shape reliance gap is `appearance_retention - shape_retention`; the "
                "absolute gap is `appearance_mAP - shape_mAP`.",
                "",
                "## Interpretation Limits",
                "",
                "- A positive gap is evidence that predictions rely on spatial structure more than color.",
                "- Box-based phase and patch transforms measure spatial structure, not contour use in isolation.",
                "- Background protection uses boxes rather than segmentation masks, so nearby context can be preserved.",
                "- Shape variants retain the original labels only to measure original-class detection retention; they are not edited-image accuracy labels.",
                "- The edge-only condition is a diagnostic and is excluded from the shape-reliance score.",
                "- Preview images must be inspected before using the results in a thesis.",
                "",
            ]
        )
    path.write_text("\n".join(lines))


def _print_results(
    summary_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    category_rows: list[dict[str, Any]],
    reliance_auc_rows: list[dict[str, Any]],
) -> None:
    condition_table = Table(title="Condition metrics")
    condition_table.add_column("Condition")
    condition_table.add_column("Category")
    condition_table.add_column("mAP50", justify="right")
    condition_table.add_column("mAP50-95", justify="right")
    condition_table.add_column("Recall", justify="right")
    for row in summary_rows:
        condition_table.add_row(
            row["key"],
            row["category"],
            f"{row['map50']:.4f}",
            f"{row['map50_95']:.4f}",
            f"{row['recall']:.4f}",
        )
    console.print(condition_table)

    if score_rows:
        score_table = Table(title="Shape reliance (mAP50-95)")
        score_table.add_column("Severity", justify="right")
        score_table.add_column("Appearance retention", justify="right")
        score_table.add_column("Shape retention", justify="right")
        score_table.add_column("Shape impact", justify="right")
        score_table.add_column("Reliance gap", justify="right")
        score_table.add_column("Absolute gap", justify="right")
        for row in score_rows:
            score_table.add_row(
                f"{row['severity']:.2f}",
                _format_optional(row["appearance_retention"]),
                _format_optional(row["shape_retention"]),
                _format_optional(row["shape_impact"]),
                _format_optional(row["shape_reliance_gap"]),
                f"{row['absolute_shape_gap']:.4f}",
            )
        console.print(score_table)

    if reliance_auc_rows:
        reliance_table = Table(title="Cue reliance from retention curves")
        reliance_table.add_column("Category")
        reliance_table.add_column("Retention AUC", justify="right")
        reliance_table.add_column("Cue reliance", justify="right")
        for row in reliance_auc_rows:
            reliance_table.add_row(
                str(row["category"]),
                _format_optional(row["retention_auc"]),
                _format_optional(row["cue_reliance"]),
            )
        console.print(reliance_table)

    extra_categories = [
        row for row in category_rows if row["category"] in {"context", "stress"}
    ]
    if extra_categories and not reliance_auc_rows:
        category_table = Table(title="Hard-mode category retention")
        category_table.add_column("Severity", justify="right")
        category_table.add_column("Category")
        category_table.add_column("mAP50-95", justify="right")
        category_table.add_column("Retention", justify="right")
        for row in extra_categories:
            category_table.add_row(
                f"{row['severity']:.2f}",
                str(row["category"]),
                f"{row['metric']:.4f}",
                _format_optional(row["retention"]),
            )
        console.print(category_table)


@app.command()
def main(
    source: Annotated[
        Path,
        typer.Argument(help="Image file or directory containing validation images."),
    ],
    weights: Annotated[
        Path,
        typer.Option("--weights", "-w", help="Ultralytics YOLO .pt weights."),
    ],
    labels: Annotated[
        Path | None,
        typer.Option(
            "--labels",
            "-l",
            help="YOLO labels directory. Inferred from a standard dataset layout.",
        ),
    ] = None,
    segments: Annotated[
        Path | None,
        typer.Option(
            "--segments",
            help=(
                "Optional YOLO segmentation-label directory aligned with "
                "--labels; V2 otherwise reports a box fallback."
            ),
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Root directory for benchmark runs."),
    ] = Path("artifacts/shape_reliance"),
    run_name: Annotated[
        str | None,
        typer.Option(help="Output folder name. Defaults to model plus timestamp."),
    ] = None,
    severities: Annotated[
        str,
        typer.Option(help="Comma-separated perturbation strengths in (0, 1]."),
    ] = "0.25,0.5,0.75",
    benchmark_version: Annotated[
        str,
        typer.Option(
            "--benchmark-version",
            help="Benchmark protocol: v1 preserves the pilot; v2 isolates cues.",
        ),
    ] = "v1",
    hard_mode: Annotated[
        bool,
        typer.Option(
            "--hard-mode/--standard-mode",
            help="Add V1 texture, phase, context, and object-dropout stress tests.",
        ),
    ] = False,
    imgsz: Annotated[
        int,
        typer.Option(help="YOLO inference image size."),
    ] = 640,
    confidence: Annotated[
        float,
        typer.Option("--conf", help="Prediction floor used for AP and fixed metrics."),
    ] = 0.001,
    nms_iou: Annotated[
        float,
        typer.Option("--nms-iou", help="YOLO non-maximum suppression IoU."),
    ] = 0.7,
    match_iou: Annotated[
        float,
        typer.Option("--match-iou", help="Prediction-to-ground-truth matching IoU."),
    ] = 0.5,
    paired_confidence: Annotated[
        float,
        typer.Option(
            "--paired-conf",
            help="Confidence threshold for V2 paired clean-detection retention.",
        ),
    ] = 0.25,
    device: Annotated[
        str | None,
        typer.Option(help="Torch device, for example 0 or cpu."),
    ] = None,
    inference_batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Number of transformed images per model call."),
    ] = 16,
    sahi: Annotated[
        bool,
        typer.Option(
            "--sahi/--no-sahi",
            help="Use tiled inference for large images containing small objects.",
        ),
    ] = False,
    slice_size: Annotated[
        int,
        typer.Option("--slice-size", help="Width and height of each SAHI tile."),
    ] = 320,
    overlap: Annotated[
        int,
        typer.Option(help="SAHI tile overlap in pixels."),
    ] = 64,
    sahi_batch_size: Annotated[
        int,
        typer.Option("--sahi-batch-size", help="Tiles per SAHI model call."),
    ] = 128,
    large_confidence: Annotated[
        float,
        typer.Option(
            "--large-conf",
            help="Confidence floor for SAHI detections above --size-threshold.",
        ),
    ] = 0.25,
    size_threshold: Annotated[
        int,
        typer.Option(
            help="Maximum box dimension where the lower --conf threshold applies.",
        ),
    ] = 16,
    max_det: Annotated[
        int,
        typer.Option(help="Maximum detections per transformed image."),
    ] = 300,
    max_images: Annotated[
        int | None,
        typer.Option(help="Optional limit for a quick benchmark run."),
    ] = None,
    random_sample: Annotated[
        bool,
        typer.Option(
            "--random-sample/--ordered-limit",
            help="Select --max-images deterministically across the complete source.",
        ),
    ] = False,
    preview_images: Annotated[
        int,
        typer.Option(help="Number of source images saved under every condition."),
    ] = 5,
    shape_grid_size: Annotated[
        int,
        typer.Option(
            "--shape-grid",
            help="Grid width and height used for object-box patch shuffling.",
        ),
    ] = 4,
    seed: Annotated[
        int,
        typer.Option(help="Seed for deterministic patch selection."),
    ] = 0,
    edge_diagnostic: Annotated[
        bool,
        typer.Option(
            "--edge-diagnostic/--no-edge-diagnostic",
            help="Evaluate an edge-only view; excluded from the reliance score.",
        ),
    ] = True,
) -> None:
    """Run the shape-reliance counterfactual benchmark."""
    if not weights.is_file():
        raise typer.BadParameter(f"Weights file does not exist: {weights}")
    if inference_batch_size < 1:
        raise typer.BadParameter("--batch-size must be at least 1")
    if max_images is not None and max_images < 1:
        raise typer.BadParameter("--max-images must be at least 1")
    if slice_size < 1:
        raise typer.BadParameter("--slice-size must be at least 1")
    if overlap < 0 or overlap >= slice_size:
        raise typer.BadParameter("--overlap must be non-negative and smaller than --slice-size")
    if sahi_batch_size < 1:
        raise typer.BadParameter("--sahi-batch-size must be at least 1")
    if size_threshold < 0:
        raise typer.BadParameter("--size-threshold cannot be negative")
    if shape_grid_size < 2:
        raise typer.BadParameter("--shape-grid must be at least 2")
    if preview_images < 0:
        raise typer.BadParameter("--preview-images cannot be negative")
    if not 0.0 <= paired_confidence <= 1.0:
        raise typer.BadParameter("--paired-conf must be in [0, 1]")

    benchmark_version = benchmark_version.lower()
    if benchmark_version not in {"v1", "v2"}:
        raise typer.BadParameter("--benchmark-version must be v1 or v2")
    if benchmark_version == "v2" and hard_mode:
        raise typer.BadParameter("--hard-mode belongs to V1 and cannot be used with V2")
    if benchmark_version == "v1" and segments is not None:
        raise typer.BadParameter("--segments is only used by benchmark V2")

    parsed_severities = _parse_severities(severities)
    if benchmark_version == "v2":
        conditions = build_v2_conditions(
            parsed_severities,
            include_edge=edge_diagnostic,
        )
    else:
        conditions = build_conditions(
            parsed_severities,
            include_edge=edge_diagnostic,
            hard_mode=hard_mode,
        )
    image_paths, image_root = _find_images(
        source,
        max_images,
        random_sample=random_sample,
        seed=seed,
    )
    if benchmark_version == "v2" and len(image_paths) < 2:
        raise typer.BadParameter("V2 requires at least two images for context swaps")
    labels_dir = labels.resolve() if labels is not None else _infer_labels_dir(source)
    if labels_dir is None or not labels_dir.is_dir():
        raise typer.BadParameter(
            "Could not infer a labels directory; pass it explicitly with --labels"
        )
    segments_dir = segments.resolve() if segments is not None else None
    if segments_dir is not None and not segments_dir.is_dir():
        raise typer.BadParameter(f"Segmentation directory does not exist: {segments_dir}")
    region_source = "segmentation_masks" if segments_dir else "bounding_boxes"

    run_name = run_name or (
        f"{weights.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = output / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "sample_images.txt").write_text(
        "".join(
            f"{image_path.relative_to(image_root)}\n"
            for image_path in image_paths
        )
    )

    console.print(f"[cyan]Loading model:[/cyan] {weights}")
    model = YOLO(str(weights))
    class_names = _model_class_names(model)
    accumulators = {
        condition.key: ConditionAccumulator(condition)
        for condition in conditions
    }
    pending: list[PendingInference] = []
    labeled_images = 0
    total_objects = 0

    for image_index, image_path in enumerate(
        track(image_paths, description="Generating variants and running inference"),
    ):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            console.print(f"[yellow]Skipping unreadable image:[/yellow] {image_path}")
            continue
        image_height, image_width = image_bgr.shape[:2]
        image_id = str(image_path.relative_to(image_root))
        label_path = _label_path(image_path, image_root, labels_dir)
        gt_boxes, gt_classes = load_yolo_labels(
            label_path,
            image_width,
            image_height,
        )
        if label_path.exists():
            labeled_images += 1
        total_objects += len(gt_boxes)

        unknown_classes = sorted(set(gt_classes) - set(class_names))
        if unknown_classes:
            raise typer.BadParameter(
                f"{label_path} contains classes absent from the model: "
                f"{unknown_classes}"
            )

        instance_masks: list[np.ndarray] | None = None
        if segments_dir is not None:
            segment_path = _label_path(image_path, image_root, segments_dir)
            try:
                instance_masks, segment_classes = _load_yolo_segment_masks(
                    segment_path,
                    image_width,
                    image_height,
                )
            except (FileNotFoundError, ValueError) as exc:
                raise typer.BadParameter(str(exc)) from exc
            if segment_classes != gt_classes:
                raise typer.BadParameter(
                    f"{segment_path} must contain one polygon per detection label "
                    "in the same class order"
                )

        background_bgr: np.ndarray | None = None
        if benchmark_version == "v2":
            donor_index = _select_background_donor_index(
                len(image_paths),
                image_index,
                seed=seed,
                image_id=image_id,
            )
            donor_path = image_paths[donor_index]
            donor_bgr = cv2.imread(str(donor_path))
            if donor_bgr is None:
                raise RuntimeError(f"Could not read background donor: {donor_path}")
            donor_height, donor_width = donor_bgr.shape[:2]
            donor_label_path = _label_path(donor_path, image_root, labels_dir)
            donor_boxes, donor_classes = load_yolo_labels(
                donor_label_path,
                donor_width,
                donor_height,
            )
            donor_masks: list[np.ndarray] | None = None
            if segments_dir is not None:
                donor_segment_path = _label_path(
                    donor_path,
                    image_root,
                    segments_dir,
                )
                try:
                    donor_masks, donor_segment_classes = _load_yolo_segment_masks(
                        donor_segment_path,
                        donor_width,
                        donor_height,
                    )
                except (FileNotFoundError, ValueError) as exc:
                    raise typer.BadParameter(str(exc)) from exc
                if donor_segment_classes != donor_classes:
                    raise typer.BadParameter(
                        f"{donor_segment_path} does not align with donor labels"
                    )
            background_bgr = erase_labeled_regions(
                donor_bgr,
                donor_boxes,
                donor_masks,
            )

        for condition in conditions:
            transformed, transform_stats = apply_condition(
                image_bgr,
                gt_boxes,
                condition,
                seed=seed,
                image_id=image_id,
                shape_grid_size=shape_grid_size,
                instance_masks=instance_masks,
                background_bgr=background_bgr,
            )
            accumulator = accumulators[condition.key]
            accumulator.transformed_objects += transform_stats.transformed_objects
            accumulator.skipped_objects += transform_stats.skipped_objects
            accumulator.ground_truths.extend(
                (image_id, class_id, box)
                for box, class_id in zip(gt_boxes, gt_classes, strict=True)
            )

            if image_index < preview_images:
                preview_dir = run_dir / "previews" / condition.key
                preview_dir.mkdir(parents=True, exist_ok=True)
                preview_name = f"{image_index:04d}_{image_path.stem}.jpg"
                cv2.imwrite(str(preview_dir / preview_name), transformed)

            pending.append(
                PendingInference(
                    condition=condition,
                    image_id=image_id,
                    image_bgr=transformed,
                    gt_boxes=gt_boxes,
                    gt_classes=gt_classes,
                )
            )
            if len(pending) >= inference_batch_size:
                _flush_inference(
                    model,
                    pending,
                    accumulators,
                    imgsz=imgsz,
                    confidence=confidence,
                    nms_iou=nms_iou,
                    match_iou=match_iou,
                    device=device,
                    max_det=max_det,
                    use_sahi=sahi,
                    slice_size=slice_size,
                    overlap=overlap,
                    sahi_batch_size=sahi_batch_size,
                    large_confidence=large_confidence,
                    size_threshold=size_threshold,
                )

    _flush_inference(
        model,
        pending,
        accumulators,
        imgsz=imgsz,
        confidence=confidence,
        nms_iou=nms_iou,
        match_iou=match_iou,
        device=device,
        max_det=max_det,
        use_sahi=sahi,
        slice_size=slice_size,
        overlap=overlap,
        sahi_batch_size=sahi_batch_size,
        large_confidence=large_confidence,
        size_threshold=size_threshold,
    )

    if total_objects == 0:
        raise typer.BadParameter(
            f"No YOLO objects were found in {labels_dir}; benchmark scores are undefined"
        )
    ground_truth_class_ids = sorted(
        {
            class_id
            for accumulator in accumulators.values()
            for _, class_id, _ in accumulator.ground_truths
        }
    )

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for condition in conditions:
        summary, class_rows = _compute_condition_metrics(
            accumulators[condition.key],
            ground_truth_class_ids,
        )
        summary_rows.append(summary)
        for row in class_rows:
            row["class_name"] = class_names[row["class_id"]]
        per_class_rows.extend(class_rows)

    metrics_by_condition = {
        row["key"]: {
            "map50": row["map50"],
            "map50_95": row["map50_95"],
        }
        for row in summary_rows
    }
    score_rows = (
        []
        if benchmark_version == "v2"
        else aggregate_shape_scores(
            metrics_by_condition,
            conditions,
            metric_name="map50_95",
        )
    )
    category_rows = aggregate_category_retention(
        metrics_by_condition,
        conditions,
        metric_name="map50_95",
    )
    reliance_auc_rows = (
        aggregate_reliance_auc(category_rows)
        if benchmark_version == "v2"
        else []
    )
    paired_rows = (
        _compute_paired_detection_rows(
            accumulators,
            conditions,
            confidence=paired_confidence,
            match_iou=match_iou,
        )
        if benchmark_version == "v2"
        else []
    )
    per_image_rows = [
        row
        for condition in conditions
        for row in accumulators[condition.key].per_image
    ]

    _write_csv(run_dir / "condition_metrics.csv", summary_rows)
    _write_csv(run_dir / "shape_reliance_scores.csv", score_rows)
    _write_csv(run_dir / "category_retention.csv", category_rows)
    _write_csv(run_dir / "cue_reliance_auc.csv", reliance_auc_rows)
    _write_csv(run_dir / "paired_detection_retention.csv", paired_rows)
    _write_csv(run_dir / "per_class_metrics.csv", per_class_rows)
    _write_csv(run_dir / "per_image_metrics.csv", per_image_rows)
    _write_markdown_report(
        run_dir / "report.md",
        summary_rows,
        score_rows,
        category_rows,
        reliance_auc_rows,
        paired_rows,
        weights=weights.resolve(),
        source=source.resolve(),
        benchmark_version=benchmark_version,
        region_source=region_source,
    )

    report = {
        "config": {
            "source": str(source.resolve()),
            "labels": str(labels_dir),
            "segments": str(segments_dir) if segments_dir else None,
            "weights": str(weights.resolve()),
            "benchmark_version": benchmark_version,
            "region_source": region_source,
            "imgsz": imgsz,
            "confidence": confidence,
            "nms_iou": nms_iou,
            "match_iou": match_iou,
            "paired_confidence": paired_confidence,
            "severities": parsed_severities,
            "hard_mode": hard_mode,
            "shape_grid_size": shape_grid_size,
            "seed": seed,
            "random_sample": random_sample,
            "sahi": sahi,
            "slice_size": slice_size,
            "overlap": overlap,
            "sahi_batch_size": sahi_batch_size,
            "large_confidence": large_confidence,
            "size_threshold": size_threshold,
        },
        "dataset": {
            "images_requested": len(image_paths),
            "images_with_label_files": labeled_images,
            "objects": total_objects,
            "classes": class_names,
        },
        "conditions": summary_rows,
        "shape_reliance_scores": score_rows,
        "category_retention": category_rows,
        "cue_reliance_auc": reliance_auc_rows,
        "paired_detection_retention": paired_rows,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))

    _print_results(
        summary_rows,
        score_rows,
        category_rows,
        reliance_auc_rows,
    )
    console.print(f"[green]Reports saved to:[/green] {run_dir}")


if __name__ == "__main__":
    app()
