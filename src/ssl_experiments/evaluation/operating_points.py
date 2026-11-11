"""Confidence selection and retention on explicitly shared target identities."""

from __future__ import annotations

from itertools import groupby
from typing import Any

import numpy as np

from ssl_experiments.evaluation.iou import iou_one_vs_many


def match_predictions(
    boxes, classes, predictions, *, threshold: float, match_iou: float
):
    """Return GT-index matches and confidence-ranked TP flags, one to one."""
    targets = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    class_ids = np.asarray(classes)
    matches: dict[int, tuple[float, float]] = {}
    ranked: list[tuple[float, bool]] = []
    for class_id, confidence, box in sorted(
        predictions, key=lambda p: p[1], reverse=True
    ):
        if confidence < threshold:
            continue
        correct = False
        if len(targets):
            ious = iou_one_vs_many(box, targets)
            ious[class_ids != class_id] = 0
            for index in matches:
                ious[index] = 0
            index = int(np.argmax(ious))
            if float(ious[index]) >= match_iou:
                matches[index] = (float(confidence), float(ious[index]))
                correct = True
        ranked.append((float(confidence), correct))
    return matches, ranked


def select_f1_threshold(
    records: dict[str, dict[str, Any]], *, match_iou: float
) -> dict:
    """Maximize micro F1 on clean validation; equal scores are never split."""
    ranked = []
    target_count = 0
    for record in records.values():
        target_count += len(record["boxes"])
        _, rows = match_predictions(
            record["boxes"],
            record["classes"],
            record["predictions"],
            threshold=0.0,
            match_iou=match_iou,
        )
        ranked.extend(rows)
    if not target_count:
        raise ValueError("F1 calibration requires annotated validation objects")
    best = {
        "threshold": 1.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": target_count,
    }
    tp = fp = 0
    for confidence, rows in groupby(
        sorted(ranked, reverse=True), key=lambda row: row[0]
    ):
        for _, correct in rows:
            tp += int(correct)
            fp += int(not correct)
        fn = target_count - tp
        f1 = 2 * tp / (2 * tp + fp + fn)
        # Strict comparison selects the higher threshold when F1 ties.
        if f1 > best["f1"] or (f1 == best["f1"] and confidence == best["threshold"]):
            best = {
                "threshold": confidence,
                "precision": tp / (tp + fp),
                "recall": tp / target_count,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
    return best


def clean_matches(
    records: dict[str, dict], *, threshold: float, match_iou: float
) -> dict:
    return {
        (image_id, index): value
        for image_id, record in records.items()
        for index, value in match_predictions(
            record["boxes"],
            record["classes"],
            record["predictions"],
            threshold=threshold,
            match_iou=match_iou,
        )[0].items()
    }


def common_targets(matches_by_model: dict[str, dict]) -> set[tuple[str, int]]:
    if len(matches_by_model) < 2:
        raise ValueError("Common-target comparison requires at least two models")
    common = set.intersection(*(set(matches) for matches in matches_by_model.values()))
    if not common:
        raise ValueError("The models have no shared correctly detected clean targets")
    return common


def common_retention(
    records: dict[str, dict],
    clean: dict,
    common: set,
    *,
    threshold: float,
    match_iou: float,
) -> dict:
    if not common or not common <= clean.keys():
        raise ValueError(
            "Common targets must be nonempty and initially correct for this model"
        )
    if not {image_id for image_id, _ in common} <= records.keys():
        raise ValueError("Intervention predictions omit required sample images")
    changed = clean_matches(records, threshold=threshold, match_iou=match_iou)
    retained = len(common & changed.keys())
    confidence_delta = sum(
        changed.get(key, (0.0, 0.0))[0] - clean[key][0] for key in common
    )
    iou_delta = sum(changed.get(key, (0.0, 0.0))[1] - clean[key][1] for key in common)
    return {
        "common_targets": len(common),
        "retained_targets": retained,
        "retention": retained / len(common),
        "mean_confidence_delta": confidence_delta / len(common),
        "mean_iou_delta": iou_delta / len(common),
    }
