from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ssl_experiments.analysis import benchmark_summary


def load_rows(
    path: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, tuple[str, float]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped[row["condition"]].append(row)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    metadata: dict[str, tuple[str, float]] = {}
    for condition, rows in grouped.items():
        rows.sort(key=lambda row: row["image_id"])
        arrays[condition] = {
            "image_id": np.asarray([row["image_id"] for row in rows]),
            "tp": np.asarray([int(row["tp"]) for row in rows], dtype=np.int64),
            "fp": np.asarray([int(row["fp"]) for row in rows], dtype=np.int64),
            "fn": np.asarray([int(row["fn"]) for row in rows], dtype=np.int64),
        }
        metadata[condition] = (rows[0]["category"], float(rows[0]["severity"]))
    return arrays, metadata


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )


def summaries(
    arrays: dict[str, dict[str, np.ndarray]],
    metadata: dict[str, tuple[str, float]],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    condition_metrics: dict[str, dict[str, np.ndarray]] = {}
    for condition, values in arrays.items():
        tp = values["tp"][indices].sum(axis=1)
        fp = values["fp"][indices].sum(axis=1)
        fn = values["fn"][indices].sum(axis=1)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        condition_metrics[condition] = {
            "precision": precision,
            "recall": recall,
            "f1": safe_divide(2.0 * precision * recall, precision + recall),
        }

    clean = condition_metrics["clean"]
    grouped: dict[tuple[str, float], list[str]] = defaultdict(list)
    for condition, (category, severity) in metadata.items():
        if category not in {"clean", "diagnostic"}:
            grouped[(category, severity)].append(condition)
    retention: dict[tuple[str, float], np.ndarray] = {}
    for key, conditions in grouped.items():
        category_recall = np.mean(
            [condition_metrics[condition]["recall"] for condition in conditions],
            axis=0,
        )
        retention[key] = safe_divide(category_recall, clean["recall"])

    reliance: dict[str, np.ndarray] = {}
    for category in ("color", "shape", "texture"):
        points = sorted(
            (severity, values)
            for (row_category, severity), values in retention.items()
            if row_category == category
        )
        previous_x = 0.0
        previous_y = np.ones_like(clean["recall"])
        area = np.zeros_like(clean["recall"])
        for severity, values in points:
            area += (severity - previous_x) * (previous_y + values) / 2.0
            previous_x, previous_y = severity, values
        reliance[category] = 1.0 - area / previous_x
    appearance = (reliance["color"] + reliance["texture"]) / 2.0
    return {
        "clean_precision": clean["precision"],
        "clean_recall": clean["recall"],
        "clean_f1": clean["f1"],
        "shape_reliance": reliance["shape"],
        "appearance_reliance": appearance,
        "shape_minus_appearance_gap": reliance["shape"] - appearance,
    }


def interval(point: float, samples: np.ndarray) -> dict[str, float]:
    lower, upper = np.percentile(samples, [2.5, 97.5])
    positive = float(np.mean(samples > 0.0))
    return {
        "point_delta": point,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "probability_delta_positive": positive,
        "two_sided_bootstrap_p": min(1.0, 2.0 * min(positive, 1.0 - positive)),
    }


def paired_bootstrap(
    left_dir: Path,
    right_dir: Path,
    *,
    left_name: str,
    right_name: str,
    samples: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    if samples < 100:
        raise ValueError("samples must be at least 100")
    loaded = {
        left_name: load_rows(left_dir / "per_image_metrics.csv"),
        right_name: load_rows(right_dir / "per_image_metrics.csv"),
    }
    reference_arrays, reference_metadata = loaded[right_name]
    for name, (arrays, metadata) in loaded.items():
        if metadata != reference_metadata or arrays.keys() != reference_arrays.keys():
            raise RuntimeError(f"Condition definitions differ for {name}")
        for condition in arrays:
            if not np.array_equal(
                arrays[condition]["image_id"],
                reference_arrays[condition]["image_id"],
            ):
                raise RuntimeError(f"Image samples differ for {name}/{condition}")

    image_count = len(reference_arrays["clean"]["image_id"])
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(
        0, image_count, size=(samples, image_count), dtype=np.int32
    )
    point_indices = np.arange(image_count, dtype=np.int32)[None, :]
    boot = {
        name: summaries(arrays, metadata, bootstrap_indices)
        for name, (arrays, metadata) in loaded.items()
    }
    point = {
        name: summaries(arrays, metadata, point_indices)
        for name, (arrays, metadata) in loaded.items()
    }
    metrics = {
        metric: interval(
            float(point[left_name][metric][0] - point[right_name][metric][0]),
            boot[left_name][metric] - boot[right_name][metric],
        )
        for metric in point[left_name]
    }
    left_benchmark = benchmark_summary(left_dir / "report.json")
    right_benchmark = benchmark_summary(right_dir / "report.json")
    return {
        "contrast": f"{left_name}_minus_{right_name}",
        "method": "paired image bootstrap of micro TP/FP/FN metrics",
        "images": image_count,
        "bootstrap_samples": samples,
        "seed": seed,
        "limitation": (
            "Exact mAP intervals require confidence-ranked predictions, which are not "
            "stored in per_image_metrics.csv."
        ),
        "metrics": metrics,
        "observed_map_deltas_without_ci": {
            "clean_map50": left_benchmark["clean_map50"] - right_benchmark["clean_map50"],
            "clean_map50_95": left_benchmark["clean_map50_95"]
            - right_benchmark["clean_map50_95"],
        },
    }


def write_bootstrap(output: Path, result: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "bootstrap.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {result['contrast']}",
        "",
        f"{result['bootstrap_samples']:,} paired resamples of {result['images']:,} images.",
        "",
        "| Metric | Delta | 95% CI | P(delta > 0) |",
        "|---|---:|---:|---:|",
    ]
    for metric, values in result["metrics"].items():
        lines.append(
            f"| {metric.replace('_', ' ')} | {values['point_delta']:+.4f} | "
            f"[{values['ci95_lower']:+.4f}, {values['ci95_upper']:+.4f}] | "
            f"{values['probability_delta_positive']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Observed clean mAP50 delta without CI: "
            f"{result['observed_map_deltas_without_ci']['clean_map50']:+.4f}",
            f"Observed clean mAP50-95 delta without CI: "
            f"{result['observed_map_deltas_without_ci']['clean_map50_95']:+.4f}",
        ]
    )
    (output / "bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
