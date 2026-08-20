from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ssl_experiments.commands import (
    clean_evaluation_dir,
    finetune_results,
    shape_evaluation_dir,
)
from ssl_experiments.config import MethodSpec, StudyConfig


def _metric_key(row: dict[str, str], fragment: str) -> str:
    matches = [key for key in row if fragment in key]
    if len(matches) != 1:
        raise ValueError(f"Expected one results.csv key containing {fragment!r}: {matches}")
    return matches[0]


def training_summary(path: Path) -> dict[str, float | int]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No training rows in {path}")
    map50_key = _metric_key(rows[0], "mAP50(B)")
    map95_key = _metric_key(rows[0], "mAP50-95")
    best = max(rows, key=lambda row: float(row[map95_key]))
    final = rows[-1]
    return {
        "epochs": len(rows),
        "best_epoch": int(float(best["epoch"])),
        "best_map50": float(best[map50_key]),
        "best_map50_95": float(best[map95_key]),
        "final_map50": float(final[map50_key]),
        "final_map50_95": float(final[map95_key]),
    }


def benchmark_summary(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    clean = next(row for row in report["conditions"] if row["key"] == "clean")
    reliance = {
        row["category"]: row["cue_reliance"]
        for row in report["cue_reliance_auc"]
        if row["cue_reliance"] is not None
    }
    appearance = (float(reliance["color"]) + float(reliance["texture"])) / 2.0
    return {
        "clean_map50": float(clean["map50"]),
        "clean_map50_95": float(clean["map50_95"]),
        "clean_recall": float(clean["recall"]),
        "color_reliance": float(reliance["color"]),
        "texture_reliance": float(reliance["texture"]),
        "appearance_reliance": appearance,
        "shape_reliance": float(reliance["shape"]),
        "shape_minus_appearance_gap": float(reliance["shape"]) - appearance,
    }


def clean_validation_summary(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    results = report.get("results", [])
    if len(results) != 1:
        raise ValueError(f"Expected one clean-validation result in {path}")
    row = results[0]
    return {
        f"validation_{key}": float(row[key])
        for key in ("precision", "recall", "map50", "map50_95")
        if key in row
    }


def method_summary(config: StudyConfig, method: MethodSpec) -> dict[str, Any]:
    result: dict[str, Any] = {"method": method.name, "kind": method.kind}
    result.update(training_summary(finetune_results(config, method)))
    clean_report = clean_evaluation_dir(config, method) / "clean_validation.json"
    if clean_report.is_file():
        result.update(clean_validation_summary(clean_report))
    report = shape_evaluation_dir(config, method) / "report.json"
    if report.is_file():
        result.update(benchmark_summary(report))
        sample = shape_evaluation_dir(config, method) / "sample_images.txt"
        if sample.is_file():
            result["sample_sha256"] = hashlib.sha256(sample.read_bytes()).hexdigest()
    return result


def criterion_result(
    criterion: dict[str, Any], by_method: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    metric = str(criterion["metric"])
    left = str(criterion["left"])
    right = str(criterion["right"])
    delta = float(by_method[left][metric]) - float(by_method[right][metric])
    threshold = float(criterion["threshold"])
    operator = str(criterion.get("operator", ">="))
    if operator == ">=":
        passed = delta >= threshold
    elif operator == "<=":
        passed = delta <= threshold
    else:
        raise ValueError(f"Unsupported criterion operator: {operator}")
    return {
        "name": criterion.get("name", f"{left}_{metric}_vs_{right}"),
        "left": left,
        "right": right,
        "metric": metric,
        "delta": delta,
        "operator": operator,
        "threshold": threshold,
        "passed": passed,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze_study(config: StudyConfig) -> dict[str, Any]:
    rows = [method_summary(config, method) for method in config.methods]
    by_method = {str(row["method"]): row for row in rows}
    sample_hashes = {row.get("sample_sha256") for row in rows if row.get("sample_sha256")}
    if len(sample_hashes) > 1:
        raise RuntimeError("Shape benchmark methods used different image samples")
    baseline = config.analysis.get("baseline")
    if baseline:
        baseline_row = by_method[str(baseline)]
        for row in rows:
            for metric in ("best_map50_95", "clean_map50_95", "shape_minus_appearance_gap"):
                if metric in row and metric in baseline_row:
                    row[f"{metric}_delta_vs_{baseline}"] = float(row[metric]) - float(
                        baseline_row[metric]
                    )
    criteria = [
        criterion_result(row, by_method)
        for row in config.analysis.get("criteria", [])
    ]
    result = {
        "study": config.name,
        "config_hash": config.config_hash,
        "same_benchmark_sample": len(sample_hashes) <= 1,
        "sample_sha256": next(iter(sample_hashes), None),
        "methods": rows,
        "criteria": criteria,
        "overall_success": all(row["passed"] for row in criteria) if criteria else None,
    }
    output = config.artifact_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output / "summary.csv", rows)

    lines = [
        f"# {config.name}",
        "",
        "| Method | Kind | Best mAP50-95 | Clean mAP50-95 | Shape reliance | Appearance reliance | Gap |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    def cell(row: dict[str, Any], key: str) -> str:
        return f"{float(row[key]):.4f}" if key in row else "n/a"

    for row in rows:
        lines.append(
            f"| {row['method']} | {row['kind']} | {cell(row, 'best_map50_95')} | "
            f"{cell(row, 'clean_map50_95')} | {cell(row, 'shape_reliance')} | "
            f"{cell(row, 'appearance_reliance')} | "
            f"{cell(row, 'shape_minus_appearance_gap')} |"
        )
    if criteria:
        lines.extend(
            [
                "",
                "## Criteria",
                "",
                "| Criterion | Delta | Required | Passed |",
                "|---|---:|---:|:---:|",
            ]
        )
        for row in criteria:
            lines.append(
                f"| {row['name']} | {row['delta']:+.4f} | "
                f"{row['operator']} {row['threshold']:+.4f} | {row['passed']} |"
            )
        lines.extend(("", f"Overall success: **{result['overall_success']}**"))
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
