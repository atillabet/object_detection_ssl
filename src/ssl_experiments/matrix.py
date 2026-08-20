from __future__ import annotations

import copy
import csv
import itertools
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from scipy.stats import t as student_t

from ssl_experiments.analysis import method_summary
from ssl_experiments.commands import build_pretrain_command
from ssl_experiments.config import StudyConfig, load_study, stable_hash
from ssl_experiments.data.splits import fraction_slug
from ssl_experiments.pipeline import Pipeline, STAGES


@dataclass(frozen=True)
class MatrixSpec:
    source: Path
    raw: dict[str, Any]
    base_study: Path
    axes: dict[str, tuple[Any, ...]]

    @property
    def name(self) -> str:
        return str(self.raw["name"])

    @property
    def config_hash(self) -> str:
        return stable_hash(self.raw)


@dataclass(frozen=True)
class MatrixCell:
    name: str
    axes: dict[str, Any]
    config: StudyConfig


def load_matrix(path: str | Path) -> MatrixSpec:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Matrix YAML must contain a mapping")
    if raw.get("schema_version") != 1:
        raise ValueError("Matrix schema_version must be 1")
    if not raw.get("name") or not raw.get("base_study"):
        raise ValueError("Matrix requires name and base_study")
    base = Path(str(raw["base_study"])).expanduser()
    base = base.resolve() if base.is_absolute() else (source.parent / base).resolve()
    if not base.is_file():
        raise FileNotFoundError(base)
    axes_raw = raw.get("axes")
    if not isinstance(axes_raw, Mapping) or not axes_raw:
        raise ValueError("Matrix axes must be a non-empty mapping")
    axes: dict[str, tuple[Any, ...]] = {}
    for name, values in axes_raw.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Matrix axis {name} must be a non-empty list")
        if len({json.dumps(value, sort_keys=True) for value in values}) != len(values):
            raise ValueError(f"Matrix axis {name} contains duplicates")
        axes[str(name)] = tuple(values)
    overrides = raw.get("overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("Matrix overrides must be a mapping")
    return MatrixSpec(source, raw, base, axes)


def _set_dotted(root: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current = root
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set matrix override through non-mapping: {dotted}")
        current = child
    current[parts[-1]] = value


def _format_value(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format_map(context)
    if isinstance(value, list):
        return [_format_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _format_value(item, context) for key, item in value.items()}
    return copy.deepcopy(value)


def axis_context(matrix: MatrixSpec, values: tuple[Any, ...]) -> dict[str, Any]:
    context = dict(zip(matrix.axes, values, strict=True))
    context["matrix"] = matrix.name
    if "fraction" in context:
        context["fraction_slug"] = fraction_slug(float(context["fraction"]))
    return context


def matrix_root(matrix: MatrixSpec, base: StudyConfig) -> Path:
    output_root = base.artifact_root.parent
    configured = matrix.raw.get("output_root")
    if configured is not None:
        path = Path(str(configured)).expanduser()
        output_root = path.resolve() if path.is_absolute() else (base.project_root / path).resolve()
    return output_root / matrix.name


def render_matrix(
    matrix: MatrixSpec,
    *,
    regenerate: bool = False,
) -> tuple[MatrixCell, ...]:
    base = load_study(matrix.base_study)
    root = matrix_root(matrix, base)
    configs_dir = root / "configs"
    cells: list[MatrixCell] = []
    shared_commands: dict[tuple[str, str], tuple[str, ...]] = {}
    for values in itertools.product(*matrix.axes.values()):
        context = axis_context(matrix, values)
        context["project_root"] = str(base.project_root)
        name_template = str(matrix.raw.get("cell_name", "{matrix}_" + "_".join(f"{{{axis}}}" for axis in matrix.axes)))
        cell_name = name_template.format_map(context)
        context["cell"] = cell_name
        raw = copy.deepcopy(base.raw)
        raw["name"] = cell_name
        raw["paths"]["project_root"] = str(base.project_root)
        raw["paths"]["python"] = str(base.python)
        raw["paths"]["output_root"] = str(root / "runs")
        clean = raw.get("evaluations", {}).get("clean_validation")
        if clean and clean.get("data"):
            clean["data"] = str(base.resolve_path(clean["data"]))
        shape = raw.get("evaluations", {}).get("shape_reliance")
        if shape:
            for key in ("images", "labels"):
                if shape.get(key):
                    shape[key] = str(base.resolve_path(shape[key]))
        group_template = str(matrix.raw.get("pretraining_group", cell_name))
        group = group_template.format_map(context)
        raw["paths"]["pretraining_root"] = str(root / "shared_pretraining" / group)
        for dotted, value in matrix.raw.get("overrides", {}).items():
            _set_dotted(raw, str(dotted), _format_value(value, context))

        config_path = configs_dir / f"{cell_name}.yaml"
        content = yaml.safe_dump(raw, sort_keys=False)
        if config_path.exists() and not regenerate:
            if config_path.read_text(encoding="utf-8") != content:
                raise RuntimeError(
                    f"Rendered matrix config differs: {config_path}; use --regenerate-configs"
                )
        else:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(content, encoding="utf-8")
        cell = MatrixCell(
            cell_name,
            dict(zip(matrix.axes, values, strict=True)),
            load_study(config_path),
        )
        for method in cell.config.methods:
            command = build_pretrain_command(cell.config, method)
            if command is None:
                continue
            key = (group, method.name)
            previous = shared_commands.setdefault(key, command.command)
            if previous != command.command:
                raise ValueError(
                    "Matrix pretraining_group renders different pretraining commands "
                    f"for group={group!r}, method={method.name!r}"
                )
        cells.append(cell)

    metadata = {
        "schema_version": 1,
        "name": matrix.name,
        "matrix_source": str(matrix.source),
        "matrix_config_hash": matrix.config_hash,
        "base_study": str(matrix.base_study),
        "axes": {key: list(values) for key, values in matrix.axes.items()},
        "cells": [
            {"name": cell.name, "axes": cell.axes, "config": str(cell.config.source)}
            for cell in cells
        ],
    }
    metadata_path = root / "matrix.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return tuple(cells)


def run_matrix(
    matrix: MatrixSpec,
    *,
    method_names: Iterable[str] | None = None,
    stages: Iterable[str] = STAGES,
    dry_run: bool = False,
    regenerate: bool = False,
) -> tuple[MatrixCell, ...]:
    cells = render_matrix(matrix, regenerate=regenerate)
    for cell in cells:
        print(f"\n=== matrix cell: {cell.name} {cell.axes} ===")
        Pipeline(cell.config, dry_run=dry_run).run(
            method_names=method_names,
            stages=stages,
        )
    return cells


def confidence_interval(values: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0, mean, mean
    standard_deviation = statistics.stdev(values)
    half_width = float(student_t.ppf(0.975, len(values) - 1)) * standard_deviation / math.sqrt(len(values))
    return mean, standard_deviation, mean - half_width, mean + half_width


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze_matrix(
    matrix: MatrixSpec,
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    cells = render_matrix(matrix)
    raw_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for cell in cells:
        for method in cell.config.methods:
            try:
                summary = method_summary(cell.config, method)
            except (FileNotFoundError, ValueError, StopIteration) as error:
                missing.append(f"{cell.name}/{method.name}: {error}")
                continue
            raw_rows.append({"cell": cell.name, **cell.axes, **summary})
    if missing and not allow_incomplete:
        raise RuntimeError("Incomplete matrix:\n" + "\n".join(missing))

    group_axes = [str(value) for value in matrix.raw.get("aggregate_by", ["fraction"])]
    numeric_metrics = sorted(
        {
            key
            for row in raw_rows
            for key, value in row.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and key not in {*matrix.axes, "epochs", "best_epoch"}
        }
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in raw_rows:
        key = tuple(row[axis] for axis in group_axes) + (row["method"],)
        grouped.setdefault(key, []).append(row)
    aggregates: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        aggregate: dict[str, Any] = {
            **dict(zip(group_axes, key[:-1], strict=True)),
            "method": key[-1],
            "n": len(rows),
        }
        for metric in numeric_metrics:
            values = [float(row[metric]) for row in rows if metric in row]
            if not values:
                continue
            mean, std, lower, upper = confidence_interval(values)
            aggregate[f"{metric}_mean"] = mean
            aggregate[f"{metric}_std"] = std
            aggregate[f"{metric}_ci95_lower"] = lower
            aggregate[f"{metric}_ci95_upper"] = upper
        aggregates.append(aggregate)

    baseline = matrix.raw.get("baseline")
    contrasts: list[dict[str, Any]] = []
    if baseline:
        raw_grouped = grouped
        for aggregate in aggregates:
            method = aggregate["method"]
            if method == baseline:
                continue
            group = tuple(aggregate[axis] for axis in group_axes)
            left_rows = raw_grouped[group + (method,)]
            right_rows = raw_grouped.get(group + (baseline,), [])
            pairing_axes = [axis for axis in matrix.axes if axis not in group_axes]
            right_by_pair = {tuple(row[axis] for axis in pairing_axes): row for row in right_rows}
            for metric in numeric_metrics:
                differences = [
                    float(row[metric]) - float(right_by_pair[tuple(row[axis] for axis in pairing_axes)][metric])
                    for row in left_rows
                    if metric in row
                    and tuple(row[axis] for axis in pairing_axes) in right_by_pair
                    and metric in right_by_pair[tuple(row[axis] for axis in pairing_axes)]
                ]
                if not differences:
                    continue
                mean, std, lower, upper = confidence_interval(differences)
                contrasts.append(
                    {
                        **dict(zip(group_axes, group, strict=True)),
                        "method": method,
                        "baseline": baseline,
                        "metric": metric,
                        "n": len(differences),
                        "mean_delta": mean,
                        "std_delta": std,
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                    }
                )

    base = load_study(matrix.base_study)
    output = matrix_root(matrix, base) / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "matrix": matrix.name,
        "matrix_config_hash": matrix.config_hash,
        "incomplete": missing,
        "rows": raw_rows,
        "aggregates": aggregates,
        "paired_contrasts": contrasts,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_csv(output / "runs.csv", raw_rows)
    _write_csv(output / "aggregates.csv", aggregates)
    _write_csv(output / "paired_contrasts.csv", contrasts)

    preferred_metric = str(matrix.raw.get("report_metric", "clean_map50_95"))
    lines = [
        f"# {matrix.name}",
        "",
        f"Aggregated by `{', '.join(group_axes)}` with 95% Student-t intervals.",
        "",
        f"| {' | '.join(group_axes)} | Method | n | {preferred_metric} mean | 95% CI |",
        f"| {' | '.join('---' for _ in group_axes)} |---|---:|---:|---:|",
    ]
    for row in aggregates:
        mean_key = f"{preferred_metric}_mean"
        if mean_key not in row:
            continue
        group_cells = " | ".join(str(row[axis]) for axis in group_axes)
        lines.append(
            f"| {group_cells} | {row['method']} | {row['n']} | {row[mean_key]:.4f} | "
            f"[{row[f'{preferred_metric}_ci95_lower']:.4f}, {row[f'{preferred_metric}_ci95_upper']:.4f}] |"
        )
    if missing:
        lines.extend(("", "## Incomplete", "", *[f"- {item}" for item in missing]))
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
