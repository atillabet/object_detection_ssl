from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
BUILTIN_METHODS = {
    "supervised",
    "vicreg_global",
    "vicreg_object",
    "vicreg_mixed",
    "vicreg_opposed",
    "vicreg_shape_uniform",
    "vicreg_shape_region",
    "vicreg_shape_mismatched",
    "custom",
}


class ConfigError(ValueError):
    pass


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class MethodSpec:
    name: str
    kind: str
    raw: dict[str, Any]

    @property
    def pretraining(self) -> dict[str, Any]:
        return dict(self.raw.get("pretraining", {}))

    @property
    def finetuning(self) -> dict[str, Any]:
        return dict(self.raw.get("finetuning", {}))


@dataclass(frozen=True)
class StudyConfig:
    source: Path
    raw: dict[str, Any]
    methods: tuple[MethodSpec, ...]

    @property
    def name(self) -> str:
        return str(self.raw["name"])

    @property
    def project_root(self) -> Path:
        return self.resolve_path(self.raw["paths"]["project_root"])

    @property
    def python(self) -> Path:
        return self.resolve_path(self.raw["paths"]["python"])

    @property
    def artifact_root(self) -> Path:
        output_root = self.resolve_path(self.raw["paths"]["output_root"])
        return output_root / self.name

    @property
    def pretraining_root(self) -> Path:
        value = self.raw["paths"].get("pretraining_root")
        if value is not None:
            return self.resolve_path(value)
        return self.artifact_root / "pretraining"

    @property
    def gpu(self) -> int:
        return int(self.raw.get("runtime", {}).get("gpu", 0))

    @property
    def pretraining(self) -> dict[str, Any]:
        return dict(self.raw.get("pretraining", {}))

    @property
    def finetuning(self) -> dict[str, Any]:
        return dict(self.raw.get("finetuning", {}))

    @property
    def evaluations(self) -> dict[str, Any]:
        return dict(self.raw.get("evaluations", {}))

    @property
    def analysis(self) -> dict[str, Any]:
        return dict(self.raw.get("analysis", {}))

    @property
    def config_hash(self) -> str:
        return stable_hash(self.raw)

    def resolve_path(self, value: str | os.PathLike[str]) -> Path:
        expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
        return expanded if expanded.is_absolute() else (self.source.parent / expanded).resolve()

    def method(self, name: str) -> MethodSpec:
        for method in self.methods:
            if method.name == name:
                return method
        raise ConfigError(f"Unknown method: {name}")

    def merged_pretraining(self, method: MethodSpec) -> dict[str, Any]:
        return deep_merge(self.pretraining, method.pretraining)

    def merged_finetuning(self, method: MethodSpec) -> dict[str, Any]:
        return deep_merge(self.finetuning, method.finetuning)


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing {context}.{key}")
    return mapping[key]


def _require_entrypoint(mapping: Mapping[str, Any], context: str) -> None:
    entrypoints = [key for key in ("module", "script") if mapping.get(key)]
    if len(entrypoints) != 1:
        raise ConfigError(f"{context} requires exactly one of module or script")


def _validate_name(value: str, context: str) -> None:
    if not NAME_PATTERN.fullmatch(value):
        raise ConfigError(
            f"{context} must match {NAME_PATTERN.pattern!r}; received {value!r}"
        )


def _validate(raw: dict[str, Any]) -> tuple[MethodSpec, ...]:
    if raw.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")
    name = str(_require(raw, "name", "study"))
    _validate_name(name, "study.name")
    paths = _require(raw, "paths", "study")
    if not isinstance(paths, Mapping):
        raise ConfigError("study.paths must be a mapping")
    for key in ("project_root", "python", "output_root"):
        _require(paths, key, "paths")

    method_rows = _require(raw, "methods", "study")
    if not isinstance(method_rows, list) or not method_rows:
        raise ConfigError("study.methods must be a non-empty list")
    methods: list[MethodSpec] = []
    names: set[str] = set()
    for index, row in enumerate(method_rows):
        if not isinstance(row, Mapping):
            raise ConfigError(f"methods[{index}] must be a mapping")
        method_name = str(_require(row, "name", f"methods[{index}]"))
        kind = str(_require(row, "kind", f"methods[{index}]"))
        _validate_name(method_name, f"methods[{index}].name")
        if method_name in names:
            raise ConfigError(f"Duplicate method name: {method_name}")
        if kind not in BUILTIN_METHODS:
            raise ConfigError(f"Unsupported method kind: {kind}")
        if kind == "custom":
            custom = row.get("pretraining", {})
            if not isinstance(custom, Mapping) or "command" not in custom:
                raise ConfigError(
                    f"Custom method {method_name} requires pretraining.command"
                )
        names.add(method_name)
        methods.append(MethodSpec(method_name, kind, copy.deepcopy(dict(row))))

    pretraining = raw.get("pretraining", {})
    finetuning = raw.get("finetuning", {})
    if not isinstance(pretraining, Mapping) or not isinstance(finetuning, Mapping):
        raise ConfigError("pretraining and finetuning must be mappings")
    builtin_ssl = [
        method for method in methods if method.kind not in {"supervised", "custom"}
    ]
    if builtin_ssl:
        _require_entrypoint(pretraining, "pretraining")
        _require(pretraining, "args", "pretraining")
    _require_entrypoint(finetuning, "finetuning")
    _require(finetuning, "args", "finetuning")

    evaluations = raw.get("evaluations", {})
    if not isinstance(evaluations, Mapping):
        raise ConfigError("evaluations must be a mapping")
    shape = evaluations.get("shape_reliance")
    if shape and shape.get("enabled", True):
        _require_entrypoint(shape, "evaluations.shape_reliance")
        for key in ("images", "labels", "args"):
            _require(shape, key, "evaluations.shape_reliance")
    clean = evaluations.get("clean_validation")
    if clean and clean.get("enabled", True):
        _require_entrypoint(clean, "evaluations.clean_validation")
        for key in ("data", "args"):
            _require(clean, key, "evaluations.clean_validation")
    for index, row in enumerate(evaluations.get("custom", [])):
        if not isinstance(row, Mapping):
            raise ConfigError(f"evaluations.custom[{index}] must be a mapping")
        for key in ("name", "command", "completion"):
            _require(row, key, f"evaluations.custom[{index}]")

    analysis = raw.get("analysis", {})
    if analysis:
        baseline = analysis.get("baseline")
        if baseline is not None and baseline not in names:
            raise ConfigError(f"analysis.baseline is not a configured method: {baseline}")
        for criterion in analysis.get("criteria", []):
            for side in ("left", "right"):
                if criterion.get(side) not in names:
                    raise ConfigError(
                        f"Criterion {criterion.get('name', '<unnamed>')} has unknown {side}"
                    )
    return tuple(methods)


def load_study(path: str | Path) -> StudyConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ConfigError("Study YAML must contain a mapping at its root")
    methods = _validate(raw)
    return StudyConfig(source=source, raw=raw, methods=methods)
