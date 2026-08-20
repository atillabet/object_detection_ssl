from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import yaml

from ssl_experiments.data.yolo import IMAGE_EXTENSIONS


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def fraction_slug(fraction: float) -> str:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fractions must be in the interval (0, 1]")
    return "f" + f"{fraction:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def resolve_dataset_path(value: str, dataset_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (dataset_root / path).resolve()


def load_source_dataset(data_yaml: Path) -> tuple[dict[str, Any], Path, Path]:
    data_yaml = data_yaml.expanduser().resolve()
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset YAML is not a mapping: {data_yaml}")
    raw_root = Path(str(payload.get("path", data_yaml.parent))).expanduser()
    dataset_root = (
        raw_root.resolve()
        if raw_root.is_absolute()
        else (data_yaml.parent / raw_root).resolve()
    )
    train_value = payload.get("train")
    val_value = payload.get("val")
    if not isinstance(train_value, str) or not isinstance(val_value, str):
        raise ValueError("Split generation requires one train path and one val path")
    train_path = resolve_dataset_path(train_value, dataset_root)
    val_path = resolve_dataset_path(val_value, dataset_root)
    if not train_path.is_dir():
        raise FileNotFoundError(f"Training image directory does not exist: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation image path does not exist: {val_path}")
    if payload.get("names") is None:
        raise ValueError(f"Dataset YAML has no class names: {data_yaml}")
    return payload, train_path, val_path


def discover_images(train_dir: Path) -> list[Path]:
    images = sorted(
        (
            entry.absolute()
            for entry in train_dir.iterdir()
            if entry.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=str,
    )
    if not images:
        raise ValueError(f"No supported images found in {train_dir}")
    return images


def manifest_text(images: Iterable[Path]) -> str:
    return "".join(f"{path}\n" for path in images)


def write_or_verify(
    path: Path,
    content: str,
    *,
    regenerate: bool,
) -> None:
    if path.exists() and not regenerate:
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(
                f"Existing reproducibility artifact differs: {path}; "
                "use --regenerate only when intentionally replacing it"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def generate_low_label_splits(
    data_yaml: Path,
    output: Path,
    *,
    fractions: Iterable[float],
    seeds: Iterable[int],
    regenerate: bool = False,
) -> dict[str, Any]:
    data_yaml = data_yaml.expanduser().resolve()
    output = output.expanduser().resolve()
    selected_fractions = sorted(set(float(value) for value in fractions))
    selected_seeds = sorted(set(int(value) for value in seeds))
    if not selected_fractions or not selected_seeds:
        raise ValueError("At least one fraction and seed are required")
    for fraction in selected_fractions:
        fraction_slug(fraction)

    source, train_dir, val_path = load_source_dataset(data_yaml)
    images = discover_images(train_dir)
    population_content = manifest_text(images)
    population_path = output / "population.txt"
    write_or_verify(population_path, population_content, regenerate=regenerate)
    population_hash = sha256_bytes(population_content.encode())
    if train_dir.name == "images":
        labels_dir = train_dir.parent / "labels"
    elif train_dir.parent.name == "images":
        labels_dir = train_dir.parent.parent / "labels" / train_dir.name
    else:
        labels_dir = train_dir.parent / "labels"

    rows: list[dict[str, Any]] = []
    for seed in selected_seeds:
        shuffled = images.copy()
        random.Random(seed).shuffle(shuffled)
        previous: set[Path] = set()
        for fraction in selected_fractions:
            count = max(1, math.ceil(len(shuffled) * fraction))
            selected = shuffled[:count]
            selected_set = set(selected)
            if not previous.issubset(selected_set):
                raise AssertionError("Fraction manifests must be nested within each seed")
            previous = selected_set

            slug = fraction_slug(fraction)
            split_dir = output / f"seed_{seed}" / slug
            train_manifest = split_dir / "train.txt"
            selected_content = manifest_text(selected)
            write_or_verify(train_manifest, selected_content, regenerate=regenerate)
            split_hash = sha256_bytes(selected_content.encode())

            generated_data = {
                "train": "train.txt",
                "val": str(val_path),
                "nc": int(source.get("nc", len(source["names"]))),
                "names": source["names"],
            }
            write_or_verify(
                split_dir / "data.yaml",
                yaml.safe_dump(generated_data, sort_keys=False),
                regenerate=regenerate,
            )
            present_labels = sum(
                (labels_dir / image.relative_to(train_dir).with_suffix(".txt")).exists()
                for image in selected
            )
            metadata = {
                "schema_version": 1,
                "source_data_yaml": str(data_yaml),
                "source_data_yaml_sha256": sha256_file(data_yaml),
                "population_manifest": str(population_path),
                "population_sha256": population_hash,
                "population_images": len(images),
                "fraction": fraction,
                "selected_images": count,
                "seed": seed,
                "nested_prefix_selection": True,
                "train_manifest": str(train_manifest),
                "train_manifest_sha256": split_hash,
                "label_files_present": present_labels,
                "images_without_label_file": count - present_labels,
            }
            write_or_verify(
                split_dir / "metadata.json",
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                regenerate=regenerate,
            )
            rows.append(
                {
                    "seed": seed,
                    "fraction": fraction,
                    "fraction_slug": slug,
                    "selected_images": count,
                    "data_yaml": str(split_dir / "data.yaml"),
                    "train_manifest": str(train_manifest),
                    "train_manifest_sha256": split_hash,
                }
            )

    result = {
        "schema_version": 1,
        "source_data_yaml": str(data_yaml),
        "source_data_yaml_sha256": sha256_file(data_yaml),
        "population_manifest": str(population_path),
        "population_sha256": population_hash,
        "population_images": len(images),
        "fractions": selected_fractions,
        "seeds": selected_seeds,
        "nested_fractions_within_seed": True,
        "splits": rows,
    }
    write_or_verify(
        output / "splits.json",
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        regenerate=regenerate,
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic nested low-label YOLO splits."
    )
    parser.add_argument("data", type=Path, help="Source YOLO data.yaml")
    parser.add_argument("--output", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.10, 0.25],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args(argv)
    result = generate_low_label_splits(
        args.data,
        args.output,
        fractions=args.fractions,
        seeds=args.seeds,
        regenerate=args.regenerate,
    )
    print(json.dumps({key: result[key] for key in ("population_images", "fractions", "seeds")}, indent=2))


if __name__ == "__main__":
    main()
