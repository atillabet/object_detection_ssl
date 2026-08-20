from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from ssl_experiments.data.splits import manifest_text, write_or_verify


def generate_scaling_manifests(
    population_manifest: Path,
    output: Path,
    *,
    sizes: Iterable[int],
    seed: int = 0,
    epochs: Iterable[int] | None = None,
    batch: int = 128,
    check_files: bool = True,
    regenerate: bool = False,
) -> dict[str, Any]:
    population_manifest = population_manifest.expanduser().resolve()
    output = output.expanduser().resolve()
    requested_sizes = sorted(set(int(value) for value in sizes))
    if not requested_sizes or requested_sizes[0] < 1:
        raise ValueError("Manifest sizes must be positive")
    if batch < 1:
        raise ValueError("Batch size must be positive")
    epoch_values = list(epochs) if epochs is not None else []
    if epoch_values and len(epoch_values) not in {1, len(requested_sizes)}:
        raise ValueError("Provide one epoch value or one value per manifest size")
    if len(epoch_values) == 1:
        epoch_values *= len(requested_sizes)

    images = [
        Path(line)
        for line in population_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(images) < requested_sizes[-1]:
        raise ValueError(
            f"Population has {len(images):,} images but {requested_sizes[-1]:,} were requested"
        )
    if len(set(images)) != len(images):
        raise ValueError("Population manifest contains duplicate paths")
    if check_files:
        missing = [path for path in images if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Population contains {len(missing)} missing images; first: {missing[0]}"
            )

    shuffled = images.copy()
    random.Random(seed).shuffle(shuffled)
    rows: list[dict[str, Any]] = []
    previous: set[Path] = set()
    for index, size in enumerate(requested_sizes):
        selected = shuffled[:size]
        selected_set = set(selected)
        if not previous.issubset(selected_set):
            raise AssertionError("Scaling manifests must be nested")
        previous = selected_set
        path = output / f"unlabeled_n{size}.txt"
        content = manifest_text(selected)
        write_or_verify(path, content, regenerate=regenerate)
        steps = size // batch
        row: dict[str, Any] = {
            "images": size,
            "manifest": str(path),
            "manifest_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "steps_per_epoch_drop_last": steps,
        }
        if epoch_values:
            row["epochs"] = int(epoch_values[index])
            row["optimizer_steps"] = steps * int(epoch_values[index])
        rows.append(row)

    result = {
        "schema_version": 1,
        "population_manifest": str(population_manifest),
        "population_sha256": hashlib.sha256(population_manifest.read_bytes()).hexdigest(),
        "population_images": len(images),
        "selection_seed": seed,
        "nested_prefix_selection": True,
        "batch": batch,
        "manifests": rows,
    }
    write_or_verify(
        output / "manifests.json",
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        regenerate=regenerate,
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic nested unlabeled-data manifests."
    )
    parser.add_argument("population", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/manifests"))
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, nargs="+")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args(argv)
    result = generate_scaling_manifests(
        args.population,
        args.output,
        sizes=args.sizes,
        seed=args.seed,
        epochs=args.epochs,
        batch=args.batch,
        check_files=args.check_files,
        regenerate=args.regenerate,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
