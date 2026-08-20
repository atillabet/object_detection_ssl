from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from ssl_experiments.data.manifests import generate_scaling_manifests
from ssl_experiments.data.splits import fraction_slug, generate_low_label_splits


def make_dataset(root: Path, count: int = 20) -> Path:
    images = root / "images/train"
    labels = root / "labels/train"
    validation = root / "images/val"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    validation.mkdir(parents=True)
    for index in range(count):
        name = f"image_{index:03d}"
        Image.new("RGB", (16, 16), color=(index, 0, 0)).save(images / f"{name}.jpg")
        (labels / f"{name}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    Image.new("RGB", (16, 16)).save(validation / "val.jpg")
    data = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "object"},
    }
    path = root / "data.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


class LowLabelSplitTests(unittest.TestCase):
    def test_generates_deterministic_nested_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = make_dataset(root)
            output = root / "splits"
            result = generate_low_label_splits(
                data,
                output,
                fractions=[0.5, 0.25],
                seeds=[1, 0],
            )
            first = set((output / "seed_0/f0p25/train.txt").read_text().splitlines())
            second = set((output / "seed_0/f0p5/train.txt").read_text().splitlines())
            repeated = generate_low_label_splits(
                data,
                output,
                fractions=[0.25, 0.5],
                seeds=[0, 1],
            )

        self.assertEqual(result, repeated)
        self.assertTrue(first < second)
        self.assertEqual(len(result["splits"]), 4)
        self.assertEqual(result["population_images"], 20)

    def test_fraction_slug_validates_range(self) -> None:
        self.assertEqual(fraction_slug(0.01), "f0p01")
        with self.assertRaises(ValueError):
            fraction_slug(0.0)


class ScalingManifestTests(unittest.TestCase):
    def test_generates_nested_compute_matched_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            paths = []
            for index in range(12):
                path = images / f"{index}.jpg"
                path.write_bytes(b"image")
                paths.append(path)
            population = root / "population.txt"
            population.write_text(
                "".join(f"{path}\n" for path in paths),
                encoding="utf-8",
            )
            result = generate_scaling_manifests(
                population,
                root / "manifests",
                sizes=[5, 10],
                epochs=[20, 10],
                batch=2,
                seed=4,
            )
            small = set((root / "manifests/unlabeled_n5.txt").read_text().splitlines())
            large = set((root / "manifests/unlabeled_n10.txt").read_text().splitlines())

        self.assertTrue(small < large)
        self.assertEqual(
            [row["optimizer_steps"] for row in result["manifests"]],
            [40, 50],
        )


if __name__ == "__main__":
    unittest.main()
