#!/usr/bin/env python3

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
import yaml
from PIL import Image
from rich.console import Console
from rich.table import Table

from ssl_experiments.data.yolo import IMAGE_EXTENSIONS, parse_yolo_yaml

app = typer.Typer()
console = Console()


@dataclass
class Box:
    cls_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


def resolve_split_dir(dataset_root: Path, split_value: str) -> tuple[Path, Path]:
    images_dir = (dataset_root / split_value).resolve()
    if images_dir.exists():
        labels_dir = Path(str(images_dir).replace("/images", "/labels", 1))
        return images_dir, labels_dir

    candidate_img = dataset_root / split_value / "images"
    candidate_lbl = dataset_root / split_value / "labels"
    if candidate_img.exists() and candidate_lbl.exists():
        return candidate_img, candidate_lbl

    raise FileNotFoundError(f"Could not resolve split path '{split_value}' under {dataset_root}")


def load_split_samples(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    image_files = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    return [(img, labels_dir / f"{img.stem}.txt") for img in image_files]


def read_boxes(label_path: Path, img_w: int, img_h: int) -> list[Box]:
    if not label_path.exists():
        return []
    boxes: list[Box] = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        x1 = (xc - w / 2.0) * img_w
        y1 = (yc - h / 2.0) * img_h
        x2 = (xc + w / 2.0) * img_w
        y2 = (yc + h / 2.0) * img_h
        boxes.append(Box(cls_id=cls_id, x1=x1, y1=y1, x2=x2, y2=y2))
    return boxes


def clip_box_to_crop(box: Box, crop: tuple[int, int, int, int]) -> Box | None:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop
    x1 = max(box.x1, crop_x1)
    y1 = max(box.y1, crop_y1)
    x2 = min(box.x2, crop_x2)
    y2 = min(box.y2, crop_y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return Box(cls_id=box.cls_id, x1=x1 - crop_x1, y1=y1 - crop_y1, x2=x2 - crop_x1, y2=y2 - crop_y1)


def box_visible_enough(original: Box, clipped: Box, min_visible_fraction: float, min_box_size: int) -> bool:
    if clipped.width < min_box_size or clipped.height < min_box_size:
        return False
    if original.area <= 0:
        return False
    return (clipped.area / original.area) >= min_visible_fraction


def yolo_line(box: Box, crop_w: int, crop_h: int) -> str:
    xc = ((box.x1 + box.x2) / 2.0) / crop_w
    yc = ((box.y1 + box.y2) / 2.0) / crop_h
    bw = box.width / crop_w
    bh = box.height / crop_h
    return f"{box.cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def clamp_crop(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(int(round(x1)), img_w - 1))
    y1 = max(0, min(int(round(y1)), img_h - 1))
    x2 = max(x1 + 1, min(int(round(x2)), img_w))
    y2 = max(y1 + 1, min(int(round(y2)), img_h))
    return x1, y1, x2, y2


def make_object_centered_crop(
    target: Box,
    img_w: int,
    img_h: int,
    margin_range: tuple[float, float],
    jitter_fraction: float,
    square_prob: float,
) -> tuple[int, int, int, int]:
    margin = random.uniform(*margin_range)
    half_w = target.width * (0.5 + margin)
    half_h = target.height * (0.5 + margin)

    if random.random() < square_prob:
        side = max(half_w * 2.0, half_h * 2.0)
        half_w = side / 2.0
        half_h = side / 2.0

    jitter_x = random.uniform(-jitter_fraction, jitter_fraction) * max(target.width, 1.0)
    jitter_y = random.uniform(-jitter_fraction, jitter_fraction) * max(target.height, 1.0)
    cx = target.cx + jitter_x
    cy = target.cy + jitter_y

    return clamp_crop(cx - half_w, cy - half_h, cx + half_w, cy + half_h, img_w, img_h)


def choose_aug_sources(samples: list[tuple[Path, Path]], img_sizes: dict[Path, tuple[int, int]], per_image_boxes: dict[Path, list[Box]]):
    sources = []
    for image_path, label_path in samples:
        boxes = per_image_boxes[image_path]
        if not boxes:
            continue
        sources.append((image_path, label_path, boxes, img_sizes[image_path]))
    if not sources:
        raise ValueError("No labeled training images found for object-centered augmentation.")
    return sources


def copy_dataset_tree(src_root: Path, dst_root: Path) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    shutil.copytree(src_root, dst_root)


def write_data_yaml(data: dict, out_path: Path) -> None:
    out_path.write_text(yaml.safe_dump(data, sort_keys=False))


def generate_variant(
    dataset_root: Path,
    dataset_config: dict,
    out_root: Path,
    ratio: float,
    margin_min: float,
    margin_max: float,
    jitter_fraction: float,
    min_visible_fraction: float,
    min_box_size: int,
    square_prob: float,
    seed: int,
) -> dict[str, int | str]:
    random.seed(seed)
    variant_name = f"{dataset_root.name}_objcentered_{int(round(ratio * 100)):02d}"
    variant_root = out_root / variant_name
    copy_dataset_tree(dataset_root, variant_root)

    train_key = "train"
    if train_key not in dataset_config:
        raise ValueError("Dataset config must contain a 'train' split.")

    train_images_dir, train_labels_dir = resolve_split_dir(variant_root, dataset_config[train_key])
    samples = load_split_samples(train_images_dir, train_labels_dir)
    if not samples:
        raise ValueError(f"No training images found in {train_images_dir}")

    img_sizes: dict[Path, tuple[int, int]] = {}
    per_image_boxes: dict[Path, list[Box]] = {}
    for image_path, label_path in samples:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
        img_sizes[image_path] = (img_w, img_h)
        per_image_boxes[image_path] = read_boxes(label_path, img_w, img_h)

    aug_sources = choose_aug_sources(samples, img_sizes, per_image_boxes)
    num_original = len(samples)
    num_augmented = max(1, int(round(num_original * ratio)))
    next_index = 1
    created = 0

    for _ in range(num_augmented):
        image_path, _label_path, boxes, (img_w, img_h) = random.choice(aug_sources)
        target = random.choice(boxes)
        crop = make_object_centered_crop(
            target=target,
            img_w=img_w,
            img_h=img_h,
            margin_range=(margin_min, margin_max),
            jitter_fraction=jitter_fraction,
            square_prob=square_prob,
        )

        with Image.open(image_path) as img:
            crop_img = img.crop(crop)

        crop_boxes: list[Box] = []
        for box in boxes:
            clipped = clip_box_to_crop(box, crop)
            if clipped is None:
                continue
            if box_visible_enough(box, clipped, min_visible_fraction=min_visible_fraction, min_box_size=min_box_size):
                crop_boxes.append(clipped)

        if not crop_boxes:
            continue

        out_stem = f"{image_path.stem}_objcentered_{next_index:06d}"
        next_index += 1
        out_image = train_images_dir / f"{out_stem}{image_path.suffix.lower()}"
        out_label = train_labels_dir / f"{out_stem}.txt"
        crop_img.save(out_image)
        crop_w, crop_h = crop_img.size
        out_label.write_text("\n".join(yolo_line(box, crop_w, crop_h) for box in crop_boxes) + "\n")
        created += 1

    write_data_yaml(dataset_config, variant_root / "data.yaml")
    return {
        "variant": variant_name,
        "output": str(variant_root),
        "original_train_images": num_original,
        "requested_augmented_images": num_augmented,
        "created_augmented_images": created,
        "train_images_total": num_original + created,
    }


@app.command()
def main(
    dataset: Annotated[Path, typer.Argument(help="Path to the source YOLO dataset root containing data.yaml")],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output directory for generated dataset variants")] = Path(
        "artifacts/datasets/object_centered"
    ),
    ratio: Annotated[list[float], typer.Option("--ratio", help="Augmentation ratio(s) to generate, e.g. --ratio 0.15 --ratio 0.2")] = [0.15, 0.20],
    margin_min: Annotated[float, typer.Option(help="Minimum extra margin around the target box, as a fraction of box size")] = 0.15,
    margin_max: Annotated[float, typer.Option(help="Maximum extra margin around the target box, as a fraction of box size")] = 0.30,
    jitter_fraction: Annotated[float, typer.Option(help="Random crop-center jitter as a fraction of box size")] = 0.10,
    min_visible_fraction: Annotated[float, typer.Option(help="Minimum retained box area fraction to keep a label inside a crop")] = 0.60,
    min_box_size: Annotated[int, typer.Option(help="Minimum width/height in pixels to keep a cropped label")] = 4,
    square_prob: Annotated[float, typer.Option(help="Probability of making the crop square")] = 0.70,
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
):
    dataset = dataset.resolve()
    data_yaml = dataset / "data.yaml"
    if not data_yaml.exists():
        raise typer.BadParameter(f"Expected data.yaml at {data_yaml}")

    dataset_config = parse_yolo_yaml(data_yaml)
    output.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, r in enumerate(ratio):
        if not (0 < r < 1):
            raise typer.BadParameter(f"Each --ratio must be between 0 and 1, got {r}")
        console.print(f"[cyan]Generating variant for ratio {r:.2f}[/cyan]")
        result = generate_variant(
            dataset_root=dataset,
            dataset_config=dataset_config,
            out_root=output,
            ratio=r,
            margin_min=margin_min,
            margin_max=margin_max,
            jitter_fraction=jitter_fraction,
            min_visible_fraction=min_visible_fraction,
            min_box_size=min_box_size,
            square_prob=square_prob,
            seed=seed + idx,
        )
        results.append(result)

    table = Table(title="Object-Centered Dataset Variants")
    table.add_column("Variant")
    table.add_column("Original Train", justify="right")
    table.add_column("Requested Aug", justify="right")
    table.add_column("Created Aug", justify="right")
    table.add_column("Train Total", justify="right")
    table.add_column("Output")
    for row in results:
        table.add_row(
            str(row["variant"]),
            str(row["original_train_images"]),
            str(row["requested_augmented_images"]),
            str(row["created_augmented_images"]),
            str(row["train_images_total"]),
            str(row["output"]),
        )
    console.print(table)


if __name__ == "__main__":
    app()
