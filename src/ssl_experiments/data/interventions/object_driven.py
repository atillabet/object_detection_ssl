#!/usr/bin/env python3

from __future__ import annotations

import io
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Iterable

import numpy as np
import typer
import yaml
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from rich.console import Console
from rich.table import Table

from ssl_experiments.data.yolo import IMAGE_EXTENSIONS

app = typer.Typer()
console = Console()


@dataclass(frozen=True)
class Box:
    class_id: int
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


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path
    boxes: tuple[Box, ...]
    width: int
    height: int

    @property
    def has_objects(self) -> bool:
        return bool(self.boxes)


@dataclass
class AugmentStats:
    originals_linked: int = 0
    copy_paste: int = 0
    background_dropout: int = 0
    tight_crop: int = 0
    robust: int = 0
    skipped: int = 0


def find_data_yaml(source: Path) -> Path:
    if source.is_file():
        return source
    for name in ("data.yaml", "dataset.yaml"):
        candidate = source / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find data.yaml in {source}")


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def label_dir_for_images(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == "images":
            parts[idx] = "labels"
            return Path(*parts)
    return images_dir.parent / "labels"


def split_image_dirs(dataset_root: Path, config: dict, split: str) -> list[Path]:
    value = config.get(split)
    if value is None and split == "val":
        value = config.get("valid")
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [resolve_path(dataset_root, str(item)) for item in value]
    return [resolve_path(dataset_root, str(value))]


def label_for_image(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def read_boxes(label_path: Path, width: int, height: int) -> tuple[Box, ...]:
    if not label_path.exists():
        return ()
    boxes: list[Box] = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        x_c, y_c, box_w, box_h = map(float, parts[1:5])
        x1 = (x_c - box_w / 2.0) * width
        y1 = (y_c - box_h / 2.0) * height
        x2 = (x_c + box_w / 2.0) * width
        y2 = (y_c + box_h / 2.0) * height
        if x2 > x1 and y2 > y1:
            boxes.append(Box(cls_id, x1, y1, x2, y2))
    return tuple(boxes)


def collect_samples(dataset_root: Path, config: dict, split: str, max_samples: int | None = None) -> list[Sample]:
    samples: list[Sample] = []
    for images_dir in split_image_dirs(dataset_root, config, split):
        labels_dir = label_dir_for_images(images_dir)
        image_files = sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        for image_path in image_files:
            label_path = label_for_image(image_path, labels_dir)
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception:
                continue
            samples.append(Sample(image_path, label_path, read_boxes(label_path, width, height), width, height))
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    return samples


def box_to_yolo(box: Box, width: int, height: int) -> str:
    x_c = ((box.x1 + box.x2) / 2.0) / width
    y_c = ((box.y1 + box.y2) / 2.0) / height
    box_w = box.width / width
    box_h = box.height / height
    x_c = min(1.0, max(0.0, x_c))
    y_c = min(1.0, max(0.0, y_c))
    box_w = min(1.0, max(0.0, box_w))
    box_h = min(1.0, max(0.0, box_h))
    return f"{box.class_id} {x_c:.6f} {y_c:.6f} {box_w:.6f} {box_h:.6f}"


def write_labels(label_path: Path, boxes: Iterable[Box], width: int, height: int) -> None:
    lines = [box_to_yolo(box, width, height) for box in boxes if box.width > 0 and box.height > 0]
    label_path.write_text(("\n".join(lines) + "\n") if lines else "")


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def symlink_or_copy(src: Path, dst: Path, *, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def link_split(samples: list[Sample], out_images: Path, out_labels: Path, *, copy_files: bool) -> int:
    linked = 0
    for sample in samples:
        symlink_or_copy(sample.image_path, out_images / sample.image_path.name, copy_files=copy_files)
        if sample.label_path.exists():
            symlink_or_copy(sample.label_path, out_labels / f"{sample.image_path.stem}.txt", copy_files=copy_files)
        else:
            (out_labels / f"{sample.image_path.stem}.txt").write_text("")
        linked += 1
    return linked


def crop_with_margin(box: Box, width: int, height: int, margin: float) -> tuple[int, int, int, int]:
    dx = box.width * margin
    dy = box.height * margin
    x1 = int(max(0, round(box.x1 - dx)))
    y1 = int(max(0, round(box.y1 - dy)))
    x2 = int(min(width, round(box.x2 + dx)))
    y2 = int(min(height, round(box.y2 + dy)))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def clip_box_to_crop(box: Box, crop: tuple[int, int, int, int]) -> Box | None:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop
    x1 = max(box.x1, crop_x1)
    y1 = max(box.y1, crop_y1)
    x2 = min(box.x2, crop_x2)
    y2 = min(box.y2, crop_y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return Box(box.class_id, x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1)


def visible_enough(original: Box, clipped: Box, min_visible_fraction: float, min_box_size: int) -> bool:
    if clipped.width < min_box_size or clipped.height < min_box_size or original.area <= 0:
        return False
    return clipped.area / original.area >= min_visible_fraction


def random_tight_crop(
    sample: Sample,
    rng: random.Random,
    *,
    margin_min: float,
    margin_max: float,
    jitter_fraction: float,
    min_visible_fraction: float,
    min_box_size: int,
) -> tuple[Image.Image, list[Box]] | None:
    if not sample.boxes:
        return None
    target = rng.choice(sample.boxes)
    margin = rng.uniform(margin_min, margin_max)
    half_w = target.width * (0.5 + margin)
    half_h = target.height * (0.5 + margin)
    jitter_x = rng.uniform(-jitter_fraction, jitter_fraction) * max(target.width, 1.0)
    jitter_y = rng.uniform(-jitter_fraction, jitter_fraction) * max(target.height, 1.0)
    cx = target.cx + jitter_x
    cy = target.cy + jitter_y
    x1 = int(max(0, round(cx - half_w)))
    y1 = int(max(0, round(cy - half_h)))
    x2 = int(min(sample.width, round(cx + half_w)))
    y2 = int(min(sample.height, round(cy + half_h)))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = (x1, y1, x2, y2)
    crop_boxes: list[Box] = []
    for box in sample.boxes:
        clipped = clip_box_to_crop(box, crop)
        if clipped is not None and visible_enough(box, clipped, min_visible_fraction, min_box_size):
            crop_boxes.append(clipped)
    if not crop_boxes:
        return None
    with Image.open(sample.image_path) as img:
        return img.convert("RGB").crop(crop), crop_boxes


def expanded_box_mask(width: int, height: int, boxes: Iterable[Box], margin: float) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for box in boxes:
        dx = box.width * margin
        dy = box.height * margin
        x1 = int(max(0, round(box.x1 - dx)))
        y1 = int(max(0, round(box.y1 - dy)))
        x2 = int(min(width, round(box.x2 + dx)))
        y2 = int(min(height, round(box.y2 + dy)))
        mask[y1:y2, x1:x2] = True
    return mask


def background_dropout_image(img: Image.Image, boxes: tuple[Box, ...], rng: random.Random, margin: float) -> Image.Image:
    rgb = img.convert("RGB")
    width, height = rgb.size
    protected = expanded_box_mask(width, height, boxes, margin)
    mode = rng.choice(["blur", "gray", "noise", "darken", "mean"])

    if mode == "blur":
        transformed = rgb.filter(ImageFilter.GaussianBlur(radius=rng.uniform(4.0, 10.0)))
        arr = np.asarray(rgb).copy()
        tarr = np.asarray(transformed)
    elif mode == "gray":
        transformed = ImageOps.grayscale(rgb).convert("RGB")
        arr = np.asarray(rgb).copy()
        tarr = np.asarray(transformed)
    else:
        arr = np.asarray(rgb).astype(np.float32)
        tarr = arr.copy()
        if mode == "noise":
            noise = np.random.default_rng(rng.randrange(2**32)).normal(0.0, rng.uniform(16.0, 34.0), arr.shape)
            tarr = np.clip(arr + noise, 0, 255)
        elif mode == "darken":
            tarr = np.clip(arr * rng.uniform(0.35, 0.70), 0, 255)
        elif mode == "mean":
            mean_color = arr.reshape(-1, 3).mean(axis=0)
            tarr[...] = mean_color.reshape(1, 1, 3)
        arr = arr.astype(np.uint8)
        tarr = tarr.astype(np.uint8)

    arr[~protected] = tarr[~protected]
    return Image.fromarray(arr, mode="RGB")


def robust_image(img: Image.Image, rng: random.Random) -> Image.Image:
    out = img.convert("RGB")
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.75, 1.25))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.75, 1.35))
    out = ImageEnhance.Color(out).enhance(rng.uniform(0.65, 1.35))

    if rng.random() < 0.65:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.6)))

    if rng.random() < 0.65:
        arr = np.asarray(out).astype(np.float32)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0.0, rng.uniform(4.0, 14.0), arr.shape)
        out = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8), mode="RGB")

    if rng.random() < 0.50:
        buffer = io.BytesIO()
        out.save(buffer, format="JPEG", quality=rng.randint(35, 80))
        buffer.seek(0)
        out = Image.open(buffer).convert("RGB")

    return out


def box_iou(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def paste_object(
    object_sample: Sample,
    background_sample: Sample,
    rng: random.Random,
    *,
    crop_margin: float,
    scale_min: float,
    scale_max: float,
    min_box_size: int,
    max_iou: float,
    attempts: int,
) -> tuple[Image.Image, list[Box]] | None:
    if not object_sample.boxes:
        return None

    obj_box = rng.choice(object_sample.boxes)
    crop = crop_with_margin(obj_box, object_sample.width, object_sample.height, crop_margin)
    crop_x1, crop_y1, crop_x2, crop_y2 = crop
    obj_in_crop = Box(
        obj_box.class_id,
        obj_box.x1 - crop_x1,
        obj_box.y1 - crop_y1,
        obj_box.x2 - crop_x1,
        obj_box.y2 - crop_y1,
    )

    with Image.open(object_sample.image_path) as src_img, Image.open(background_sample.image_path) as bg_img:
        patch = src_img.convert("RGB").crop(crop)
        bg = bg_img.convert("RGB")

    scale = rng.uniform(scale_min, scale_max)
    new_w = max(1, int(round(patch.width * scale)))
    new_h = max(1, int(round(patch.height * scale)))
    if new_w >= background_sample.width or new_h >= background_sample.height:
        scale = min((background_sample.width - 1) / max(1, patch.width), (background_sample.height - 1) / max(1, patch.height))
        if scale <= 0:
            return None
        new_w = max(1, int(round(patch.width * scale)))
        new_h = max(1, int(round(patch.height * scale)))

    pasted_box_w = obj_in_crop.width * scale
    pasted_box_h = obj_in_crop.height * scale
    if pasted_box_w < min_box_size or pasted_box_h < min_box_size:
        return None

    patch = patch.resize((new_w, new_h), Image.Resampling.BILINEAR)
    alpha = Image.new("L", patch.size, 255)
    feather = max(1, int(round(min(patch.size) * 0.04)))
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather))

    existing = list(background_sample.boxes)
    for _ in range(attempts):
        px = rng.randint(0, max(0, background_sample.width - new_w))
        py = rng.randint(0, max(0, background_sample.height - new_h))
        pasted_box = Box(
            obj_box.class_id,
            px + obj_in_crop.x1 * scale,
            py + obj_in_crop.y1 * scale,
            px + obj_in_crop.x2 * scale,
            py + obj_in_crop.y2 * scale,
        )
        if pasted_box.x2 > background_sample.width or pasted_box.y2 > background_sample.height:
            continue
        if all(box_iou(pasted_box, box) <= max_iou for box in existing):
            bg.paste(patch, (px, py), alpha)
            return bg, existing + [pasted_box]
    return None


def save_image(img: Image.Image, path: Path, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        img.save(path, quality=quality)
    else:
        img.save(path)


def count_from_ratio(total: int, ratio: float, cap: int | None) -> int:
    count = max(0, int(round(total * ratio)))
    if cap is not None:
        count = min(count, cap)
    return count


def generate_dataset(
    source_yaml: Path,
    output: Path,
    *,
    copy_paste_ratio: float,
    background_dropout_ratio: float,
    tight_crop_ratio: float,
    robust_ratio: float,
    max_per_type: int | None,
    seed: int,
    overwrite: bool,
    copy_files: bool,
    link_originals: bool,
    preserve_eval_splits: bool,
    paste_crop_margin: float,
    paste_scale_min: float,
    paste_scale_max: float,
    paste_max_iou: float,
    background_dropout_margin: float,
    crop_margin_min: float,
    crop_margin_max: float,
    crop_jitter: float,
    min_visible_fraction: float,
    min_box_size: int,
    max_source_scan: int | None,
) -> dict:
    source_yaml = source_yaml.resolve()
    source_root = source_yaml.parent
    config = yaml.safe_load(source_yaml.read_text()) or {}
    dataset_root = Path(config.get("path") or source_root)
    if not dataset_root.is_absolute():
        dataset_root = (source_root / dataset_root).resolve()

    ensure_empty_dir(output, overwrite=overwrite)
    train_samples = collect_samples(dataset_root, config, "train", max_samples=max_source_scan)
    if not train_samples:
        raise ValueError(f"No train samples found from {source_yaml}")

    object_samples = [sample for sample in train_samples if sample.has_objects]
    background_samples = [sample for sample in train_samples if not sample.has_objects]
    if not object_samples:
        raise ValueError("No labeled train samples found for object-driven augmentations.")
    if not background_samples:
        console.print("[yellow]No empty-label train images found; copy-paste will use non-empty backgrounds too.[/yellow]")
        background_samples = train_samples

    out_train_images = output / "train" / "images"
    out_train_labels = output / "train" / "labels"
    out_train_images.mkdir(parents=True, exist_ok=True)
    out_train_labels.mkdir(parents=True, exist_ok=True)

    stats = AugmentStats()
    if link_originals:
        console.print(f"[cyan]Linking originals:[/cyan] {len(train_samples)} train images")
        stats.originals_linked = link_split(train_samples, out_train_images, out_train_labels, copy_files=copy_files)

    rng = random.Random(seed)
    total = len(train_samples)
    counts = {
        "copy_paste": count_from_ratio(total, copy_paste_ratio, max_per_type),
        "background_dropout": count_from_ratio(total, background_dropout_ratio, max_per_type),
        "tight_crop": count_from_ratio(total, tight_crop_ratio, max_per_type),
        "robust": count_from_ratio(total, robust_ratio, max_per_type),
    }

    console.print(f"[cyan]Generating augmentations:[/cyan] {counts}")
    suffix = ".jpg"

    console.print(f"[cyan]copy-paste:[/cyan] creating {counts['copy_paste']} images")
    for idx in range(counts["copy_paste"]):
        for _ in range(20):
            result = paste_object(
                rng.choice(object_samples),
                rng.choice(background_samples),
                rng,
                crop_margin=paste_crop_margin,
                scale_min=paste_scale_min,
                scale_max=paste_scale_max,
                min_box_size=min_box_size,
                max_iou=paste_max_iou,
                attempts=30,
            )
            if result is not None:
                img, boxes = result
                stem = f"odaug_copypaste_{idx:06d}"
                save_image(img, out_train_images / f"{stem}{suffix}")
                write_labels(out_train_labels / f"{stem}.txt", boxes, img.width, img.height)
                stats.copy_paste += 1
                break
        else:
            stats.skipped += 1
    console.print(f"  [green]copy-paste done:[/green] {stats.copy_paste} created, {stats.skipped} skipped so far")

    console.print(f"[cyan]background dropout:[/cyan] creating {counts['background_dropout']} images")
    for idx in range(counts["background_dropout"]):
        sample = rng.choice(object_samples)
        with Image.open(sample.image_path) as img:
            out = background_dropout_image(img, sample.boxes, rng, margin=background_dropout_margin)
        stem = f"odaug_bgdrop_{idx:06d}"
        save_image(out, out_train_images / f"{stem}{suffix}")
        write_labels(out_train_labels / f"{stem}.txt", sample.boxes, sample.width, sample.height)
        stats.background_dropout += 1
    console.print(f"  [green]background dropout done:[/green] {stats.background_dropout} created")

    console.print(f"[cyan]tight crops:[/cyan] creating {counts['tight_crop']} images")
    for idx in range(counts["tight_crop"]):
        for _ in range(20):
            result = random_tight_crop(
                rng.choice(object_samples),
                rng,
                margin_min=crop_margin_min,
                margin_max=crop_margin_max,
                jitter_fraction=crop_jitter,
                min_visible_fraction=min_visible_fraction,
                min_box_size=min_box_size,
            )
            if result is not None:
                img, boxes = result
                stem = f"odaug_tightcrop_{idx:06d}"
                save_image(img, out_train_images / f"{stem}{suffix}")
                write_labels(out_train_labels / f"{stem}.txt", boxes, img.width, img.height)
                stats.tight_crop += 1
                break
        else:
            stats.skipped += 1
    console.print(f"  [green]tight crops done:[/green] {stats.tight_crop} created, {stats.skipped} skipped so far")

    console.print(f"[cyan]robust color/blur/noise:[/cyan] creating {counts['robust']} images")
    for idx in range(counts["robust"]):
        sample = rng.choice(train_samples)
        with Image.open(sample.image_path) as img:
            out = robust_image(img, rng)
        stem = f"odaug_robust_{idx:06d}"
        save_image(out, out_train_images / f"{stem}{suffix}")
        write_labels(out_train_labels / f"{stem}.txt", sample.boxes, sample.width, sample.height)
        stats.robust += 1
    console.print(f"  [green]robust done:[/green] {stats.robust} created")

    output_config = dict(config)
    output_config["path"] = str(output.resolve())
    output_config["train"] = "train/images"

    eval_linked: dict[str, int] = {}
    if preserve_eval_splits:
        for split in ("val", "test"):
            split_samples = collect_samples(dataset_root, config, split, max_samples=None)
            if not split_samples:
                continue
            out_images = output / split / "images"
            out_labels = output / split / "labels"
            console.print(f"[cyan]Linking {split} split:[/cyan] {len(split_samples)} images")
            eval_linked[split] = link_split(split_samples, out_images, out_labels, copy_files=copy_files)
            output_config[split] = f"{split}/images"
    else:
        for split in ("val", "test"):
            if split in config:
                output_config[split] = config[split]

    (output / "data.yaml").write_text(yaml.safe_dump(output_config, sort_keys=False))

    summary = {
        "source_yaml": str(source_yaml),
        "source_root": str(dataset_root),
        "output": str(output.resolve()),
        "seed": seed,
        "copy_files": copy_files,
        "link_originals": link_originals,
        "train_samples_scanned": len(train_samples),
        "object_samples": len(object_samples),
        "background_samples": len(background_samples),
        "requested": counts,
        "created": asdict(stats),
        "eval_linked": eval_linked,
        "params": {
            "copy_paste_ratio": copy_paste_ratio,
            "background_dropout_ratio": background_dropout_ratio,
            "tight_crop_ratio": tight_crop_ratio,
            "robust_ratio": robust_ratio,
            "max_per_type": max_per_type,
            "paste_crop_margin": paste_crop_margin,
            "paste_scale_min": paste_scale_min,
            "paste_scale_max": paste_scale_max,
            "paste_max_iou": paste_max_iou,
            "background_dropout_margin": background_dropout_margin,
            "crop_margin_min": crop_margin_min,
            "crop_margin_max": crop_margin_max,
            "crop_jitter": crop_jitter,
            "min_visible_fraction": min_visible_fraction,
            "min_box_size": min_box_size,
            "max_source_scan": max_source_scan,
        },
    }
    (output / "augmentation_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


@app.command()
def main(
    source: Annotated[Path, typer.Option("-s", "--source", help="Source YOLO data.yaml or dataset directory")],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output YOLO dataset directory")],
    copy_paste_ratio: Annotated[
        float, typer.Option("--copy-paste-ratio", help="Extra copy-paste images as a fraction of train images")
    ] = 0.10,
    background_dropout_ratio: Annotated[
        float,
        typer.Option("--background-dropout-ratio", help="Extra images with background corrupted outside boxes"),
    ] = 0.10,
    tight_crop_ratio: Annotated[
        float, typer.Option("--tight-crop-ratio", help="Extra tight/object-centered crop images")
    ] = 0.10,
    robust_ratio: Annotated[
        float, typer.Option("--robust-ratio", help="Extra color/blur/noise/compression robustness images")
    ] = 0.10,
    max_per_type: Annotated[
        int | None,
        typer.Option("--max-per-type", help="Optional cap for each augmentation type; useful for experiments"),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 0,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Replace output directory if it exists")] = False,
    copy_files: Annotated[
        bool, typer.Option("--copy-files/--symlink-files", help="Copy source files instead of symlinking originals")
    ] = False,
    link_originals: Annotated[
        bool, typer.Option("--link-originals/--augmented-only", help="Include original train images in output")
    ] = True,
    preserve_eval_splits: Annotated[
        bool, typer.Option("--preserve-eval-splits/--train-only", help="Symlink val/test splits into output")
    ] = True,
    paste_crop_margin: Annotated[float, typer.Option("--paste-crop-margin", help="Context margin around pasted boxes")] = 0.03,
    paste_scale_min: Annotated[float, typer.Option("--paste-scale-min", help="Minimum copy-paste scale")] = 0.75,
    paste_scale_max: Annotated[float, typer.Option("--paste-scale-max", help="Maximum copy-paste scale")] = 1.35,
    paste_max_iou: Annotated[float, typer.Option("--paste-max-iou", help="Max IoU with existing target boxes")] = 0.05,
    background_dropout_margin: Annotated[
        float, typer.Option("--background-dropout-margin", help="Protected margin around boxes for background dropout")
    ] = 0.08,
    crop_margin_min: Annotated[float, typer.Option("--crop-margin-min", help="Minimum tight crop margin")] = 0.15,
    crop_margin_max: Annotated[float, typer.Option("--crop-margin-max", help="Maximum tight crop margin")] = 0.45,
    crop_jitter: Annotated[float, typer.Option("--crop-jitter", help="Tight crop center jitter as box-size fraction")] = 0.15,
    min_visible_fraction: Annotated[
        float, typer.Option("--min-visible-fraction", help="Minimum visible object area to keep crop labels")
    ] = 0.60,
    min_box_size: Annotated[int, typer.Option("--min-box-size", help="Minimum generated label width/height in px")] = 4,
    max_source_scan: Annotated[
        int | None,
        typer.Option("--max-source-scan", help="Optional cap on source train images for smoke tests"),
    ] = None,
):
    """Build a YOLO dataset variant that weakens background shortcuts."""

    data_yaml = find_data_yaml(source)
    console.rule("[bold blue]Object-Driven Augmentation Dataset[/bold blue]")
    console.print(f"  [cyan]Source:[/cyan] {data_yaml}")
    console.print(f"  [cyan]Output:[/cyan] {output}")
    console.print("  [cyan]Augmentations:[/cyan] copy-paste, background dropout, tight crops, color/blur/noise")

    summary = generate_dataset(
        data_yaml,
        output,
        copy_paste_ratio=copy_paste_ratio,
        background_dropout_ratio=background_dropout_ratio,
        tight_crop_ratio=tight_crop_ratio,
        robust_ratio=robust_ratio,
        max_per_type=max_per_type,
        seed=seed,
        overwrite=overwrite,
        copy_files=copy_files,
        link_originals=link_originals,
        preserve_eval_splits=preserve_eval_splits,
        paste_crop_margin=paste_crop_margin,
        paste_scale_min=paste_scale_min,
        paste_scale_max=paste_scale_max,
        paste_max_iou=paste_max_iou,
        background_dropout_margin=background_dropout_margin,
        crop_margin_min=crop_margin_min,
        crop_margin_max=crop_margin_max,
        crop_jitter=crop_jitter,
        min_visible_fraction=min_visible_fraction,
        min_box_size=min_box_size,
        max_source_scan=max_source_scan,
    )

    created = summary["created"]
    table = Table(title="Created Dataset")
    table.add_column("Item")
    table.add_column("Count", justify="right")
    for key in ("originals_linked", "copy_paste", "background_dropout", "tight_crop", "robust", "skipped"):
        table.add_row(key, str(created[key]))
    for split, count in summary["eval_linked"].items():
        table.add_row(f"{split}_linked", str(count))
    console.print(table)
    console.print(f"[green]data.yaml:[/green] {output / 'data.yaml'}")
    console.print(f"[green]summary:[/green] {output / 'augmentation_summary.json'}")


if __name__ == "__main__":
    app()
