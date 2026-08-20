#!/usr/bin/env python3
"""
Create a stylized copy of a YOLO-format object detection dataset.

The script preserves the dataset structure and label files, while replacing
each image with a stylized version produced by OpenCV image filters.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
import yaml
from PIL import Image, ImageFilter, ImageOps
from rich.console import Console

from ssl_experiments.progress import make_progress

console = Console()
app = typer.Typer()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class SplitDirs:
    name: str
    images: Path
    labels: Path
    image_rel: str
    label_rel: str


@dataclass(frozen=True)
class StylizeSettings:
    mode: str
    sigma_s: float
    sigma_r: float
    color_levels: int
    detail_weight: float
    edge_strength: float
    tint_strength: int
    edge_boost: float
    seed: int


def _find_data_yaml(source: Path) -> Path:
    if source.is_file():
        return source
    for candidate in ("data.yaml", "dataset.yaml"):
        yaml_path = source / candidate
        if yaml_path.exists():
            return yaml_path
    raise FileNotFoundError(f"Could not find data.yaml in {source}")


def _resolve_split_dirs(dataset_root: Path, config: dict) -> list[SplitDirs]:
    splits: list[SplitDirs] = []
    for key in ("train", "val", "valid", "test"):
        rel = config.get(key)
        if not rel:
            continue

        image_rel = str(rel)
        image_dir = dataset_root / image_rel
        if not image_dir.exists():
            continue

        if "/images" in image_rel:
            label_rel = image_rel.replace("/images", "/labels", 1)
        elif image_dir.name == "images":
            label_rel = str(Path(image_rel).parent / "labels")
        else:
            label_rel = str(Path(image_rel).parent / "labels" / image_dir.name)

        label_dir = dataset_root / label_rel
        canonical = "val" if key == "valid" else key
        splits.append(
            SplitDirs(
                name=canonical,
                images=image_dir,
                labels=label_dir,
                image_rel=image_rel,
                label_rel=label_rel,
            )
        )

    if splits:
        return splits

    # Fallback for datasets that have split/images and split/labels without yaml entries.
    for split_name in ("train", "val", "valid", "test"):
        img_dir = dataset_root / split_name / "images"
        lbl_dir = dataset_root / split_name / "labels"
        if img_dir.exists():
            canonical = "val" if split_name == "valid" else split_name
            splits.append(
                SplitDirs(
                    name=canonical,
                    images=img_dir,
                    labels=lbl_dir,
                    image_rel=f"{split_name}/images",
                    label_rel=f"{split_name}/labels",
                )
            )

    if splits:
        return splits

    # Flat dataset layout.
    img_dir = dataset_root / "images"
    lbl_dir = dataset_root / "labels"
    if img_dir.exists():
        return [SplitDirs(name="all", images=img_dir, labels=lbl_dir, image_rel="images", label_rel="labels")]

    raise FileNotFoundError(f"Could not resolve dataset splits under {dataset_root}")


def _copy_label(src: Path | None, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src and src.exists():
        shutil.copy2(src, dst)
    else:
        dst.write_text("")


def _seed_for_path(base_seed: int, image_path: Path) -> int:
    path_seed = zlib.crc32(str(image_path.resolve()).encode("utf-8"))
    return (base_seed ^ path_seed) & 0xFFFFFFFF


def _quantize_colors(img: np.ndarray, levels: int) -> np.ndarray:
    levels = max(2, min(levels, 64))
    step = max(1, 256 // levels)
    quantized = (img // step) * step + step // 2
    return np.clip(quantized, 0, 255).astype(np.uint8)


def _random_tint(rng: random.Random, tint_strength: int) -> np.ndarray:
    tint_strength = max(0, int(tint_strength))
    tint = np.array(
        [
            rng.randint(-tint_strength, tint_strength),
            rng.randint(-tint_strength, tint_strength),
            rng.randint(-tint_strength, tint_strength),
        ],
        dtype=np.int16,
    )
    return tint.reshape(1, 1, 3)


def stylize_image(
    img: Image.Image,
    *,
    mode: str,
    sigma_s: float,
    sigma_r: float,
    color_levels: int,
    detail_weight: float,
    edge_strength: float,
    tint_strength: int,
    edge_boost: float,
    rng: random.Random,
) -> Image.Image:
    if mode == "stylization":
        styled = img.filter(ImageFilter.ModeFilter(size=5))
        if color_levels > 0:
            styled = ImageOps.posterize(styled, bits=max(2, min(7, int(np.log2(color_levels)) + 1)))
        if detail_weight > 0:
            detail = img.filter(ImageFilter.DETAIL)
            styled = Image.blend(styled, detail, alpha=min(1.0, detail_weight))
        return styled

    if mode == "cartoon":
        smooth = img.filter(ImageFilter.MedianFilter(size=5))
        smooth = ImageOps.posterize(smooth, bits=max(2, min(6, int(np.log2(color_levels or 8)) + 1)))
        edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
        edges = ImageOps.autocontrast(edges)
        edges = ImageOps.invert(edges)
        edge_arr = np.asarray(edges, dtype=np.float32) / 255.0
        edge_arr = np.clip(edge_arr * (1.0 - 0.35 * edge_strength) + (0.35 * edge_strength), 0.0, 1.0)
        smooth_arr = np.asarray(smooth, dtype=np.float32)
        styled_arr = np.clip(smooth_arr * edge_arr[..., None], 0, 255).astype(np.uint8)
        styled = Image.fromarray(styled_arr, mode="RGB")
        return styled

    if mode == "texture_suppressed":
        smooth = img.filter(ImageFilter.GaussianBlur(radius=max(1.0, sigma_r * 8.0)))
        smooth = smooth.filter(ImageFilter.ModeFilter(size=7))
        smooth = ImageOps.posterize(smooth, bits=max(2, min(5, int(np.log2(color_levels or 6)) + 1)))
        smooth_arr = np.asarray(smooth, dtype=np.int16)
        tinted = np.clip(smooth_arr + _random_tint(rng, tint_strength), 0, 255).astype(np.uint8)
        styled = Image.fromarray(tinted, mode="RGB")
        if detail_weight > 0:
            base_edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
            base_edges = ImageOps.autocontrast(base_edges)
            edge_arr = np.asarray(base_edges, dtype=np.float32) / 255.0
            styled_arr = np.asarray(styled, dtype=np.float32)
            styled_arr = np.clip(styled_arr + (edge_arr[..., None] * edge_boost * detail_weight), 0, 255).astype(
                np.uint8
            )
            styled = Image.fromarray(styled_arr, mode="RGB")
        if edge_strength > 0:
            styled = styled.filter(ImageFilter.UnsharpMask(radius=2, percent=int(80 * edge_strength), threshold=3))
        return styled

    raise ValueError(f"Unsupported mode: {mode}")


def _stylize_one(
    image_path: Path,
    label_path: Path,
    out_image: Path,
    out_label: Path,
    settings: StylizeSettings,
) -> bool:
    try:
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
    except Exception:
        _copy_label(label_path, out_label)
        return False

    rng = random.Random(_seed_for_path(settings.seed, image_path))
    styled = stylize_image(
        image,
        mode=settings.mode,
        sigma_s=settings.sigma_s,
        sigma_r=settings.sigma_r,
        color_levels=settings.color_levels,
        detail_weight=settings.detail_weight,
        edge_strength=settings.edge_strength,
        tint_strength=settings.tint_strength,
        edge_boost=settings.edge_boost,
        rng=rng,
    )
    styled.save(out_image)
    _copy_label(label_path, out_label)
    return True


def _save_yaml(output_dir: Path, config: dict, splits: list[SplitDirs]) -> None:
    new_config = dict(config)
    new_config["path"] = str(output_dir.absolute())

    split_map = {split.name: split for split in splits}
    if "train" in split_map:
        new_config["train"] = split_map["train"].image_rel
    if "val" in split_map:
        new_config["val"] = split_map["val"].image_rel
    if "test" in split_map:
        new_config["test"] = split_map["test"].image_rel
    if "all" in split_map:
        new_config["train"] = split_map["all"].image_rel
        new_config["val"] = split_map["all"].image_rel

    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)


@app.command()
def main(
    source: Annotated[Path, typer.Option("-s", "--source", help="Dataset directory or data.yaml path")],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output directory for the stylized dataset")],
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="Stylization mode: stylization, cartoon, texture_suppressed",
            case_sensitive=False,
        ),
    ] = "texture_suppressed",
    sigma_s: Annotated[float, typer.Option("--sigma-s", help="Spatial smoothness for OpenCV stylization")] = 80.0,
    sigma_r: Annotated[float, typer.Option("--sigma-r", help="Range smoothness for OpenCV stylization")] = 0.45,
    color_levels: Annotated[int, typer.Option("--color-levels", help="Color quantization levels")] = 7,
    detail_weight: Annotated[float, typer.Option("--detail-weight", help="Detail enhancement blend for stylization mode")] = 0.15,
    edge_strength: Annotated[float, typer.Option("--edge-strength", help="Edge darkening strength for cartoon mode")] = 0.6,
    tint_strength: Annotated[int, typer.Option("--tint-strength", help="Per-channel random tint range for texture_suppressed mode")] = 18,
    edge_boost: Annotated[float, typer.Option("--edge-boost", help="Edge intensity boost added back in texture_suppressed mode")] = 24.0,
    seed: Annotated[int, typer.Option("--seed", help="Random seed for repeatable tint variation")] = 0,
    workers: Annotated[int, typer.Option("--workers", help="Parallel worker processes for image stylization")] = max(1, (os.cpu_count() or 1) - 1),
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Replace the output directory if it exists")] = False,
):
    """Create a stylized YOLO dataset copy with unchanged annotations."""
    mode = mode.lower()
    if mode not in {"stylization", "cartoon", "texture_suppressed"}:
        raise typer.BadParameter("mode must be one of: stylization, cartoon, texture_suppressed")

    yaml_path = _find_data_yaml(source)
    dataset_root = yaml_path.parent
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    splits = _resolve_split_dirs(dataset_root, config)

    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    settings = StylizeSettings(
        mode=mode,
        sigma_s=sigma_s,
        sigma_r=sigma_r,
        color_levels=color_levels,
        detail_weight=detail_weight,
        edge_strength=edge_strength,
        tint_strength=tint_strength,
        edge_boost=edge_boost,
        seed=seed,
    )

    console.rule("[bold blue]Stylize YOLO Dataset[/bold blue]")
    console.print(f"  [cyan]Source:[/cyan] {dataset_root}")
    console.print(f"  [cyan]Output:[/cyan] {output}")
    console.print(f"  [cyan]Mode:[/cyan]   {mode}")
    console.print(f"  [cyan]Workers:[/cyan] {workers}")
    console.print(f"  [cyan]Splits:[/cyan] {', '.join(split.name for split in splits)}")

    total_images = 0
    unreadable_images = 0

    for split in splits:
        split_output_images = output / split.image_rel
        split_output_labels = output / split.label_rel
        split_output_images.mkdir(parents=True, exist_ok=True)
        split_output_labels.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            path for path in split.images.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        total_images += len(image_files)

        jobs = [
            (
                image_path,
                split.labels / f"{image_path.stem}.txt",
                split_output_images / image_path.name,
                split_output_labels / f"{image_path.stem}.txt",
                settings,
            )
            for image_path in image_files
        ]

        with make_progress() as progress:
            task = progress.add_task(f"Stylizing {split.name}", total=len(jobs))
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_stylize_one, *job) for job in jobs]
                for future in as_completed(futures):
                    if not future.result():
                        unreadable_images += 1
                    progress.advance(task)

    _save_yaml(output, config, splits)

    metadata = {
        "source": str(dataset_root),
        "source_yaml": str(yaml_path),
        "output": str(output),
        "total_images": total_images,
        "unreadable_images": unreadable_images,
        "params": {
            **asdict(settings),
            "workers": workers,
        },
        "splits": [asdict(split) for split in splits],
    }
    for split in metadata["splits"]:
        split["images"] = str(split["images"])
        split["labels"] = str(split["labels"])

    (output / "stylization_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    console.print()
    console.rule("[bold green]Complete![/bold green]")
    console.print(f"  [green]Images stylized:[/green] {total_images - unreadable_images}")
    console.print(f"  [yellow]Unreadable images:[/yellow] {unreadable_images}")
    console.print(f"  [green]Dataset yaml:[/green] {output / 'data.yaml'}")
    console.print(f"  [green]Metadata:[/green] {output / 'stylization_metadata.json'}")


if __name__ == "__main__":
    app()
