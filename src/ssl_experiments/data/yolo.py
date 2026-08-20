from collections import defaultdict
from pathlib import Path

import yaml
from rich.console import Console

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

console = Console()


def parse_yolo_yaml(yaml_path: Path) -> dict:
    """Parse a YOLO data.yaml file."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def load_yolo_labels(label_path: Path, img_w: int, img_h: int):
    """Load YOLO-format label file, returning pixel-coordinate boxes and class ids."""
    boxes, classes = [], []
    if not label_path.exists():
        return boxes, classes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, x_c, y_c, w, h = map(float, parts[:5])
        x1 = (x_c - w / 2) * img_w
        y1 = (y_c - h / 2) * img_h
        x2 = (x_c + w / 2) * img_w
        y2 = (y_c + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
        classes.append(int(cls))
    return boxes, classes


def load_mot_gt(gt_txt: Path) -> dict:
    """Parse MOT16 gt.txt → dict {frame_int: list of [x1,y1,x2,y2]}."""
    gt = defaultdict(list)
    for line in gt_txt.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        frame = int(parts[0])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        gt[frame].append([x, y, x + w, y + h])
    return gt


def find_label_path(image_path: Path, labels_dir: Path) -> Path | None:
    """Find the corresponding label file for an image."""
    label_path = labels_dir / f"{image_path.stem}.txt"
    return label_path if label_path.exists() else None


def collect_all_images(source_dir: Path) -> list[tuple[Path, Path | None]]:
    """Collect all image-label pairs from a dataset, merging all splits."""
    source_dir = Path(source_dir)
    samples = []

    images_root = source_dir / "images"
    labels_root = source_dir / "labels"

    if not images_root.exists():
        images_root = source_dir
        labels_root = source_dir

    split_dirs = ["train", "val", "test", "valid"]
    splits_found = [s for s in split_dirs if (images_root / s).exists()]

    if not splits_found:
        splits_found = [s for s in split_dirs if (source_dir / s / "images").exists()]
        if splits_found:
            images_root = source_dir
            labels_root = source_dir

    if splits_found:
        console.print(f"  [cyan]Found splits:[/cyan] {', '.join(splits_found)}")
        console.print(f"  [cyan]Images root:[/cyan] {images_root}")

        for split in splits_found:
            img_dir = images_root / split

            if (img_dir / "images").is_dir():
                actual_img_dir = img_dir / "images"
                lbl_dir = img_dir / "labels" if (img_dir / "labels").exists() else actual_img_dir
            else:
                actual_img_dir = img_dir
                if (source_dir / "labels" / split).exists():
                    lbl_dir = source_dir / "labels" / split
                elif (source_dir / "labels").exists():
                    lbl_dir = source_dir / "labels"
                else:
                    lbl_dir = actual_img_dir

            if not actual_img_dir.exists():
                console.print(f"    [yellow]{split}: directory not found at {actual_img_dir}[/yellow]")
                continue

            count_before = len(samples)
            for item in sorted(actual_img_dir.iterdir()):
                if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
                    lbl_path = lbl_dir / f"{item.stem}.txt"
                    samples.append((item, lbl_path if lbl_path.exists() else None))

            console.print(f"    [cyan]{split}:[/cyan] {len(samples) - count_before} images from {actual_img_dir}")
    else:
        if images_root.exists():
            for img_path in sorted(images_root.iterdir()):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    lbl_path = labels_root / f"{img_path.stem}.txt"
                    samples.append((img_path, lbl_path if lbl_path.exists() else None))
            console.print(f"  [cyan]Found flat structure in[/cyan] {images_root}")
        else:
            raise FileNotFoundError(
                f"Could not find images in {source_dir}. Expected 'images/' directory or split directories (train/val/test)."
            )

    return samples
