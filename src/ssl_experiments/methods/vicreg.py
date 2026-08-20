from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from ultralytics import YOLO

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SamplingMode = Literal["global", "object", "mixed"]
ViewPairing = Literal["independent", "aligned"]


def lexical_absolute(path: Path) -> Path:
    """Normalize dots without resolving dataset symlinks to backing storage."""
    return Path(os.path.abspath(path.expanduser()))


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float


def load_yolo_boxes(path: Path) -> list[YoloBox]:
    """Load normalized YOLO detection boxes from one label file."""
    if not path.exists():
        return []

    boxes: list[YoloBox] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        values = line.split()
        if not values:
            continue
        if len(values) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO box values")
        class_value, center_x, center_y, width, height = map(float, values)
        if not class_value.is_integer():
            raise ValueError(f"{path}:{line_number}: class id must be an integer")
        if width <= 0 or height <= 0:
            raise ValueError(f"{path}:{line_number}: box dimensions must be positive")
        boxes.append(
            YoloBox(
                class_id=int(class_value),
                center_x=center_x,
                center_y=center_y,
                width=width,
                height=height,
            )
        )
    return boxes


def crop_around_box(
    image: Image.Image,
    box: YoloBox,
    *,
    context_scale: tuple[float, float],
    center_jitter: float,
    rng: random.Random | Any = random,
) -> Image.Image:
    """Sample a square crop that keeps the selected object inside the view."""
    width, height = image.size
    box_width = max(1.0, box.width * width)
    box_height = max(1.0, box.height * height)
    center_x = box.center_x * width
    center_y = box.center_y * height

    center_x += rng.uniform(-center_jitter, center_jitter) * box_width
    center_y += rng.uniform(-center_jitter, center_jitter) * box_height
    side = max(box_width, box_height) * rng.uniform(*context_scale)
    side = min(max(side, 8.0), float(width), float(height))

    left = min(max(center_x - side / 2.0, 0.0), width - side)
    top = min(max(center_y - side / 2.0, 0.0), height - side)
    right = left + side
    bottom = top + side
    return image.crop((round(left), round(top), round(right), round(bottom)))


def _appearance_transform(image_size: int) -> transforms.Compose:
    blur_kernel = max(3, (image_size // 10) // 2 * 2 + 1)
    return transforms.Compose(
        [
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.7,
                        contrast=0.7,
                        saturation=0.7,
                        hue=0.15,
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(blur_kernel, sigma=(0.1, 2.0))],
                p=0.5,
            ),
            transforms.RandomSolarize(threshold=160, p=0.1),
            transforms.ToTensor(),
        ]
    )


def _view_transform(image_size: int, *, random_crop: bool) -> transforms.Compose:
    resize = (
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.5, 1.0),
            ratio=(0.75, 1.33),
            antialias=True,
        )
        if random_crop
        else transforms.Resize((image_size, image_size), antialias=True)
    )
    return transforms.Compose(
        [
            resize,
            transforms.RandomHorizontalFlip(),
            _appearance_transform(image_size),
        ]
    )


def aligned_geometry_views(
    image: Image.Image,
    image_size: int,
    *,
    random_crop: bool,
) -> tuple[Image.Image, Image.Image]:
    """Apply one sampled geometry to both SSL views."""
    if random_crop:
        top, left, height, width = transforms.RandomResizedCrop.get_params(
            image,
            scale=(0.5, 1.0),
            ratio=(0.75, 1.33),
        )
        transformed = TF.resized_crop(
            image,
            top,
            left,
            height,
            width,
            [image_size, image_size],
            antialias=True,
        )
    else:
        transformed = TF.resize(
            image,
            [image_size, image_size],
            antialias=True,
        )
    if random.random() < 0.5:
        transformed = TF.hflip(transformed)
    return transformed, transformed.copy()


class ShapeDominantTransform:
    """Suppress color and fine texture while retaining image contours."""

    def __call__(self, image: Image.Image) -> torch.Tensor:
        grayscale = ImageOps.grayscale(image)
        grayscale = grayscale.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5))
        )
        edges = grayscale.filter(ImageFilter.FIND_EDGES)
        edges = ImageOps.autocontrast(edges, cutoff=1)
        if random.random() < 0.5:
            edges = ImageOps.invert(edges)
        return TF.to_tensor(edges.convert("RGB"))


def contour_spatial_weights(
    shape_images: torch.Tensor,
    levels: tuple[int, ...],
    *,
    region_fraction: float = 0.25,
) -> dict[int, torch.Tensor]:
    """Select high-contour spatial cells without labels or detector proposals."""
    if shape_images.ndim != 4 or shape_images.shape[1] not in {1, 3}:
        raise ValueError("shape_images must be BCHW with one or three channels")
    if not levels or any(level < 1 for level in levels):
        raise ValueError("levels must contain positive integers")
    if not 0.0 < region_fraction <= 1.0:
        raise ValueError("region_fraction must be in (0, 1]")

    grayscale = shape_images.float().mean(dim=1, keepdim=True)
    horizontal = F.pad(
        (grayscale[:, :, :, 1:] - grayscale[:, :, :, :-1]).abs(),
        (0, 1, 0, 0),
    )
    vertical = F.pad(
        (grayscale[:, :, 1:, :] - grayscale[:, :, :-1, :]).abs(),
        (0, 0, 0, 1),
    )
    contour_energy = horizontal + vertical

    weights: dict[int, torch.Tensor] = {}
    for level in levels:
        pooled = F.adaptive_avg_pool2d(contour_energy, level).flatten(1)
        cells = pooled.shape[1]
        selected_cells = max(1, round(cells * region_fraction))
        selected = pooled.topk(selected_cells, dim=1).indices
        mask = torch.zeros_like(pooled).scatter_(1, selected, 1.0)
        has_contours = pooled.amax(dim=1, keepdim=True) > 1e-8
        mask = torch.where(has_contours, mask, torch.ones_like(mask))
        weights[level] = (mask / mask.mean(dim=1, keepdim=True)).flatten()
    return weights


class ObjectCentricDataset(
    Dataset[
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]
):
    """Create RGB SSL views and, optionally, an aligned contour view."""

    def __init__(
        self,
        images: Path,
        *,
        image_list: Path | None = None,
        labels: Path | None = None,
        sampling: SamplingMode = "mixed",
        object_probability: float = 0.75,
        image_size: int = 224,
        context_scale: tuple[float, float] = (1.25, 2.5),
        center_jitter: float = 0.1,
        view_pairing: ViewPairing = "independent",
        shape_view_probability: float = 0.0,
        auxiliary_shape_view: bool = False,
        max_images: int | None = None,
        seed: int = 0,
    ) -> None:
        if sampling not in {"global", "object", "mixed"}:
            raise ValueError(f"Unsupported sampling mode: {sampling}")
        if sampling != "global" and labels is None:
            raise ValueError(f"labels are required for {sampling} sampling")
        if not 0.0 <= object_probability <= 1.0:
            raise ValueError("object_probability must be in [0, 1]")
        if image_size < 32:
            raise ValueError("image_size must be at least 32")
        if context_scale[0] < 1.0 or context_scale[1] < context_scale[0]:
            raise ValueError("context_scale must be ordered and start at 1 or greater")
        if view_pairing not in {"independent", "aligned"}:
            raise ValueError(f"Unsupported view pairing: {view_pairing}")
        if not 0.0 <= shape_view_probability <= 1.0:
            raise ValueError("shape_view_probability must be in [0, 1]")
        if shape_view_probability > 0.0 and view_pairing != "aligned":
            raise ValueError("shape-dominant views require aligned view pairing")
        if auxiliary_shape_view and shape_view_probability > 0.0:
            raise ValueError(
                "auxiliary shape views cannot be combined with shape-view replacement"
            )

        self.images_root = lexical_absolute(images)
        self.labels_root = lexical_absolute(labels) if labels is not None else None
        self.sampling = sampling
        self.object_probability = object_probability
        self.context_scale = context_scale
        self.center_jitter = center_jitter
        self.image_size = image_size
        self.view_pairing = view_pairing
        self.shape_view_probability = shape_view_probability
        self.auxiliary_shape_view = auxiliary_shape_view
        self.global_transform = _view_transform(image_size, random_crop=True)
        self.object_transform = _view_transform(image_size, random_crop=False)
        self.appearance_transform = _appearance_transform(image_size)
        self.shape_transform = ShapeDominantTransform()

        if image_list is None:
            image_paths = sorted(
                path
                for path in self.images_root.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        else:
            manifest_path = image_list.expanduser().resolve()
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Image manifest does not exist: {manifest_path}")
            image_paths = []
            for line_number, line in enumerate(
                manifest_path.read_text().splitlines(), start=1
            ):
                value = line.strip()
                if not value:
                    continue
                path = Path(value).expanduser()
                path = lexical_absolute(
                    path if path.is_absolute() else self.images_root / path
                )
                try:
                    path.relative_to(self.images_root)
                except ValueError as error:
                    raise ValueError(
                        f"{manifest_path}:{line_number}: image is outside "
                        f"{self.images_root}: {path}"
                    ) from error
                if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                    raise ValueError(
                        f"{manifest_path}:{line_number}: unsupported or missing image: {path}"
                    )
                image_paths.append(path)
            if len(set(image_paths)) != len(image_paths):
                raise ValueError(f"Image manifest contains duplicate paths: {manifest_path}")
        if max_images is not None and len(image_paths) > max_images:
            image_paths = sorted(random.Random(seed).sample(image_paths, max_images))
        if not image_paths:
            raise ValueError(f"No usable images found under {images}")
        self.image_paths = image_paths

    def _label_path(self, image_path: Path) -> Path:
        if self.labels_root is None:
            raise RuntimeError("Labels are not configured")
        relative = image_path.relative_to(self.images_root)
        return (self.labels_root / relative).with_suffix(".txt")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _use_object_view(self, boxes: list[YoloBox]) -> bool:
        if not boxes or self.sampling == "global":
            return False
        if self.sampling == "object":
            return True
        return random.random() < self.object_probability

    def __getitem__(
        self, index: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        image_path = self.image_paths[index]
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        boxes = (
            load_yolo_boxes(self._label_path(image_path))
            if self.labels_root is not None
            else []
        )
        if self._use_object_view(boxes):
            box = random.choice(boxes)
            first_source = crop_around_box(
                image,
                box,
                context_scale=self.context_scale,
                center_jitter=self.center_jitter,
            )
            second_source = (
                first_source.copy()
                if self.view_pairing == "aligned"
                else crop_around_box(
                    image,
                    box,
                    context_scale=self.context_scale,
                    center_jitter=self.center_jitter,
                )
            )
            transform = self.object_transform
            random_crop = False
        else:
            first_source = image
            second_source = image
            transform = self.global_transform
            random_crop = True

        if self.auxiliary_shape_view:
            first = transform(first_source)
            second_source, shape_source = aligned_geometry_views(
                second_source,
                self.image_size,
                random_crop=random_crop,
            )
            return (
                first,
                self.appearance_transform(second_source),
                self.shape_transform(shape_source),
            )

        if self.view_pairing == "aligned":
            first_source, second_source = aligned_geometry_views(
                first_source,
                self.image_size,
                random_crop=random_crop,
            )
            second_transform = (
                self.shape_transform
                if random.random() < self.shape_view_probability
                else self.appearance_transform
            )
            return (
                self.appearance_transform(first_source),
                second_transform(second_source),
            )

        return transform(first_source), transform(second_source)


def create_yolo_backbone(
    model_spec: str = "yolo26n.yaml",
    *,
    probe_size: int = 224,
) -> tuple[nn.Sequential, int, int]:
    """Build a YOLO backbone and return it with output width and layer count."""
    detector = YOLO(model_spec)
    backbone_layers = len(detector.model.yaml["backbone"])
    modules = list(detector.model.model.children())[:backbone_layers]
    invalid_sources = [getattr(module, "f", -1) for module in modules if getattr(module, "f", -1) != -1]
    if invalid_sources:
        raise ValueError(
            "The selected YOLO backbone contains skip inputs and cannot be wrapped "
            "as a simple sequential encoder"
        )
    backbone = nn.Sequential(*modules)

    was_training = backbone.training
    backbone.eval()
    with torch.no_grad():
        output = backbone(torch.zeros(1, 3, probe_size, probe_size))
    backbone.train(was_training)
    if not isinstance(output, torch.Tensor) or output.ndim != 4:
        raise ValueError("Expected the YOLO backbone to return one BCHW tensor")
    return backbone, int(output.shape[1]), backbone_layers


class YoloGraphTrunk(nn.Sequential):
    """Run a YOLO backbone and neck while honoring graph skip inputs."""

    def __init__(self, modules: list[nn.Module], output_layers: tuple[int, ...]) -> None:
        super().__init__(*modules)
        self.output_layers = output_layers

    def forward(self, images: torch.Tensor) -> dict[int, torch.Tensor]:
        current: torch.Tensor | list[torch.Tensor] = images
        outputs: list[torch.Tensor] = []
        selected: dict[int, torch.Tensor] = {}
        for module in self:
            sources = getattr(module, "f", -1)
            if sources != -1:
                current = (
                    outputs[sources]
                    if isinstance(sources, int)
                    else [
                        current if source == -1 else outputs[source]
                        for source in sources
                    ]
                )
            current = module(current)
            if not isinstance(current, torch.Tensor):
                raise TypeError("YOLO trunk modules must return tensors")
            outputs.append(current)
            layer = int(getattr(module, "i", len(outputs) - 1))
            if layer in self.output_layers:
                selected[layer] = current
        return selected


def create_yolo_trunk(
    model_spec: str = "yolo26n.yaml",
    *,
    probe_size: int = 224,
) -> tuple[YoloGraphTrunk, dict[int, int], int, tuple[int, ...]]:
    """Build the graph-routed YOLO trunk and expose Detect input scales."""
    detector = YOLO(model_spec)
    modules = list(detector.model.model.children())
    if len(modules) < 2:
        raise ValueError("Expected a YOLO model with a Detect head")
    detect_sources = getattr(modules[-1], "f", None)
    if not isinstance(detect_sources, list) or not detect_sources:
        raise ValueError("Expected the final YOLO head to consume multiple feature maps")
    output_layers = tuple(int(source) for source in detect_sources)
    trunk_layers = len(modules) - 1
    if any(layer < 0 or layer >= trunk_layers for layer in output_layers):
        raise ValueError("Detect input layers fall outside the YOLO trunk")
    trunk = YoloGraphTrunk(modules[:-1], output_layers)

    was_training = trunk.training
    trunk.eval()
    with torch.no_grad():
        outputs = trunk(torch.zeros(1, 3, probe_size, probe_size))
    trunk.train(was_training)
    if tuple(outputs) != output_layers:
        raise ValueError("YOLO trunk did not return every Detect input feature")
    invalid = [layer for layer, output in outputs.items() if output.ndim != 4]
    if invalid:
        raise ValueError(f"Expected BCHW trunk features at layers: {invalid}")
    feature_dims = {layer: int(output.shape[1]) for layer, output in outputs.items()}
    return trunk, feature_dims, trunk_layers, output_layers


class VICRegModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        *,
        hidden_dim: int = 1024,
        projection_dim: int = 1024,
        spatial_levels: tuple[int, ...] = (1,),
    ) -> None:
        super().__init__()
        levels = tuple(sorted(set(spatial_levels)))
        if not levels or levels[0] < 1 or 1 not in levels:
            raise ValueError("spatial_levels must contain positive level 1")
        self.backbone = backbone
        self.spatial_levels = levels

        def make_projector() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(feature_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, projection_dim),
            )

        self.projector = make_projector()
        self.spatial_projectors = nn.ModuleDict(
            {str(level): make_projector() for level in levels if level != 1}
        )

    @staticmethod
    def _spatial_vectors(features: torch.Tensor, level: int) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(features, level)
        return pooled.permute(0, 2, 3, 1).reshape(-1, pooled.shape[1])

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if not isinstance(features, torch.Tensor) or features.ndim != 4:
            raise ValueError("Expected the backbone to return a BCHW tensor")
        return features

    def spatial_features(
        self, features: torch.Tensor, levels: tuple[int, ...]
    ) -> dict[int, torch.Tensor]:
        return {level: self._spatial_vectors(features, level) for level in levels}

    def project_features(
        self, features: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        projections: dict[int, torch.Tensor] = {}
        for level in self.spatial_levels:
            vectors = self._spatial_vectors(features, level)
            projector = (
                self.projector
                if level == 1
                else self.spatial_projectors[str(level)]
            )
            projections[level] = projector(vectors)
        return projections

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projector(self._spatial_vectors(self.encode(images), 1))

    def forward_spatial(self, images: torch.Tensor) -> dict[int, torch.Tensor]:
        """Project corresponding spatial cells at each configured level."""
        return self.project_features(self.encode(images))


class MultiOutputVICRegModel(nn.Module):
    """Apply separate global VICReg projectors to multiscale trunk outputs."""

    def __init__(
        self,
        backbone: YoloGraphTrunk,
        feature_dims: dict[int, int],
        *,
        hidden_dim: int = 1024,
        projection_dim: int = 1024,
    ) -> None:
        super().__init__()
        if not feature_dims:
            raise ValueError("feature_dims must not be empty")
        self.backbone = backbone
        self.output_layers = tuple(feature_dims)
        self.projector = nn.ModuleDict(
            {
                str(layer): nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim, bias=False),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, projection_dim),
                )
                for layer, feature_dim in feature_dims.items()
            }
        )
        self.spatial_projectors = nn.ModuleDict()

    def forward_spatial(self, images: torch.Tensor) -> dict[int, torch.Tensor]:
        features = self.backbone(images)
        return {
            layer: self.projector[str(layer)](
                F.adaptive_avg_pool2d(features[layer], 1).flatten(1)
            )
            for layer in self.output_layers
        }

    def forward(self, images: torch.Tensor) -> dict[int, torch.Tensor]:
        return self.forward_spatial(images)


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    rows, columns = matrix.shape
    if rows != columns:
        raise ValueError("Expected a square covariance matrix")
    return matrix.flatten()[:-1].view(rows - 1, rows + 1)[:, 1:].flatten()


class VICRegLoss(nn.Module):
    def __init__(
        self,
        *,
        invariance_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
        variance_target: float = 1.0,
        epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        self.invariance_weight = invariance_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.variance_target = variance_target
        self.epsilon = epsilon

    def forward(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if first.shape != second.shape or first.ndim != 2:
            raise ValueError("VICReg projections must be matching [batch, features] tensors")
        if first.shape[0] < 2:
            raise ValueError("VICReg requires at least two samples per batch")

        invariance = F.mse_loss(first, second)
        first_centered = first - first.mean(dim=0)
        second_centered = second - second.mean(dim=0)
        first_std = torch.sqrt(first_centered.var(dim=0, unbiased=True) + self.epsilon)
        second_std = torch.sqrt(second_centered.var(dim=0, unbiased=True) + self.epsilon)
        variance = 0.5 * (
            F.relu(self.variance_target - first_std).mean()
            + F.relu(self.variance_target - second_std).mean()
        )

        denominator = first.shape[0] - 1
        first_covariance = first_centered.T @ first_centered / denominator
        second_covariance = second_centered.T @ second_centered / denominator
        feature_dim = first.shape[1]
        covariance = (
            _off_diagonal(first_covariance).pow(2).sum()
            + _off_diagonal(second_covariance).pow(2).sum()
        ) / feature_dim
        total = (
            self.invariance_weight * invariance
            + self.variance_weight * variance
            + self.covariance_weight * covariance
        )
        return {
            "loss": total,
            "invariance": invariance,
            "variance": variance,
            "covariance": covariance,
            "embedding_std": 0.5 * (first_std.mean() + second_std.mean()),
        }


class SpatialVICRegLoss(nn.Module):
    """Average VICReg objectives over equally weighted spatial resolutions."""

    metric_names = (
        "loss",
        "invariance",
        "variance",
        "covariance",
        "embedding_std",
    )

    def __init__(self, criterion: VICRegLoss | None = None) -> None:
        super().__init__()
        self.criterion = criterion or VICRegLoss()

    def forward(
        self,
        first: dict[int, torch.Tensor],
        second: dict[int, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if first.keys() != second.keys() or not first:
            raise ValueError("Spatial VICReg projections must have matching levels")
        per_level = {
            level: self.criterion(first[level], second[level])
            for level in first
        }
        losses = {
            name: torch.stack(
                [metrics[name] for metrics in per_level.values()]
            ).mean()
            for name in self.metric_names
        }
        losses.update(
            {
                f"level_{level}_loss": metrics["loss"]
                for level, metrics in per_level.items()
            }
        )
        return losses


class AlignedFeatureConsistencyLoss(nn.Module):
    """Match corresponding pooled backbone cells across aligned modalities."""

    def forward(
        self,
        appearance: dict[int, torch.Tensor],
        shape: dict[int, torch.Tensor],
        weights: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if appearance.keys() != shape.keys() or not appearance:
            raise ValueError("Aligned features must have matching levels")
        if weights is not None and weights.keys() != appearance.keys():
            raise ValueError("Spatial weights must match aligned feature levels")
        losses = []
        for level in appearance:
            if appearance[level].shape != shape[level].shape:
                raise ValueError(f"Feature shapes differ at level {level}")
            distance = 1.0 - F.cosine_similarity(
                appearance[level].float(),
                shape[level].float(),
                dim=1,
            )
            if weights is None:
                losses.append(distance.mean())
                continue
            level_weights = weights[level].to(distance, non_blocking=True)
            if level_weights.ndim != 1 or level_weights.shape != distance.shape:
                raise ValueError(f"Spatial weights differ at level {level}")
            losses.append((distance * level_weights).sum() / level_weights.sum())
        return torch.stack(losses).mean()


def load_ssl_backbone(
    detector: YOLO,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load an SSL backbone into a YOLO detector with architecture validation."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != "vicreg":
        raise ValueError("Checkpoint is not an object-detector VICReg checkpoint")
    if "backbone_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain backbone_state_dict")

    backbone_layers = int(checkpoint["backbone_layers"])
    target_modules = list(detector.model.model.children())[:backbone_layers]
    if len(target_modules) != backbone_layers:
        raise ValueError("Detector has fewer backbone layers than the SSL checkpoint")
    target = nn.Sequential(*target_modules)
    source_state = checkpoint["backbone_state_dict"]
    target_state = target.state_dict()
    missing = sorted(set(target_state) - set(source_state))
    unexpected = sorted(set(source_state) - set(target_state))
    shape_mismatches = sorted(
        key
        for key in set(source_state) & set(target_state)
        if source_state[key].shape != target_state[key].shape
    )
    if strict and (missing or unexpected or shape_mismatches):
        raise ValueError(
            "SSL backbone does not match the detector: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"shape_mismatches={len(shape_mismatches)}"
        )

    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state and value.shape == target_state[key].shape
    }
    target.load_state_dict(compatible, strict=False)
    return {
        "loaded_tensors": len(compatible),
        "backbone_layers": backbone_layers,
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatches": shape_mismatches,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "sampling": checkpoint.get("config", {}).get("sampling"),
    }
