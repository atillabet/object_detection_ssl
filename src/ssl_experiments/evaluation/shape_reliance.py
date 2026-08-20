"""Counterfactual transforms and scores for measuring detector shape reliance."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class Condition:
    """One clean or counterfactual evaluation condition."""

    key: str
    category: str
    transform: str
    severity: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


@dataclass(frozen=True)
class TransformStats:
    """Counts describing how many labeled objects a transform changed."""

    transformed_objects: int = 0
    skipped_objects: int = 0


def _severity_slug(severity: float) -> str:
    value = f"{severity:.3f}".rstrip("0").rstrip(".")
    return value.replace(".", "p")


def build_conditions(
    severities: Sequence[float],
    *,
    include_edge: bool = True,
    hard_mode: bool = False,
) -> list[Condition]:
    """Create the fixed set of conditions used by the benchmark."""
    unique_severities = sorted(set(float(value) for value in severities))
    if not unique_severities:
        raise ValueError("At least one severity is required")
    if any(value <= 0.0 or value > 1.0 for value in unique_severities):
        raise ValueError("Severities must be in the interval (0, 1]")

    conditions = [Condition("clean", "clean", "none", 0.0)]
    for severity in unique_severities:
        slug = _severity_slug(severity)
        conditions.extend(
            [
                Condition(
                    f"appearance_hue_s{slug}",
                    "appearance",
                    "hue_shift",
                    severity,
                ),
                Condition(
                    f"appearance_gray_s{slug}",
                    "appearance",
                    "grayscale_blend",
                    severity,
                ),
                Condition(
                    f"shape_shuffle_s{slug}",
                    "shape",
                    "patch_shuffle",
                    severity,
                ),
            ]
        )
        if hard_mode:
            conditions.extend(
                [
                    Condition(
                        f"appearance_posterize_s{slug}",
                        "appearance",
                        "posterize",
                        severity,
                    ),
                    Condition(
                        f"shape_phase_s{slug}",
                        "shape",
                        "phase_scramble",
                        severity,
                    ),
                    Condition(
                        f"context_noise_s{slug}",
                        "context",
                        "background_noise",
                        severity,
                    ),
                    Condition(
                        f"stress_dropout_s{slug}",
                        "stress",
                        "object_dropout",
                        severity,
                    ),
                ]
            )
    if include_edge:
        conditions.append(Condition("edge_only", "diagnostic", "edge_only", 1.0))
    return conditions


def build_v2_conditions(
    severities: Sequence[float],
    *,
    include_edge: bool = True,
) -> list[Condition]:
    """Create cue-isolating color, texture, shape, and context conditions."""
    unique_severities = sorted(set(float(value) for value in severities))
    if not unique_severities:
        raise ValueError("At least one severity is required")
    if any(value <= 0.0 or value > 1.0 for value in unique_severities):
        raise ValueError("Severities must be in the interval (0, 1]")

    conditions = [Condition("clean", "clean", "none", 0.0)]
    for severity in unique_severities:
        slug = _severity_slug(severity)
        conditions.extend(
            [
                Condition(
                    f"color_gray_s{slug}",
                    "color",
                    "object_grayscale",
                    severity,
                ),
                Condition(
                    f"color_channels_s{slug}",
                    "color",
                    "object_channel_shuffle",
                    severity,
                ),
                Condition(
                    f"texture_bilateral_s{slug}",
                    "texture",
                    "object_bilateral",
                    severity,
                ),
                Condition(
                    f"texture_gaussian_s{slug}",
                    "texture",
                    "object_gaussian",
                    severity,
                ),
                Condition(
                    f"shape_shuffle_s{slug}",
                    "shape",
                    "object_patch_shuffle",
                    severity,
                ),
                Condition(
                    f"shape_rotate_s{slug}",
                    "shape",
                    "object_patch_rotate",
                    severity,
                ),
            ]
        )
    conditions.append(
        Condition("context_swap", "context", "background_swap", 1.0)
    )
    if include_edge:
        conditions.append(Condition("edge_only", "diagnostic", "edge_only", 1.0))
    return conditions


def stable_rng(seed: int, image_id: str, condition_key: str) -> np.random.Generator:
    """Return a process-independent deterministic generator for one image."""
    payload = f"{seed}\0{image_id}\0{condition_key}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    derived_seed = int.from_bytes(digest, byteorder="little", signed=False)
    return np.random.default_rng(derived_seed)


def shift_hue(image_bgr: np.ndarray, severity: float) -> np.ndarray:
    """Shift hue without moving pixels or changing luminance geometry."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    hue_shift = max(1, int(round(60 * severity)))
    hsv[:, :, 0] = ((hue + hue_shift) % 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def blend_grayscale(image_bgr: np.ndarray, severity: float) -> np.ndarray:
    """Remove color progressively while retaining spatial structure."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(
        image_bgr,
        1.0 - severity,
        gray_bgr,
        severity,
        0.0,
    )


def posterize(image_bgr: np.ndarray, severity: float) -> np.ndarray:
    """Reduce color precision while leaving pixel positions unchanged."""
    bits = max(2, min(8, int(round(8.0 - 7.0 * severity))))
    shift = 8 - bits
    if shift == 0:
        return image_bgr.copy()
    step = 1 << shift
    quantized = (image_bgr >> shift) << shift
    return np.clip(quantized.astype(np.int16) + step // 2, 0, 255).astype(
        np.uint8
    )


def blend_channel_shuffle(image_bgr: np.ndarray, severity: float) -> np.ndarray:
    """Disrupt chromatic channel relationships without moving pixels."""
    shuffled = image_bgr[:, :, [1, 2, 0]]
    return cv2.addWeighted(image_bgr, 1.0 - severity, shuffled, severity, 0.0)


def suppress_texture_bilateral(
    image_bgr: np.ndarray,
    severity: float,
) -> np.ndarray:
    """Suppress local texture while preferentially retaining strong edges."""
    diameter = 5 + 2 * int(round(3.0 * severity))
    sigma_color = 30.0 + 140.0 * severity
    sigma_space = 3.0 + 9.0 * severity
    filtered = cv2.bilateralFilter(
        image_bgr,
        diameter,
        sigma_color,
        sigma_space,
    )
    return cv2.addWeighted(image_bgr, 1.0 - severity, filtered, severity, 0.0)


def suppress_texture_gaussian(
    image_bgr: np.ndarray,
    severity: float,
) -> np.ndarray:
    """Suppress high-frequency texture with a severity-scaled Gaussian blur."""
    sigma = 0.5 + 3.5 * severity
    kernel = 2 * int(np.ceil(3.0 * sigma)) + 1
    filtered = cv2.GaussianBlur(image_bgr, (kernel, kernel), sigma)
    return cv2.addWeighted(image_bgr, 1.0 - severity, filtered, severity, 0.0)


def build_object_region_mask(
    image_shape: Sequence[int],
    boxes: Sequence[Sequence[float]],
    instance_masks: Sequence[np.ndarray] | None = None,
    *,
    box_margin: float = 0.0,
) -> np.ndarray:
    """Return the union of instance masks or an explicit box fallback."""
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    region = np.zeros((image_height, image_width), dtype=bool)
    if instance_masks is not None:
        if len(instance_masks) != len(boxes):
            raise ValueError("instance_masks must align one-to-one with boxes")
        for mask in instance_masks:
            if mask.shape[:2] != (image_height, image_width):
                raise ValueError("Every instance mask must match the image dimensions")
            region |= mask.astype(bool)
        return region

    for box in boxes:
        width = max(0.0, float(box[2]) - float(box[0]))
        height = max(0.0, float(box[3]) - float(box[1]))
        expanded = (
            float(box[0]) - box_margin * width,
            float(box[1]) - box_margin * height,
            float(box[2]) + box_margin * width,
            float(box[3]) + box_margin * height,
        )
        x1, y1, x2, y2 = _clipped_box(expanded, image_width, image_height)
        region[y1:y2, x1:x2] = True
    return region


def apply_to_object_regions(
    image_bgr: np.ndarray,
    transformed_bgr: np.ndarray,
    boxes: Sequence[Sequence[float]],
    instance_masks: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Composite a transformed view only inside labeled object regions."""
    region = build_object_region_mask(image_bgr.shape, boxes, instance_masks)
    output = image_bgr.copy()
    output[region] = transformed_bgr[region]
    return output


def erase_labeled_regions(
    image_bgr: np.ndarray,
    boxes: Sequence[Sequence[float]],
    instance_masks: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Create a real-scene background donor by inpainting labeled regions."""
    region = build_object_region_mask(
        image_bgr.shape,
        boxes,
        instance_masks,
        box_margin=0.05,
    ).astype(np.uint8)
    if not np.any(region):
        return image_bgr.copy()
    region = cv2.dilate(region, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(image_bgr, region, 3.0, cv2.INPAINT_TELEA)


def replace_background(
    image_bgr: np.ndarray,
    background_bgr: np.ndarray,
    boxes: Sequence[Sequence[float]],
    severity: float,
    instance_masks: Sequence[np.ndarray] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Replace a severity-controlled background area with real donor pixels."""
    image_height, image_width = image_bgr.shape[:2]
    if background_bgr.shape[:2] != (image_height, image_width):
        background_bgr = cv2.resize(
            background_bgr,
            (image_width, image_height),
            interpolation=cv2.INTER_LINEAR,
        )
    protected = build_object_region_mask(image_bgr.shape, boxes, instance_masks)
    if severity >= 1.0:
        alpha = np.ones((image_height, image_width), dtype=np.float32)
    else:
        rng = rng or np.random.default_rng(0)
        coarse_height = max(2, min(10, image_height // 24))
        coarse_width = max(2, min(10, image_width // 24))
        field = cv2.resize(
            rng.random((coarse_height, coarse_width)).astype(np.float32),
            (image_width, image_height),
            interpolation=cv2.INTER_CUBIC,
        )
        threshold = float(np.quantile(field, 1.0 - severity))
        alpha = (field >= threshold).astype(np.float32)
        alpha = cv2.GaussianBlur(alpha, (0, 0), 1.5)
    alpha[protected] = 0.0
    alpha = alpha[:, :, None]
    composite = (
        image_bgr.astype(np.float32) * (1.0 - alpha)
        + background_bgr.astype(np.float32) * alpha
    )
    return np.clip(composite, 0, 255).astype(np.uint8)


def edge_only(image_bgr: np.ndarray) -> np.ndarray:
    """Create a high-contrast edge diagnostic view."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blurred))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, max(lower + 1, 1.33 * median)))
    edges = cv2.Canny(blurred, lower, upper)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def _clipped_box(
    box: Sequence[float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1 = max(0, min(image_width, int(np.floor(box[0]))))
    y1 = max(0, min(image_height, int(np.floor(box[1]))))
    x2 = max(0, min(image_width, int(np.ceil(box[2]))))
    y2 = max(0, min(image_height, int(np.ceil(box[3]))))
    return x1, y1, x2, y2


def shuffle_box_patches(
    image_bgr: np.ndarray,
    boxes: Sequence[Sequence[float]],
    severity: float,
    rng: np.random.Generator,
    *,
    grid_size: int = 4,
) -> tuple[np.ndarray, TransformStats]:
    """Disrupt spatial arrangement inside boxes while preserving their pixels.

    Equal-sized tiles are cyclically permuted. The unchanged border accounts for
    box dimensions that are not divisible by the selected grid size.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    source = image_bgr
    output = image_bgr.copy()
    image_height, image_width = image_bgr.shape[:2]
    transformed = 0
    skipped = 0

    for box in boxes:
        x1, y1, x2, y2 = _clipped_box(box, image_width, image_height)
        width = x2 - x1
        height = y2 - y1
        grid = min(grid_size, width // 2, height // 2)
        if grid < 2:
            skipped += 1
            continue

        tile_width = width // grid
        tile_height = height // grid
        tiled_width = tile_width * grid
        tiled_height = tile_height * grid
        roi = source[y1 : y1 + tiled_height, x1 : x1 + tiled_width]
        tiles = [
            roi[
                row * tile_height : (row + 1) * tile_height,
                column * tile_width : (column + 1) * tile_width,
            ].copy()
            for row in range(grid)
            for column in range(grid)
        ]

        selected_count = min(
            len(tiles),
            max(2, int(round(severity * len(tiles)))),
        )
        selected = rng.choice(len(tiles), size=selected_count, replace=False)
        replacement = np.roll(selected, 1)

        shuffled_roi = roi.copy()
        for destination, source_index in zip(selected, replacement, strict=True):
            row, column = divmod(int(destination), grid)
            shuffled_roi[
                row * tile_height : (row + 1) * tile_height,
                column * tile_width : (column + 1) * tile_width,
            ] = tiles[int(source_index)]

        output[y1 : y1 + tiled_height, x1 : x1 + tiled_width] = shuffled_roi
        transformed += 1

    return output, TransformStats(transformed, skipped)


def shuffle_object_patches(
    image_bgr: np.ndarray,
    boxes: Sequence[Sequence[float]],
    severity: float,
    rng: np.random.Generator,
    *,
    grid_size: int = 4,
    instance_masks: Sequence[np.ndarray] | None = None,
) -> tuple[np.ndarray, TransformStats]:
    """Shuffle object pixels and alpha masks together when masks are available."""
    if instance_masks is None:
        return shuffle_box_patches(
            image_bgr,
            boxes,
            severity,
            rng,
            grid_size=grid_size,
        )
    if len(instance_masks) != len(boxes):
        raise ValueError("instance_masks must align one-to-one with boxes")

    source = image_bgr
    output = image_bgr.copy()
    image_height, image_width = image_bgr.shape[:2]
    transformed = 0
    skipped = 0
    for box, instance_mask in zip(boxes, instance_masks, strict=True):
        x1, y1, x2, y2 = _clipped_box(box, image_width, image_height)
        width, height = x2 - x1, y2 - y1
        grid = min(grid_size, width // 2, height // 2)
        if grid < 2:
            skipped += 1
            continue
        tile_width, tile_height = width // grid, height // grid
        tiled_width, tiled_height = tile_width * grid, tile_height * grid
        roi = source[y1 : y1 + tiled_height, x1 : x1 + tiled_width]
        mask_roi = instance_mask[
            y1 : y1 + tiled_height,
            x1 : x1 + tiled_width,
        ].astype(bool)
        if not np.any(mask_roi):
            skipped += 1
            continue

        rgb_tiles = []
        mask_tiles = []
        for row in range(grid):
            for column in range(grid):
                ys = slice(row * tile_height, (row + 1) * tile_height)
                xs = slice(column * tile_width, (column + 1) * tile_width)
                rgb_tiles.append(roi[ys, xs].copy())
                mask_tiles.append(mask_roi[ys, xs].copy())

        selected_count = min(
            len(rgb_tiles),
            max(2, int(round(severity * len(rgb_tiles)))),
        )
        selected = rng.choice(len(rgb_tiles), size=selected_count, replace=False)
        occupied = [index for index, tile in enumerate(mask_tiles) if np.any(tile)]
        if not any(int(index) in occupied for index in selected):
            selected[0] = int(rng.choice(occupied))
        replacement = np.roll(selected, 1)

        transformed_roi = roi.copy()
        transformed_mask = mask_roi.copy()
        for destination, source_index in zip(selected, replacement, strict=True):
            row, column = divmod(int(destination), grid)
            ys = slice(row * tile_height, (row + 1) * tile_height)
            xs = slice(column * tile_width, (column + 1) * tile_width)
            transformed_roi[ys, xs] = rgb_tiles[int(source_index)]
            transformed_mask[ys, xs] = mask_tiles[int(source_index)]

        inpaint_mask = (mask_roi.astype(np.uint8) * 255)
        background = cv2.inpaint(roi, inpaint_mask, 3.0, cv2.INPAINT_TELEA)
        composite = background
        composite[transformed_mask] = transformed_roi[transformed_mask]
        output[y1 : y1 + tiled_height, x1 : x1 + tiled_width] = composite
        transformed += 1
    return output, TransformStats(transformed, skipped)


def rotate_object_patches(
    image_bgr: np.ndarray,
    boxes: Sequence[Sequence[float]],
    severity: float,
    rng: np.random.Generator,
    *,
    grid_size: int = 4,
    instance_masks: Sequence[np.ndarray] | None = None,
) -> tuple[np.ndarray, TransformStats]:
    """Rotate local object patches while retaining their pixel distribution."""
    if instance_masks is not None and len(instance_masks) != len(boxes):
        raise ValueError("instance_masks must align one-to-one with boxes")

    source = image_bgr
    output = image_bgr.copy()
    image_height, image_width = image_bgr.shape[:2]
    transformed = 0
    skipped = 0
    for box_index, box in enumerate(boxes):
        x1, y1, x2, y2 = _clipped_box(box, image_width, image_height)
        width, height = x2 - x1, y2 - y1
        grid = min(grid_size, width // 2, height // 2)
        if grid < 2:
            skipped += 1
            continue
        boundaries_x = np.linspace(x1, x2, grid + 1, dtype=int)
        boundaries_y = np.linspace(y1, y2, grid + 1, dtype=int)
        mask = None if instance_masks is None else instance_masks[box_index].astype(bool)
        candidates = []
        for cell_index in range(grid * grid):
            row, column = divmod(cell_index, grid)
            if mask is None or np.any(
                mask[
                    boundaries_y[row] : boundaries_y[row + 1],
                    boundaries_x[column] : boundaries_x[column + 1],
                ]
            ):
                candidates.append(cell_index)
        if not candidates:
            skipped += 1
            continue
        selected_count = min(
            len(candidates),
            max(1, int(round(severity * len(candidates)))),
        )
        selected = rng.choice(candidates, size=selected_count, replace=False)

        roi = source[y1:y2, x1:x2]
        if mask is None:
            composite = roi.copy()
            transformed_mask = None
        else:
            mask_roi = mask[y1:y2, x1:x2]
            background = cv2.inpaint(
                roi,
                mask_roi.astype(np.uint8) * 255,
                3.0,
                cv2.INPAINT_TELEA,
            )
            transformed_roi = roi.copy()
            transformed_mask = mask_roi.copy()

        for cell_index in selected:
            row, column = divmod(int(cell_index), grid)
            local_x1 = boundaries_x[column] - x1
            local_x2 = boundaries_x[column + 1] - x1
            local_y1 = boundaries_y[row] - y1
            local_y2 = boundaries_y[row + 1] - y1
            tile = roi[local_y1:local_y2, local_x1:local_x2]
            tile_height, tile_width = tile.shape[:2]
            angle = int(rng.choice((90, 180, 270)))
            matrix = cv2.getRotationMatrix2D(
                (tile_width / 2.0, tile_height / 2.0),
                angle,
                1.0,
            )
            rotated = cv2.warpAffine(
                tile,
                matrix,
                (tile_width, tile_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            if mask is None:
                composite[local_y1:local_y2, local_x1:local_x2] = rotated
            else:
                mask_tile = mask_roi[
                    local_y1:local_y2,
                    local_x1:local_x2,
                ].astype(np.uint8)
                rotated_mask = cv2.warpAffine(
                    mask_tile,
                    matrix,
                    (tile_width, tile_height),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ).astype(bool)
                transformed_roi[
                    local_y1:local_y2,
                    local_x1:local_x2,
                ] = rotated
                transformed_mask[
                    local_y1:local_y2,
                    local_x1:local_x2,
                ] = rotated_mask

        if mask is not None:
            composite = background
            composite[transformed_mask] = transformed_roi[transformed_mask]
        output[y1:y2, x1:x2] = composite
        transformed += 1
    return output, TransformStats(transformed, skipped)


def phase_scramble_boxes(
    image_bgr: np.ndarray,
    boxes: Sequence[Sequence[float]],
    severity: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, TransformStats]:
    """Disrupt spatial phase inside boxes while retaining frequency magnitude."""
    source = image_bgr
    output = image_bgr.copy()
    image_height, image_width = image_bgr.shape[:2]
    transformed = 0
    skipped = 0

    for box in boxes:
        x1, y1, x2, y2 = _clipped_box(box, image_width, image_height)
        if x2 - x1 < 4 or y2 - y1 < 4:
            skipped += 1
            continue

        roi = source[y1:y2, x1:x2].astype(np.float32)
        scrambled = np.empty_like(roi)
        for channel_index in range(roi.shape[2]):
            channel = roi[:, :, channel_index]
            spectrum = np.fft.fft2(channel)
            original_unit = np.exp(1j * np.angle(spectrum))
            noise_unit = np.exp(
                1j * np.angle(np.fft.fft2(rng.normal(size=channel.shape)))
            )
            mixed_unit = (
                (1.0 - severity) * original_unit + severity * noise_unit
            )
            mixed_unit /= np.maximum(np.abs(mixed_unit), 1e-8)
            mixed_unit[0, 0] = original_unit[0, 0]
            reconstructed = np.fft.ifft2(np.abs(spectrum) * mixed_unit).real

            original_mean = float(channel.mean())
            original_std = float(channel.std())
            reconstructed_std = float(reconstructed.std())
            if original_std > 1e-6 and reconstructed_std > 1e-6:
                reconstructed = (
                    (reconstructed - reconstructed.mean())
                    * (original_std / reconstructed_std)
                    + original_mean
                )
            scrambled[:, :, channel_index] = reconstructed

        output[y1:y2, x1:x2] = np.clip(scrambled, 0, 255).astype(np.uint8)
        transformed += 1

    return output, TransformStats(transformed, skipped)


def corrupt_background(
    image_bgr: np.ndarray,
    boxes: Sequence[Sequence[float]],
    severity: float,
    rng: np.random.Generator,
    *,
    margin: float = 0.1,
) -> np.ndarray:
    """Replace background progressively while preserving labeled box regions."""
    image_height, image_width = image_bgr.shape[:2]
    protected = np.zeros((image_height, image_width), dtype=bool)
    for box in boxes:
        width = max(0.0, float(box[2]) - float(box[0]))
        height = max(0.0, float(box[3]) - float(box[1]))
        expanded = (
            float(box[0]) - margin * width,
            float(box[1]) - margin * height,
            float(box[2]) + margin * width,
            float(box[3]) + margin * height,
        )
        x1, y1, x2, y2 = _clipped_box(expanded, image_width, image_height)
        protected[y1:y2, x1:x2] = True

    mean = image_bgr.reshape(-1, 3).mean(axis=0, keepdims=True)
    noise = rng.normal(
        loc=mean.reshape(1, 1, 3),
        scale=48.0,
        size=image_bgr.shape,
    )
    corrupted = np.clip(noise, 0, 255).astype(np.uint8)
    output = image_bgr.copy()
    blended = cv2.addWeighted(
        image_bgr,
        1.0 - severity,
        corrupted,
        severity,
        0.0,
    )
    output[~protected] = blended[~protected]
    return output


def dropout_object_patches(
    image_bgr: np.ndarray,
    boxes: Sequence[Sequence[float]],
    severity: float,
    rng: np.random.Generator,
    *,
    grid_size: int = 4,
) -> tuple[np.ndarray, TransformStats]:
    """Remove a severity-controlled fraction of spatial cells inside boxes."""
    source = image_bgr
    output = image_bgr.copy()
    image_height, image_width = image_bgr.shape[:2]
    transformed = 0
    skipped = 0

    for box in boxes:
        x1, y1, x2, y2 = _clipped_box(box, image_width, image_height)
        width = x2 - x1
        height = y2 - y1
        grid = min(grid_size, width // 2, height // 2)
        if grid < 2:
            skipped += 1
            continue

        boundaries_x = np.linspace(x1, x2, grid + 1, dtype=int)
        boundaries_y = np.linspace(y1, y2, grid + 1, dtype=int)
        cell_count = grid * grid
        dropout_count = min(
            cell_count,
            max(1, int(round(severity * cell_count))),
        )
        selected = rng.choice(cell_count, size=dropout_count, replace=False)
        roi = source[y1:y2, x1:x2]
        fill = np.median(roi.reshape(-1, 3), axis=0).astype(np.uint8)
        for cell_index in selected:
            row, column = divmod(int(cell_index), grid)
            output[
                boundaries_y[row] : boundaries_y[row + 1],
                boundaries_x[column] : boundaries_x[column + 1],
            ] = fill
        transformed += 1

    return output, TransformStats(transformed, skipped)


def apply_condition(
    image_bgr: np.ndarray,
    boxes: Sequence[Sequence[float]],
    condition: Condition,
    *,
    seed: int,
    image_id: str,
    shape_grid_size: int = 4,
    instance_masks: Sequence[np.ndarray] | None = None,
    background_bgr: np.ndarray | None = None,
) -> tuple[np.ndarray, TransformStats]:
    """Apply one benchmark condition to an image."""
    if condition.transform == "none":
        return image_bgr.copy(), TransformStats()
    if condition.transform == "object_grayscale":
        transformed = blend_grayscale(image_bgr, condition.severity)
        return (
            apply_to_object_regions(
                image_bgr,
                transformed,
                boxes,
                instance_masks,
            ),
            TransformStats(len(boxes), 0),
        )
    if condition.transform == "object_channel_shuffle":
        transformed = blend_channel_shuffle(image_bgr, condition.severity)
        return (
            apply_to_object_regions(
                image_bgr,
                transformed,
                boxes,
                instance_masks,
            ),
            TransformStats(len(boxes), 0),
        )
    if condition.transform == "object_bilateral":
        transformed = suppress_texture_bilateral(image_bgr, condition.severity)
        return (
            apply_to_object_regions(
                image_bgr,
                transformed,
                boxes,
                instance_masks,
            ),
            TransformStats(len(boxes), 0),
        )
    if condition.transform == "object_gaussian":
        transformed = suppress_texture_gaussian(image_bgr, condition.severity)
        return (
            apply_to_object_regions(
                image_bgr,
                transformed,
                boxes,
                instance_masks,
            ),
            TransformStats(len(boxes), 0),
        )
    if condition.transform == "object_patch_shuffle":
        rng = stable_rng(seed, image_id, condition.key)
        return shuffle_object_patches(
            image_bgr,
            boxes,
            condition.severity,
            rng,
            grid_size=shape_grid_size,
            instance_masks=instance_masks,
        )
    if condition.transform == "object_patch_rotate":
        rng = stable_rng(seed, image_id, condition.key)
        return rotate_object_patches(
            image_bgr,
            boxes,
            condition.severity,
            rng,
            grid_size=shape_grid_size,
            instance_masks=instance_masks,
        )
    if condition.transform == "background_swap":
        if background_bgr is None:
            raise ValueError("background_swap requires a background image")
        rng = stable_rng(seed, image_id, condition.key)
        return (
            replace_background(
                image_bgr,
                background_bgr,
                boxes,
                condition.severity,
                instance_masks,
                rng,
            ),
            TransformStats(),
        )
    if condition.transform == "hue_shift":
        return shift_hue(image_bgr, condition.severity), TransformStats()
    if condition.transform == "grayscale_blend":
        return blend_grayscale(image_bgr, condition.severity), TransformStats()
    if condition.transform == "posterize":
        return posterize(image_bgr, condition.severity), TransformStats()
    if condition.transform == "edge_only":
        return edge_only(image_bgr), TransformStats()
    if condition.transform == "patch_shuffle":
        rng = stable_rng(seed, image_id, condition.key)
        return shuffle_box_patches(
            image_bgr,
            boxes,
            condition.severity,
            rng,
            grid_size=shape_grid_size,
        )
    if condition.transform == "phase_scramble":
        rng = stable_rng(seed, image_id, condition.key)
        return phase_scramble_boxes(
            image_bgr,
            boxes,
            condition.severity,
            rng,
        )
    if condition.transform == "background_noise":
        rng = stable_rng(seed, image_id, condition.key)
        return (
            corrupt_background(
                image_bgr,
                boxes,
                condition.severity,
                rng,
            ),
            TransformStats(),
        )
    if condition.transform == "object_dropout":
        rng = stable_rng(seed, image_id, condition.key)
        return dropout_object_patches(
            image_bgr,
            boxes,
            condition.severity,
            rng,
            grid_size=shape_grid_size,
        )
    raise ValueError(f"Unsupported transform: {condition.transform}")


def aggregate_shape_scores(
    condition_metrics: dict[str, dict[str, float]],
    conditions: Sequence[Condition],
    *,
    metric_name: str = "map50_95",
) -> list[dict[str, float | None]]:
    """Aggregate appearance and shape retention at every severity."""
    clean_metric = condition_metrics["clean"][metric_name]
    severities = sorted(
        {condition.severity for condition in conditions if condition.category == "shape"}
    )
    rows: list[dict[str, float | None]] = []

    for severity in severities:
        appearance_keys = [
            condition.key
            for condition in conditions
            if condition.category == "appearance" and condition.severity == severity
        ]
        shape_keys = [
            condition.key
            for condition in conditions
            if condition.category == "shape" and condition.severity == severity
        ]
        appearance_metric = float(
            np.mean([condition_metrics[key][metric_name] for key in appearance_keys])
        )
        shape_metric = float(
            np.mean([condition_metrics[key][metric_name] for key in shape_keys])
        )

        absolute_shape_gap = appearance_metric - shape_metric
        if clean_metric > 0:
            appearance_retention = appearance_metric / clean_metric
            shape_retention = shape_metric / clean_metric
            shape_impact = 1.0 - shape_retention
            shape_reliance_gap = appearance_retention - shape_retention
        else:
            appearance_retention = None
            shape_retention = None
            shape_impact = None
            shape_reliance_gap = None

        rows.append(
            {
                "severity": severity,
                "clean_metric": clean_metric,
                "appearance_metric": appearance_metric,
                "shape_metric": shape_metric,
                "appearance_retention": appearance_retention,
                "shape_retention": shape_retention,
                "shape_impact": shape_impact,
                "shape_reliance_gap": shape_reliance_gap,
                "absolute_shape_gap": absolute_shape_gap,
            }
        )

    return rows


def aggregate_category_retention(
    condition_metrics: dict[str, dict[str, float]],
    conditions: Sequence[Condition],
    *,
    metric_name: str = "map50_95",
) -> list[dict[str, str | float | None]]:
    """Report clean-normalized retention for every causal condition category."""
    clean_metric = condition_metrics["clean"][metric_name]
    categories = sorted(
        {
            condition.category
            for condition in conditions
            if condition.category not in {"clean", "diagnostic"}
        }
    )
    severities = sorted(
        {
            condition.severity
            for condition in conditions
            if condition.category in categories
        }
    )
    rows: list[dict[str, str | float | None]] = []
    for severity in severities:
        for category in categories:
            keys = [
                condition.key
                for condition in conditions
                if condition.category == category
                and condition.severity == severity
            ]
            if not keys:
                continue
            metric = float(
                np.mean([condition_metrics[key][metric_name] for key in keys])
            )
            rows.append(
                {
                    "severity": severity,
                    "category": category,
                    "metric": metric,
                    "retention": metric / clean_metric if clean_metric > 0 else None,
                }
            )
    return rows


def aggregate_reliance_auc(
    category_rows: Sequence[dict[str, str | float | None]],
) -> list[dict[str, str | float | None]]:
    """Integrate normalized retention curves into one reliance value per cue."""
    categories = sorted({str(row["category"]) for row in category_rows})
    results: list[dict[str, str | float | None]] = []
    for category in categories:
        points = sorted(
            (
                float(row["severity"]),
                row["retention"],
            )
            for row in category_rows
            if row["category"] == category and row["retention"] is not None
        )
        if len(points) < 2:
            results.append(
                {
                    "category": category,
                    "max_severity": points[-1][0] if points else None,
                    "retention_auc": None,
                    "cue_reliance": None,
                }
            )
            continue
        curve = [(0.0, 1.0)] + [
            (severity, float(retention))
            for severity, retention in points
        ]
        area = sum(
            (right_x - left_x) * (left_y + right_y) / 2.0
            for (left_x, left_y), (right_x, right_y) in zip(
                curve,
                curve[1:],
                strict=False,
            )
        )
        max_severity = curve[-1][0]
        retention_auc = area / max_severity if max_severity > 0 else None
        results.append(
            {
                "category": category,
                "max_severity": max_severity,
                "retention_auc": retention_auc,
                "cue_reliance": (
                    1.0 - retention_auc
                    if retention_auc is not None
                    else None
                ),
            }
        )
    return results
