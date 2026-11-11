"""Coupled view construction; configurable reconstruction, not historical code."""

from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision.transforms.functional import to_tensor


OPPOSED_GROUPS = (
    "context_crop",
    "color_contour",
    "complementary_mask",
    "exposure",
    "sharp_blur",
    "perspective",
    "object_background_focus",
    "motion",
    "posterize",
    "morphology",
)


def object_mask(
    image: Image.Image, box: Any
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    width, height = image.size
    x1 = max(0, min(width - 1, int((box.center_x - box.width / 2) * width)))
    y1 = max(0, min(height - 1, int((box.center_y - box.height / 2) * height)))
    x2 = min(width, max(x1 + 1, round((box.center_x + box.width / 2) * width)))
    y2 = min(height, max(y1 + 1, round((box.center_y + box.height / 2) * height)))
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask, (x1, y1, x2, y2)


class OpposedViewTransform:
    """Select one coupled group, retaining the ordinary VICReg objective.

    Magnitudes below define this implementation only. Bounding boxes guide view
    construction and never become VICReg loss targets.
    """

    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def __call__(
        self, image: Image.Image, boxes: list[Any], *, group: int | None = None
    ):
        allowed = list(range(len(OPPOSED_GROUPS))) if boxes else [1, 3, 4, 5, 7, 8, 9]
        group = random.choice(allowed) if group is None else group
        if group not in allowed:
            raise ValueError(
                "Selected opposed group is invalid or requires a bounding box"
            )
        image = image.convert("RGB")
        pixels = np.asarray(image)
        box = random.choice(boxes) if boxes else None
        mask, bounds = object_mask(image, box) if box is not None else (None, None)
        first, second = image.copy(), image.copy()
        if group == 0:
            factor = random.uniform(0.35, 0.65)
            small = image.resize(
                (
                    max(1, round(image.width * factor)),
                    max(1, round(image.height * factor)),
                )
            )
            first = Image.new(
                "RGB", image.size, tuple(pixels.mean(axis=(0, 1)).astype(int))
            )
            first.paste(
                small,
                ((image.width - small.width) // 2, (image.height - small.height) // 2),
            )
            second = image.crop(bounds)
        elif group == 1:
            first = ImageEnhance.Color(image).enhance(random.uniform(1.7, 2.5))
            first = ImageEnhance.Contrast(first).enhance(random.uniform(1.5, 2.0))
            gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
            dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            edge = cv2.convertScaleAbs(cv2.magnitude(dx, dy))
            second = Image.fromarray(edge).convert("RGB")
        elif group == 2:
            x1, y1, x2, y2 = bounds
            left, right = pixels.copy(), pixels.copy()
            if random.random() < 0.5:
                middle = (x1 + x2) // 2
                left[y1:y2, x1:middle] = 0
                right[y1:y2, middle:x2] = 0
            else:
                middle = (y1 + y2) // 2
                left[y1:middle, x1:x2] = 0
                right[middle:y2, x1:x2] = 0
            first, second = Image.fromarray(left), Image.fromarray(right)
        elif group == 3:
            first = ImageEnhance.Brightness(image).enhance(random.uniform(1.8, 2.5))
            second = ImageEnhance.Contrast(
                ImageEnhance.Brightness(image).enhance(0.4)
            ).enhance(0.35)
        elif group == 4:
            sharp = np.asarray(
                image.filter(ImageFilter.UnsharpMask(radius=2, percent=220))
            ).astype(float)
            rng = np.random.default_rng(random.randrange(2**32))
            first = Image.fromarray(
                np.clip(sharp + rng.normal(0, 8, sharp.shape), 0, 255).astype(np.uint8)
            )
            second = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(2, 4)))
        elif group == 5:
            width, height = image.size
            source = np.float32(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
            )
            target = source.copy()
            target[0, 0] += width * random.uniform(0.1, 0.25)
            target[1, 1] += height * random.uniform(0.1, 0.25)
            target[2, 0] -= width * random.uniform(0.1, 0.25)
            second = Image.fromarray(
                cv2.warpPerspective(
                    pixels, cv2.getPerspectiveTransform(source, target), image.size
                )
            )
        elif group == 6:
            blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=3)))
            first = Image.fromarray(np.where(mask[..., None], pixels, blurred))
            second = Image.fromarray(np.where(mask[..., None], blurred, pixels))
        elif group == 7:
            length = random.choice([3, 5, 7])
            kernel = np.zeros((length, length), np.float32)
            kernel[length // 2, :] = 1 / length
            first = Image.fromarray(cv2.filter2D(pixels, -1, kernel))
            second = Image.fromarray(cv2.filter2D(pixels, -1, kernel.T))
        elif group == 8:
            second = ImageOps.posterize(image, bits=random.choice([2, 3, 4]))
        elif group == 9:
            kernel = np.ones((3, 3), np.uint8)
            first = Image.fromarray(cv2.dilate(pixels, kernel))
            second = Image.fromarray(cv2.erode(pixels, kernel))
        size = (self.image_size, self.image_size)
        return to_tensor(first.resize(size, Image.Resampling.BILINEAR)), to_tensor(
            second.resize(size, Image.Resampling.BILINEAR)
        )
