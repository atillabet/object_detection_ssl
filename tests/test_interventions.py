from __future__ import annotations

import random
import unittest

import numpy as np
from PIL import Image

from ssl_experiments.data.interventions.object_centered import (
    Box as CenteredBox,
    box_visible_enough,
    clamp_crop,
    yolo_line,
)
from ssl_experiments.data.interventions.object_driven import (
    Box as DrivenBox,
    clip_box_to_crop,
    visible_enough,
)
from ssl_experiments.data.interventions.stylize import stylize_image


class ObjectInterventionTests(unittest.TestCase):
    def test_object_driven_clipping_tracks_visibility(self) -> None:
        original = DrivenBox(0, 10, 10, 30, 30)
        clipped = clip_box_to_crop(original, (20, 0, 40, 40))
        assert clipped is not None

        self.assertEqual((clipped.x1, clipped.x2), (0, 10))
        self.assertTrue(visible_enough(original, clipped, 0.5, 4))
        self.assertFalse(visible_enough(original, clipped, 0.75, 4))

    def test_object_centered_labels_are_normalized(self) -> None:
        box = CenteredBox(2, 10, 20, 30, 40)
        line = yolo_line(box, 40, 80)

        self.assertEqual(line, "2 0.500000 0.375000 0.500000 0.250000")
        self.assertTrue(box_visible_enough(box, box, 1.0, 4))
        self.assertEqual(clamp_crop(-5, -5, 80, 90, 40, 80), (0, 0, 40, 80))

    def test_texture_suppression_is_seed_deterministic(self) -> None:
        pixels = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        image = Image.fromarray(pixels)
        kwargs = {
            "mode": "texture_suppressed",
            "sigma_s": 80.0,
            "sigma_r": 0.45,
            "color_levels": 7,
            "detail_weight": 0.15,
            "edge_strength": 0.6,
            "tint_strength": 18,
            "edge_boost": 24.0,
        }

        first = stylize_image(image, rng=random.Random(7), **kwargs)
        second = stylize_image(image, rng=random.Random(7), **kwargs)

        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
        self.assertFalse(np.array_equal(np.asarray(first), pixels))


if __name__ == "__main__":
    unittest.main()
