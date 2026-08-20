import tempfile
import unittest
from pathlib import Path

import numpy as np

from ssl_experiments.evaluation.shape_benchmark import _load_yolo_segment_masks
from ssl_experiments.evaluation.shape_reliance import (
    aggregate_category_retention,
    aggregate_reliance_auc,
    aggregate_shape_scores,
    apply_condition,
    build_conditions,
    build_object_region_mask,
    build_v2_conditions,
    corrupt_background,
    phase_scramble_boxes,
    posterize,
    replace_background,
    rotate_object_patches,
    shuffle_box_patches,
)


class ShapeRelianceTest(unittest.TestCase):
    def test_conditions_are_built_for_each_severity(self):
        conditions = build_conditions([0.5, 0.25, 0.5], include_edge=True)

        self.assertEqual(conditions[0].key, "clean")
        self.assertEqual(
            [condition.key for condition in conditions],
            [
                "clean",
                "appearance_hue_s0p25",
                "appearance_gray_s0p25",
                "shape_shuffle_s0p25",
                "appearance_hue_s0p5",
                "appearance_gray_s0p5",
                "shape_shuffle_s0p5",
                "edge_only",
            ],
        )

    def test_hard_mode_adds_independent_condition_categories(self):
        conditions = build_conditions([0.5], include_edge=True, hard_mode=True)

        self.assertEqual(
            [(condition.category, condition.transform) for condition in conditions],
            [
                ("clean", "none"),
                ("appearance", "hue_shift"),
                ("appearance", "grayscale_blend"),
                ("shape", "patch_shuffle"),
                ("appearance", "posterize"),
                ("shape", "phase_scramble"),
                ("context", "background_noise"),
                ("stress", "object_dropout"),
                ("diagnostic", "edge_only"),
            ],
        )

    def test_hard_transforms_are_deterministic_and_box_aware(self):
        rng = np.random.default_rng(3)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        boxes = [[8, 8, 24, 24]]

        first, stats = phase_scramble_boxes(
            image,
            boxes,
            severity=0.75,
            rng=np.random.default_rng(11),
        )
        second, _ = phase_scramble_boxes(
            image,
            boxes,
            severity=0.75,
            rng=np.random.default_rng(11),
        )
        self.assertEqual(stats.transformed_objects, 1)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first[8:24, 8:24], image[8:24, 8:24]))
        np.testing.assert_array_equal(first[:8], image[:8])

        background = corrupt_background(
            image,
            boxes,
            severity=1.0,
            rng=np.random.default_rng(9),
            margin=0.0,
        )
        np.testing.assert_array_equal(background[8:24, 8:24], image[8:24, 8:24])
        self.assertFalse(np.array_equal(background[:8], image[:8]))

        reduced = posterize(image, severity=1.0)
        self.assertTrue(np.all(np.isin(reduced, [32, 96, 160, 224])))

    def test_patch_shuffle_preserves_box_pixels(self):
        image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)

        shuffled, stats = shuffle_box_patches(
            image,
            [[0, 0, 16, 16]],
            severity=1.0,
            rng=np.random.default_rng(42),
            grid_size=4,
        )

        self.assertEqual(stats.transformed_objects, 1)
        self.assertEqual(stats.skipped_objects, 0)
        self.assertFalse(np.array_equal(image, shuffled))
        np.testing.assert_array_equal(
            np.sort(image.reshape(-1)),
            np.sort(shuffled.reshape(-1)),
        )

    def test_patch_shuffle_is_deterministic_per_image(self):
        image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        condition = next(
            condition
            for condition in build_conditions([0.75], include_edge=False)
            if condition.category == "shape"
        )

        first, _ = apply_condition(
            image,
            [[0, 0, 16, 16]],
            condition,
            seed=7,
            image_id="sample.jpg",
        )
        second, _ = apply_condition(
            image,
            [[0, 0, 16, 16]],
            condition,
            seed=7,
            image_id="sample.jpg",
        )

        np.testing.assert_array_equal(first, second)

    def test_aggregate_scores_use_clean_metric(self):
        conditions = build_conditions([0.5], include_edge=False)
        metrics = {
            "clean": {"map50_95": 0.8},
            "appearance_hue_s0p5": {"map50_95": 0.7},
            "appearance_gray_s0p5": {"map50_95": 0.74},
            "shape_shuffle_s0p5": {"map50_95": 0.4},
        }

        scores = aggregate_shape_scores(metrics, conditions)

        self.assertEqual(len(scores), 1)
        self.assertAlmostEqual(scores[0]["appearance_metric"], 0.72)
        self.assertAlmostEqual(scores[0]["appearance_retention"], 0.9)
        self.assertAlmostEqual(scores[0]["shape_retention"], 0.5)
        self.assertAlmostEqual(scores[0]["shape_impact"], 0.5)
        self.assertAlmostEqual(scores[0]["shape_reliance_gap"], 0.4)
        self.assertAlmostEqual(scores[0]["absolute_shape_gap"], 0.32)

    def test_hard_mode_aggregates_each_category_separately(self):
        conditions = build_conditions([0.5], include_edge=False, hard_mode=True)
        metrics = {
            "clean": {"map50_95": 0.8},
            "appearance_hue_s0p5": {"map50_95": 0.7},
            "appearance_gray_s0p5": {"map50_95": 0.74},
            "appearance_posterize_s0p5": {"map50_95": 0.72},
            "shape_shuffle_s0p5": {"map50_95": 0.4},
            "shape_phase_s0p5": {"map50_95": 0.32},
            "context_noise_s0p5": {"map50_95": 0.64},
            "stress_dropout_s0p5": {"map50_95": 0.24},
        }

        scores = aggregate_shape_scores(metrics, conditions)
        categories = aggregate_category_retention(metrics, conditions)
        retention = {row["category"]: row["retention"] for row in categories}

        self.assertAlmostEqual(scores[0]["appearance_metric"], 0.72)
        self.assertAlmostEqual(scores[0]["shape_metric"], 0.36)
        self.assertAlmostEqual(scores[0]["shape_reliance_gap"], 0.45)
        self.assertAlmostEqual(retention["appearance"], 0.9)
        self.assertAlmostEqual(retention["shape"], 0.45)
        self.assertAlmostEqual(retention["context"], 0.8)
        self.assertAlmostEqual(retention["stress"], 0.3)


    def test_v2_conditions_separate_visual_cues(self):
        conditions = build_v2_conditions([0.5], include_edge=True)

        self.assertEqual(
            [(condition.category, condition.transform) for condition in conditions],
            [
                ("clean", "none"),
                ("color", "object_grayscale"),
                ("color", "object_channel_shuffle"),
                ("texture", "object_bilateral"),
                ("texture", "object_gaussian"),
                ("shape", "object_patch_shuffle"),
                ("shape", "object_patch_rotate"),
                ("context", "background_swap"),
                ("diagnostic", "edge_only"),
            ],
        )

    def test_v2_color_transform_changes_only_object_region(self):
        rng = np.random.default_rng(19)
        image = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
        condition = next(
            condition
            for condition in build_v2_conditions([1.0], include_edge=False)
            if condition.transform == "object_grayscale"
        )

        transformed, stats = apply_condition(
            image,
            [[6, 7, 18, 19]],
            condition,
            seed=0,
            image_id="sample.jpg",
        )

        self.assertEqual(stats.transformed_objects, 1)
        np.testing.assert_array_equal(transformed[:7], image[:7])
        self.assertFalse(np.array_equal(transformed[7:19, 6:18], image[7:19, 6:18]))

    def test_mask_context_swap_preserves_only_masked_object(self):
        image = np.full((16, 16, 3), 40, dtype=np.uint8)
        donor = np.full((16, 16, 3), 220, dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=bool)
        mask[6:10, 6:10] = True
        boxes = [[4, 4, 12, 12]]

        transformed = replace_background(
            image,
            donor,
            boxes,
            severity=1.0,
            instance_masks=[mask],
        )

        np.testing.assert_array_equal(transformed[mask], image[mask])
        self.assertTrue(np.all(transformed[~mask] == 220))
        region = build_object_region_mask(image.shape, boxes, [mask])
        np.testing.assert_array_equal(region, mask)

        partial_first = replace_background(
            image,
            donor,
            boxes,
            severity=0.5,
            instance_masks=[mask],
            rng=np.random.default_rng(17),
        )
        partial_second = replace_background(
            image,
            donor,
            boxes,
            severity=0.5,
            instance_masks=[mask],
            rng=np.random.default_rng(17),
        )
        np.testing.assert_array_equal(partial_first, partial_second)
        np.testing.assert_array_equal(partial_first[mask], image[mask])
        self.assertFalse(np.array_equal(partial_first[~mask], image[~mask]))
        self.assertFalse(np.array_equal(partial_first[~mask], donor[~mask]))

    def test_mask_patch_rotation_is_deterministic(self):
        rng = np.random.default_rng(5)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 10:22] = True
        boxes = [[8, 8, 24, 24]]

        first, first_stats = rotate_object_patches(
            image,
            boxes,
            severity=1.0,
            rng=np.random.default_rng(13),
            instance_masks=[mask],
        )
        second, _ = rotate_object_patches(
            image,
            boxes,
            severity=1.0,
            rng=np.random.default_rng(13),
            instance_masks=[mask],
        )

        self.assertEqual(first_stats.transformed_objects, 1)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:8], image[:8])
        self.assertFalse(np.array_equal(first[8:24, 8:24], image[8:24, 8:24]))

    def test_reliance_auc_integrates_from_clean_retention(self):
        rows = [
            {"severity": 0.5, "category": "shape", "retention": 0.8},
            {"severity": 1.0, "category": "shape", "retention": 0.6},
            {"severity": 0.5, "category": "texture", "retention": 0.9},
            {"severity": 1.0, "category": "texture", "retention": 0.8},
        ]

        results = {
            row["category"]: row
            for row in aggregate_reliance_auc(rows)
        }

        self.assertAlmostEqual(results["shape"]["retention_auc"], 0.8)
        self.assertAlmostEqual(results["shape"]["cue_reliance"], 0.2)
        self.assertAlmostEqual(results["texture"]["retention_auc"], 0.9)
        self.assertAlmostEqual(results["texture"]["cue_reliance"], 0.1)


    def test_yolo_segmentation_polygons_load_as_instance_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text("2 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n")

            masks, classes = _load_yolo_segment_masks(path, 20, 20)

        self.assertEqual(classes, [2])
        self.assertEqual(len(masks), 1)
        self.assertTrue(masks[0][10, 10])
        self.assertFalse(masks[0][1, 1])


if __name__ == "__main__":
    unittest.main()
