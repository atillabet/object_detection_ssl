from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
from PIL import Image

from ssl_experiments.explainability.core import (
    Detection,
    GroundTruth,
    box_iou_xyxy,
    compute_calibration_outputs,
    find_images,
    letterbox_content_region,
    match_detections_to_ground_truth,
    resize_letterboxed_heatmap_to_original,
    reset_detector_inference_cache,
    select_detections_for_explanation,
    write_rows_parquet,
)


def detection(
    detection_id: int,
    class_id: int,
    confidence: float,
    box: list[float],
) -> Detection:
    return Detection(
        detection_id=detection_id,
        cls_id=class_id,
        cls_name=str(class_id),
        conf=confidence,
        box_xyxy=np.asarray(box, dtype=np.float32),
    )


class ExplainabilityGeometryTests(unittest.TestCase):
    def test_detector_inference_cache_is_invalidated_for_gradients(self) -> None:
        head = SimpleNamespace(shape=(1, 3, 10, 10))
        model = SimpleNamespace(model=SimpleNamespace(model=[head]))

        reset_detector_inference_cache(model)

        self.assertIsNone(head.shape)

    def test_find_images_filters_sorts_and_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("b.png", "a.jpg", "ignored.txt"):
                if Path(name).suffix == ".txt":
                    (root / name).write_text("not an image", encoding="utf-8")
                else:
                    Image.new("RGB", (8, 8)).save(root / name)

            images = find_images(root, max_images=1)

        self.assertEqual([path.name for path in images], ["a.jpg"])

    def test_iou_and_letterbox_mapping(self) -> None:
        iou = box_iou_xyxy(
            np.asarray([0, 0, 10, 10], dtype=np.float32),
            np.asarray([5, 5, 15, 15], dtype=np.float32),
        )
        self.assertAlmostEqual(iou, 25 / 175)
        self.assertEqual(letterbox_content_region((100, 200), (320, 320)), (0, 80, 320, 240))

        heatmap = np.zeros((10, 10), dtype=np.float32)
        heatmap[3:7, 2:8] = 1.0
        resized, region = resize_letterboxed_heatmap_to_original(
            heatmap,
            input_shape=(320, 320),
            original_shape=(100, 200),
        )

        self.assertEqual(resized.shape, (100, 200))
        self.assertEqual(region, (0, 80, 320, 240))
        self.assertAlmostEqual(float(resized.max()), 1.0)


class ExplainabilityDiagnosisTests(unittest.TestCase):
    def test_matching_identifies_tp_duplicate_confusion_and_localization(self) -> None:
        ground_truths = [
            GroundTruth(1, 0, "car", np.asarray([0, 0, 10, 10], dtype=np.float32))
        ]
        detections = [
            detection(1, 0, 0.95, [0, 0, 10, 10]),
            detection(2, 0, 0.80, [0, 0, 10, 10]),
            detection(3, 1, 0.70, [0, 0, 10, 10]),
            detection(4, 0, 0.60, [5, 5, 15, 15]),
        ]

        matches, false_negatives, summary = match_detections_to_ground_truth(
            detections,
            ground_truths,
            iou_threshold=0.5,
        )

        self.assertEqual(matches[1]["gt_status"], "tp")
        self.assertEqual(matches[2]["fp_type"], "duplicate")
        self.assertEqual(matches[3]["fp_type"], "class_confusion")
        self.assertEqual(matches[4]["fp_type"], "localization")
        self.assertEqual(false_negatives, [])
        self.assertEqual(summary["num_tp"], 1)
        self.assertEqual(summary["num_fp"], 3)

    def test_selection_modes_are_deterministic(self) -> None:
        detections = [
            detection(1, 0, 0.9, [0, 0, 2, 2]),
            detection(2, 0, 0.2, [0, 0, 2, 2]),
            detection(3, 0, 0.7, [0, 0, 2, 2]),
        ]
        matches = {
            1: {"gt_status": "tp"},
            2: {"gt_status": "fp"},
            3: {"gt_status": "fp"},
        }

        top = select_detections_for_explanation(detections, matches, 2, "top_conf")
        low = select_detections_for_explanation(detections, matches, 2, "low_conf")
        false_positive = select_detections_for_explanation(
            detections,
            matches,
            2,
            "false_positive",
        )

        self.assertEqual([item.detection_id for item in top], [1, 3])
        self.assertEqual([item.detection_id for item in low], [2, 3])
        self.assertEqual([item.detection_id for item in false_positive], [3, 2])

    def test_calibration_and_parquet_outputs(self) -> None:
        image_rows = [{"num_ground_truth": 2}]
        prediction_rows = [
            {"pipeline": "source", "class_name": "car", "score": 0.9, "gt_status": "tp"},
            {"pipeline": "source", "class_name": "car", "score": 0.8, "gt_status": "fp"},
            {"pipeline": "counterfactual", "class_name": "car", "score": 1.0, "gt_status": "tp"},
        ]

        calibration, thresholds, summary = compute_calibration_outputs(
            image_rows,
            prediction_rows,
        )

        self.assertEqual(summary["num_predictions"], 2)
        self.assertEqual(summary["num_ground_truth"], 2)
        self.assertTrue(calibration)
        self.assertEqual(len([row for row in thresholds if row["class_name"] == "all"]), 101)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rows.parquet"
            write_rows_parquet(output, [{"value": 3}], ["value"])
            frame = pl.read_parquet(output)

        self.assertEqual(frame.to_dicts(), [{"value": 3}])


if __name__ == "__main__":
    unittest.main()
