from __future__ import annotations

import argparse
import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from ssl_experiments.commands import build_finetune_command, build_pretrain_command
from ssl_experiments.config import load_study
from ssl_experiments.data.interventions.object_driven import (
    Box,
    Sample,
    feather_mask,
    generate_dataset,
    paste_object,
    validate_output_path,
)
from ssl_experiments.evaluation.operating_points import (
    clean_matches,
    common_retention,
    common_targets,
    select_f1_threshold,
)
from ssl_experiments.evaluation.thesis import find_samples, validate_calibration
from ssl_experiments.evaluation.thesis import calibrate, objectness
from ssl_experiments.evaluation.sahi import sahi_inference_batched, slice_image
from ssl_experiments.methods.opposed_views import OPPOSED_GROUPS, OpposedViewTransform
from ssl_experiments.methods.vicreg import YoloBox
from ssl_experiments.training.finetune_detector import (
    configure_augmentations,
    make_backbone_freezer,
)
from ssl_experiments.training.pretrain_vicreg import validate_args


class GeneratorRegressionTests(unittest.TestCase):
    def test_source_overwrite_is_rejected_before_modification(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dataset"
            source.mkdir()
            sentinel = source / "data.yaml"
            sentinel.write_text("unchanged")
            for output in (source, source.parent):
                with self.assertRaises(ValueError):
                    validate_output_path(output, [source, sentinel])
            self.assertEqual(sentinel.read_text(), "unchanged")
            validate_output_path(source.parent / "augmented", [source])

    def test_generator_rejects_source_output_even_with_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            data = source / "data.yaml"
            data.write_text("train: images\n")
            with self.assertRaisesRegex(ValueError, "contains source data"):
                generate_dataset(
                    data,
                    source,
                    copy_paste_ratio=0,
                    background_dropout_ratio=0,
                    tight_crop_ratio=0,
                    robust_ratio=0,
                    max_per_type=None,
                    seed=0,
                    overwrite=True,
                    copy_files=True,
                    link_originals=True,
                    preserve_eval_splits=True,
                    paste_crop_margin=0.03,
                    paste_scale_min=1,
                    paste_scale_max=1,
                    paste_max_iou=0.05,
                    background_dropout_margin=0.08,
                    crop_margin_min=0.15,
                    crop_margin_max=0.45,
                    crop_jitter=0.15,
                    min_visible_fraction=0.6,
                    min_box_size=4,
                    max_source_scan=None,
                )
            self.assertEqual(data.read_text(), "train: images\n")

    def test_blending_has_transparent_border_and_opaque_center(self):
        pixels = np.asarray(feather_mask((64, 64)))
        self.assertLess(int(pixels[0, 0]), int(pixels[32, 32]))
        self.assertEqual(int(pixels[32, 32]), 255)
        self.assertTrue(np.any((pixels > 0) & (pixels < 255)))

    def test_copy_paste_transfers_all_visible_source_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "src.png", root / "dst.png"
            Image.new("RGB", (80, 80), "red").save(source)
            Image.new("RGB", (160, 160), "white").save(destination)
            sample = Sample(
                source,
                root / "src.txt",
                (Box(0, 10, 10, 50, 50), Box(1, 25, 25, 35, 35)),
                80,
                80,
            )
            background = Sample(destination, root / "dst.txt", (), 160, 160)
            result = paste_object(
                sample,
                background,
                random.Random(1),
                crop_margin=0.03,
                scale_min=1.01,
                scale_max=1.01,
                min_box_size=1,
                max_iou=0.05,
                attempts=5,
            )
            self.assertIsNotNone(result)
            image, boxes = result
            self.assertEqual([box.class_id for box in boxes], [0, 1])
            for box in boxes:
                self.assertTrue(0 <= box.x1 < box.x2 <= image.width)
                self.assertTrue(0 <= box.y1 < box.y2 <= image.height)


class OperatingPointTests(unittest.TestCase):
    def record(self, predictions):
        return {
            "one.png": {
                "boxes": [[0, 0, 10, 10], [20, 20, 30, 30]],
                "classes": [0, 0],
                "predictions": predictions,
            }
        }

    def test_f1_maximum_and_equal_confidences(self):
        records = self.record(
            [
                (0, 0.9, [0, 0, 10, 10]),
                (0, 0.8, [60, 60, 70, 70]),
                (0, 0.7, [20, 20, 30, 30]),
                (0, 0.7, [80, 80, 90, 90]),
            ]
        )
        point = select_f1_threshold(records, match_iou=0.5)
        # At 0.9 and 0.7 F1 ties; retain the higher threshold, not a partial tie group.
        self.assertEqual(point["threshold"], 0.9)
        self.assertEqual(point["tp"], 1)

    def test_model_specific_thresholds_and_common_identity_intersection(self):
        first = clean_matches(
            self.record([(0, 0.9, [0, 0, 10, 10]), (0, 0.6, [20, 20, 30, 30])]),
            threshold=0.5,
            match_iou=0.5,
        )
        second = clean_matches(
            self.record([(0, 0.8, [0, 0, 10, 10]), (0, 0.3, [20, 20, 30, 30])]),
            threshold=0.7,
            match_iou=0.5,
        )
        shared = common_targets({"first": first, "second": second, "third": first})
        self.assertEqual(shared, {("one.png", 0)})
        changed = self.record([(0, 0.95, [20, 20, 30, 30])])
        result = common_retention(changed, first, shared, threshold=0.5, match_iou=0.5)
        self.assertEqual(result["retention"], 0)
        self.assertAlmostEqual(result["mean_confidence_delta"], -0.9)
        self.assertLess(result["mean_iou_delta"], -0.99)

    def test_no_shared_targets_is_an_error(self):
        with self.assertRaises(ValueError):
            common_targets({"a": {("one", 0): (0.8, 1)}, "b": {("two", 0): (0.8, 1)}})

    def test_zero_f1_still_counts_predictions_at_threshold_one(self):
        point = select_f1_threshold(
            self.record([(0, 1.0, [60, 60, 70, 70])]), match_iou=0.5
        )
        self.assertEqual(point["threshold"], 1.0)
        self.assertEqual(point["fp"], 1)

    def test_calibration_rejects_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pt"
            checkpoint.write_bytes(b"new weights")
            payload = {
                "inference": {"confidence_floor": 0.001},
                "models": {"model": {"checkpoint_sha256": "old", "threshold": 0.5}},
            }
            with self.assertRaisesRegex(ValueError, "Checkpoint changed"):
                validate_calibration(
                    payload, {"model": checkpoint}, payload["inference"]
                )

    def test_missing_labels_and_wrong_sample_size_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (10, 10)).save(root / "image.png")
            with self.assertRaises(ValueError):
                find_samples(root, root, expected=2000)
            with self.assertRaises(FileNotFoundError):
                find_samples(root, root, expected=1)


class ThesisTrainingTests(unittest.TestCase):
    def test_all_opposed_groups_produce_finite_distinct_views(self):
        image = Image.fromarray(
            np.random.default_rng(2).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        )
        transform = OpposedViewTransform(64)
        for group in range(len(OPPOSED_GROUPS)):
            random.seed(group)
            first, second = transform(
                image, [YoloBox(0, 0.5, 0.5, 0.5, 0.5)], group=group
            )
            self.assertEqual(tuple(first.shape), (3, 64, 64))
            self.assertTrue(
                torch.isfinite(first).all() and torch.isfinite(second).all()
            )
            self.assertFalse(torch.equal(first, second), OPPOSED_GROUPS[group])

    def test_opposed_mode_rejects_auxiliary_loss(self):
        args = argparse.Namespace(
            view_pairing="opposed",
            shape_loss_weight=0.1,
            shape_view_probability=0.0,
            auxiliary_shape_probability=0.0,
            spatial_levels=[1],
        )
        with self.assertRaisesRegex(ValueError, "ordinary global VICReg"):
            validate_args(args)

    def test_frozen_backbone_preserves_batchnorm_and_leaves_head_trainable(self):
        backbone = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3))
        head = nn.Conv2d(3, 1, 1)
        model = SimpleNamespace(model=nn.Sequential(backbone, head))
        model.model.train()
        freeze = make_backbone_freezer(1)
        freeze(SimpleNamespace(model=model))
        before = backbone[1].running_mean.clone()
        model.model(torch.randn(4, 3, 8, 8)).sum().backward()
        self.assertTrue(torch.equal(before, backbone[1].running_mean))
        self.assertTrue(
            all(p.grad is None and not p.requires_grad for p in backbone.parameters())
        )
        self.assertTrue(
            all(p.grad is not None and p.requires_grad for p in head.parameters())
        )

    def test_prepared_profile_disables_stock_augmentation(self):
        settings = {"mosaic": 0.5, "hsv_h": 0.1, "epochs": 10}
        configure_augmentations(settings, "prepared")
        self.assertEqual(settings["fliplr"], 0)
        self.assertEqual(settings["mosaic"], 0)
        self.assertEqual(settings["epochs"], 10)

    def test_thesis_plan_has_coco_frozen_vicreg_and_opposed_pairs(self):
        root = Path(__file__).resolve().parents[1]
        with patch.dict(
            os.environ,
            {
                "SSL_PYTHON": "python",
                "THESIS_TRAIN_IMAGES": "/tiles",
                "THESIS_TRAIN_LABELS": "/labels",
                "THESIS_TRAIN_DATA": "/train.yaml",
                "THESIS_IMPROVED_DATA": "/augmented.yaml",
            },
        ):
            config = load_study(root / "configs/studies/thesis_vicreg.yaml")
            method = config.method("improved_vicreg")
            pretrain = " ".join(build_pretrain_command(config, method).command)
            finetune = " ".join(build_finetune_command(config, method).command)
            self.assertIn("--model yolo26n.pt", pretrain)
            self.assertIn("--view-pairing opposed", pretrain)
            self.assertIn("--freeze-backbone", finetune)
            self.assertIn("--augmentation-profile prepared", finetune)


class JointEvaluationTests(unittest.TestCase):
    def test_sahi_merging_does_not_suppress_different_classes(self):
        # Use a real lightweight object rather than a detector or checkpoint.
        class Boxes:
            xyxy = torch.tensor([[1.0, 1.0, 20.0, 20.0], [1.0, 1.0, 20.0, 20.0]])
            conf = torch.tensor([0.9, 0.8])
            cls = torch.tensor([0.0, 1.0])

            def __len__(self):
                return 2

        result = sahi_inference_batched(
            lambda images, **kwargs: [SimpleNamespace(boxes=Boxes()) for _ in images],
            np.zeros((32, 32, 3), np.uint8),
            (32, 32),
            (0, 0),
            1,
            half=False,
        )
        self.assertEqual(result[2], [0, 1])
        with self.assertRaises(ValueError):
            slice_image(np.zeros((32, 32, 3), np.uint8), (32, 32), (32, 0))

    def test_calibration_then_common_evaluation_with_three_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, labels = root / "images", root / "labels"
            images.mkdir()
            labels.mkdir()
            for name, shade in (("one", 20), ("two", 220)):
                Image.new("RGB", (64, 64), (shade, shade, shade)).save(
                    images / f"{name}.png"
                )
                (labels / f"{name}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
            weights = {}
            for name in ("a", "b", "c"):
                weights[name] = root / f"{name}.pt"
                weights[name].write_bytes(name.encode())
            samples = find_samples(images, labels, expected=2)
            settings = {"confidence_floor": 0.001, "match_iou": 0.5}

            def fake_predict(model, image, _settings):
                if model.name != "a" and image.mean() > 100:
                    return []
                return [
                    (0, {"a": 0.9, "b": 0.8, "c": 0.7}[model.name], [16, 16, 48, 48])
                ]

            with (
                patch(
                    "ultralytics.YOLO",
                    side_effect=lambda path: SimpleNamespace(name=Path(path).stem),
                ),
                patch(
                    "ssl_experiments.evaluation.thesis.predict",
                    side_effect=fake_predict,
                ),
            ):
                calibration = calibrate(weights, samples, settings, root)
                rows = objectness(
                    weights, samples, settings, calibration, root / "behavior", [0.5], 0
                )
            self.assertEqual({r["common_targets"] for r in rows}, {1})
            self.assertEqual({r["threshold"] for r in rows if r["model"] == "b"}, {0.8})
            self.assertFalse(any("edge" in r["key"] for r in rows))
            self.assertTrue(
                all(
                    r.get("diagnostic_only")
                    for r in rows
                    if r["key"] == "object_removed"
                )
            )
            self.assertTrue((root / "behavior/common_targets.json").is_file())


if __name__ == "__main__":
    unittest.main()
