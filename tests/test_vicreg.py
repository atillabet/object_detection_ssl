from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from ssl_experiments.methods.vicreg import (
    AlignedFeatureConsistencyLoss,
    MultiOutputVICRegModel,
    ObjectCentricDataset,
    SpatialVICRegLoss,
    VICRegLoss,
    VICRegModel,
    aligned_geometry_views,
    contour_spatial_weights,
    create_yolo_backbone,
    create_yolo_trunk,
    load_ssl_backbone,
    load_yolo_boxes,
)


class ObjectCentricDatasetTests(unittest.TestCase):
    def test_load_yolo_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label_path = Path(directory) / "sample.txt"
            label_path.write_text("2 0.5 0.4 0.2 0.3\n")
            boxes = load_yolo_boxes(label_path)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].class_id, 2)
        self.assertAlmostEqual(boxes[0].height, 0.3)

    def test_object_dataset_returns_two_tensor_views(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            pixels = np.zeros((96, 128, 3), dtype=np.uint8)
            pixels[24:72, 44:84] = (230, 80, 30)
            Image.fromarray(pixels).save(images / "sample.jpg")
            (labels / "sample.txt").write_text("0 0.5 0.5 0.3125 0.5\n")

            dataset = ObjectCentricDataset(
                images,
                labels=labels,
                sampling="object",
                image_size=64,
            )
            first, second = dataset[0]

        self.assertEqual(tuple(first.shape), (3, 64, 64))
        self.assertEqual(tuple(second.shape), (3, 64, 64))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.isfinite(second).all())

    def test_object_sampling_retains_negatives_with_global_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            Image.new("RGB", (64, 64)).save(images / "positive.png")
            Image.new("RGB", (64, 64)).save(images / "negative.png")
            (labels / "positive.txt").write_text("0 0.5 0.5 0.5 0.5\n")

            dataset = ObjectCentricDataset(
                images,
                labels=labels,
                sampling="object",
                image_size=64,
            )

        self.assertEqual(len(dataset), 2)

    def test_aligned_geometry_is_identical_for_both_views(self) -> None:
        pixels = np.arange(80 * 120 * 3, dtype=np.uint8).reshape(80, 120, 3)
        image = Image.fromarray(pixels)
        first, second = aligned_geometry_views(image, 64, random_crop=True)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

    def test_shape_view_is_aligned_grayscale_edge_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            images = Path(directory) / "images"
            images.mkdir()
            pixels = np.zeros((96, 128, 3), dtype=np.uint8)
            pixels[20:76, 36:92] = (220, 60, 30)
            Image.fromarray(pixels).save(images / "sample.png")
            dataset = ObjectCentricDataset(
                images,
                sampling="global",
                image_size=64,
                view_pairing="aligned",
                shape_view_probability=1.0,
            )
            first, second = dataset[0]

        self.assertEqual(tuple(first.shape), (3, 64, 64))
        self.assertEqual(tuple(second.shape), (3, 64, 64))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.isfinite(second).all())
        self.assertTrue(torch.equal(second[0], second[1]))
        self.assertTrue(torch.equal(second[1], second[2]))

    def test_auxiliary_shape_view_is_aligned_with_second_rgb_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            images = Path(directory) / "images"
            images.mkdir()
            pixels = np.zeros((96, 128, 3), dtype=np.uint8)
            pixels[20:76, 36:92] = (220, 60, 30)
            Image.fromarray(pixels).save(images / "sample.png")
            dataset = ObjectCentricDataset(
                images,
                sampling="global",
                image_size=64,
                auxiliary_shape_view=True,
            )
            dataset.appearance_transform = lambda image: torch.from_numpy(
                np.asarray(image).copy()
            ).permute(2, 0, 1)
            dataset.shape_transform = dataset.appearance_transform
            first, second, shape = dataset[0]

        self.assertEqual(tuple(first.shape), (3, 64, 64))
        self.assertTrue(torch.equal(second, shape))

    def test_shape_view_requires_aligned_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            images = Path(directory) / "images"
            images.mkdir()
            Image.new("RGB", (64, 64)).save(images / "sample.png")
            with self.assertRaisesRegex(ValueError, "require aligned"):
                ObjectCentricDataset(
                    images,
                    sampling="global",
                    shape_view_probability=0.5,
                )

    def test_image_manifest_controls_dataset_membership_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            for name in ("a.png", "b.png", "c.png"):
                Image.new("RGB", (64, 64)).save(images / name)
            manifest = root / "images.txt"
            manifest.write_text("b.png\na.png\n")

            dataset = ObjectCentricDataset(
                images,
                image_list=manifest,
                sampling="global",
            )

        self.assertEqual([path.name for path in dataset.image_paths], ["b.png", "a.png"])

    def test_image_manifest_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (64, 64)).save(images / "sample.png")
            manifest = root / "images.txt"
            manifest.write_text("sample.png\nsample.png\n")

            with self.assertRaisesRegex(ValueError, "duplicate"):
                ObjectCentricDataset(
                    images,
                    image_list=manifest,
                    sampling="global",
                )

    def test_image_manifest_preserves_dataset_symlinks_for_label_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "dataset/images"
            labels = root / "dataset/labels"
            backing = root / "backing"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            backing.mkdir()
            Image.new("RGB", (64, 64)).save(backing / "sample.png")
            (images / "sample.png").symlink_to(backing / "sample.png")
            (labels / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n")
            manifest = root / "images.txt"
            manifest.write_text(f"{images / 'sample.png'}\n")

            dataset = ObjectCentricDataset(
                images,
                image_list=manifest,
                labels=labels,
                sampling="object",
                image_size=64,
            )
            first, second = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(tuple(first.shape), (3, 64, 64))
        self.assertEqual(tuple(second.shape), (3, 64, 64))

    def test_image_manifest_rejects_lexical_parent_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (64, 64)).save(root / "outside.png")
            manifest = root / "images.txt"
            manifest.write_text("../outside.png\n")

            with self.assertRaisesRegex(ValueError, "outside"):
                ObjectCentricDataset(
                    images,
                    image_list=manifest,
                    sampling="global",
                )


class VICRegTests(unittest.TestCase):
    def test_loss_is_finite_and_backpropagates(self) -> None:
        first = torch.randn(8, 16, requires_grad=True)
        second = torch.randn(8, 16, requires_grad=True)
        losses = VICRegLoss()(first, second)
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertIsNotNone(first.grad)
        self.assertIsNotNone(second.grad)

    def test_spatial_loss_is_finite_and_backpropagates(self) -> None:
        model = VICRegModel(
            torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
            8,
            hidden_dim=16,
            projection_dim=16,
            spatial_levels=(1, 2, 4),
        )
        first = model.forward_spatial(torch.randn(4, 3, 32, 32))
        second = model.forward_spatial(torch.randn(4, 3, 32, 32))
        self.assertEqual(tuple(first[1].shape), (4, 16))
        self.assertEqual(tuple(first[2].shape), (16, 16))
        self.assertEqual(tuple(first[4].shape), (64, 16))

        losses = SpatialVICRegLoss()(first, second)
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertIn("level_4_loss", losses)
        self.assertIsNotNone(model.backbone.weight.grad)

    def test_aligned_feature_loss_is_finite_and_backpropagates(self) -> None:
        appearance = {
            2: torch.randn(16, 8, requires_grad=True),
            4: torch.randn(64, 8, requires_grad=True),
        }
        shape = {
            2: torch.randn(16, 8, requires_grad=True),
            4: torch.randn(64, 8, requires_grad=True),
        }
        loss = AlignedFeatureConsistencyLoss()(appearance, shape)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(appearance[4].grad)
        self.assertIsNotNone(shape[4].grad)

    def test_contour_weights_select_expected_fraction_and_ignore_inversion(self) -> None:
        shape = torch.zeros(2, 3, 32, 32)
        shape[0, :, 8:24, 15:17] = 1.0
        shape[1] = 1.0 - shape[0]

        weights = contour_spatial_weights(shape, (2, 4), region_fraction=0.25)

        self.assertEqual(tuple(weights[2].shape), (8,))
        self.assertEqual(tuple(weights[4].shape), (32,))
        self.assertTrue(torch.equal(weights[2][:4], weights[2][4:]))
        self.assertTrue(torch.equal(weights[4][:16], weights[4][16:]))
        self.assertEqual(int((weights[4][:16] > 0).sum()), 4)
        self.assertAlmostEqual(float(weights[4].mean()), 1.0)

    def test_aligned_feature_loss_accepts_spatial_weights(self) -> None:
        appearance = {2: torch.tensor([[1.0, 0.0], [1.0, 0.0]])}
        shape = {2: torch.tensor([[1.0, 0.0], [-1.0, 0.0]])}
        weights = {2: torch.tensor([1.0, 0.0])}

        loss = AlignedFeatureConsistencyLoss()(appearance, shape, weights)

        self.assertAlmostEqual(float(loss), 0.0)

    def test_yolo_trunk_projects_all_detect_input_scales(self) -> None:
        trunk, feature_dims, trunk_layers, output_layers = create_yolo_trunk(
            "yolo26n.yaml",
            probe_size=64,
        )
        model = MultiOutputVICRegModel(
            trunk,
            feature_dims,
            hidden_dim=16,
            projection_dim=16,
        )
        projections = model.forward_spatial(torch.randn(2, 3, 64, 64))

        self.assertEqual(trunk_layers, 23)
        self.assertEqual(output_layers, (16, 19, 22))
        self.assertEqual(feature_dims, {16: 64, 19: 128, 22: 256})
        self.assertEqual(
            {layer: tuple(value.shape) for layer, value in projections.items()},
            {16: (2, 16), 19: (2, 16), 22: (2, 16)},
        )
        self.assertEqual(next(iter(trunk.state_dict())), "0.conv.weight")

    def test_trunk_checkpoint_loads_neck_into_detector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "trunk_ssl.pt"
            trunk, _feature_dims, trunk_layers, _outputs = create_yolo_trunk(
                "yolo26n.yaml",
                probe_size=64,
            )
            neck_parameter = next(trunk[22].parameters())
            with torch.no_grad():
                neck_parameter.fill_(0.25)
            torch.save(
                {
                    "method": "vicreg",
                    "epoch": 1,
                    "backbone_layers": trunk_layers,
                    "backbone_state_dict": trunk.state_dict(),
                    "config": {
                        "sampling": "global",
                        "encoder_scope": "trunk",
                    },
                },
                checkpoint_path,
            )

            detector = YOLO("yolo26n.yaml")
            report = load_ssl_backbone(detector, checkpoint_path)
            loaded_parameter = next(detector.model.model[22].parameters())

        self.assertEqual(report["backbone_layers"], 23)
        self.assertGreater(report["loaded_tensors"], 240)
        self.assertTrue(
            torch.allclose(loaded_parameter, torch.full_like(loaded_parameter, 0.25))
        )

    def test_checkpoint_loads_into_matching_yolo_backbone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "ssl.pt"
            backbone, _feature_dim, backbone_layers = create_yolo_backbone(
                "yolo26n.yaml",
                probe_size=64,
            )
            first_parameter = next(backbone.parameters())
            with torch.no_grad():
                first_parameter.fill_(0.125)
            torch.save(
                {
                    "method": "vicreg",
                    "epoch": 1,
                    "backbone_layers": backbone_layers,
                    "backbone_state_dict": backbone.state_dict(),
                    "config": {"sampling": "object"},
                },
                checkpoint_path,
            )

            detector = YOLO("yolo26n.yaml")
            report = load_ssl_backbone(detector, checkpoint_path)
            loaded_parameter = next(detector.model.model[0].parameters())

        self.assertEqual(report["sampling"], "object")
        self.assertGreater(report["loaded_tensors"], 100)
        self.assertTrue(torch.allclose(loaded_parameter, torch.full_like(loaded_parameter, 0.125)))


if __name__ == "__main__":
    unittest.main()
