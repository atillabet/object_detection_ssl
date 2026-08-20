from __future__ import annotations

import unittest

import torch

from ssl_experiments.training.swav_detector import (
    DEFAULT_MODEL,
    extract_backbone_state_dict,
    remap_resnet_keys_for_ultralytics,
)


class SwAVTransferTests(unittest.TestCase):
    def test_extracts_prefixed_backbone_and_ignores_projection_head(self) -> None:
        checkpoint = {
            "state_dict": {
                "module.backbone.conv1.weight": torch.ones(2, 2),
                "module.backbone.layer2.0.conv1.weight": torch.ones(3, 3),
                "module.projection_head.0.weight": torch.ones(1),
                "metadata": "ignored",
            }
        }

        state, source_format = extract_backbone_state_dict(checkpoint)
        remapped = remap_resnet_keys_for_ultralytics(state)

        self.assertEqual(source_format, "nested_state_dict")
        self.assertEqual(set(state), {"conv1.weight", "layer2.0.conv1.weight"})
        self.assertEqual(set(remapped), {"0.weight", "5.0.conv1.weight"})

    def test_packaged_model_definition_exists(self) -> None:
        self.assertTrue(DEFAULT_MODEL.is_file())


if __name__ == "__main__":
    unittest.main()
