from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from helpers import write_study
from ssl_experiments.config import ConfigError, deep_merge, load_study


class ConfigTests(unittest.TestCase):
    def test_loads_valid_study_and_resolves_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_study(write_study(root))

            self.assertEqual(config.name, "test_study")
            self.assertEqual([method.name for method in config.methods], ["supervised", "region"])
            self.assertEqual(config.artifact_root, root / "artifacts/test_study")
            self.assertEqual(config.method("region").kind, "vicreg_shape_region")

    def test_rejects_duplicate_method_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_study(root)
            raw = yaml.safe_load(path.read_text())
            raw["methods"].append({"name": "region", "kind": "vicreg_global"})
            path.write_text(yaml.safe_dump(raw))

            with self.assertRaisesRegex(ConfigError, "Duplicate method"):
                load_study(path)

    def test_deep_merge_preserves_unmodified_nested_values(self) -> None:
        merged = deep_merge(
            {"args": {"epochs": 100, "batch": 128}, "script": "a.py"},
            {"args": {"batch": 64}},
        )
        self.assertEqual(
            merged,
            {"args": {"epochs": 100, "batch": 64}, "script": "a.py"},
        )


if __name__ == "__main__":
    unittest.main()
