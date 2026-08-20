from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from helpers import write_report
from ssl_experiments.bootstrap import paired_bootstrap, write_bootstrap


class BootstrapTests(unittest.TestCase):
    def write_per_image(self, path: Path, offset: int) -> None:
        path.mkdir(parents=True)
        conditions = [("clean", "clean", 0.0)]
        conditions.extend(
            (f"{category}_{severity}", category, severity)
            for category in ("color", "shape", "texture")
            for severity in (0.25, 0.5, 0.75)
        )
        with (path / "per_image_metrics.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "condition", "category", "transform", "severity", "image_id",
                    "num_gt", "num_predictions", "tp", "fp", "fn", "precision",
                    "recall", "f1",
                )
            )
            for condition, category, severity in conditions:
                for image_index in range(4):
                    tp = 5 + offset if category == "clean" else max(1, 5 - int(severity * 4) + offset)
                    writer.writerow(
                        (condition, category, "test", severity, f"{image_index}.jpg", 10, 10, tp, 1, 10 - tp, 0, 0, 0)
                    )

    def test_generic_paired_bootstrap_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left, right = root / "left", root / "right"
            self.write_per_image(left, 1)
            self.write_per_image(right, 0)
            write_report(left / "report.json", clean=0.12, shape=0.3, color=0.05, texture=0.1)
            write_report(right / "report.json", clean=0.10, shape=0.2, color=0.05, texture=0.1)

            result = paired_bootstrap(
                left, right, left_name="left", right_name="right", samples=100, seed=0
            )
            output = root / "output"
            write_bootstrap(output, result)

            self.assertEqual(result["contrast"], "left_minus_right")
            self.assertIn("shape_minus_appearance_gap", result["metrics"])
            self.assertTrue((output / "bootstrap.json").is_file())
            self.assertTrue((output / "bootstrap.md").is_file())


if __name__ == "__main__":
    unittest.main()
