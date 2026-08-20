from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from helpers import write_report, write_results, write_study
from ssl_experiments.analysis import (
    analyze_study,
    benchmark_summary,
    clean_validation_summary,
    training_summary,
)
from ssl_experiments.config import load_study


class AnalysisTests(unittest.TestCase):
    def test_training_summary_uses_best_map95_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            write_results(path, [(1, 0.2, 0.1), (2, 0.3, 0.09)])
            result = training_summary(path)

        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["best_map50_95"], 0.1)
        self.assertEqual(result["final_map50_95"], 0.09)

    def test_benchmark_summary_computes_shape_appearance_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_report(path, clean=0.12, shape=0.3, color=0.04, texture=0.1)
            result = benchmark_summary(path)

        self.assertAlmostEqual(result["appearance_reliance"], 0.07)
        self.assertAlmostEqual(result["shape_minus_appearance_gap"], 0.23)

    def test_clean_validation_summary_uses_independent_validation_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clean_validation.json"
            path.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "precision": 0.7,
                                "recall": 0.6,
                                "map50": 0.5,
                                "map50_95": 0.3,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result = clean_validation_summary(path)

        self.assertEqual(result["validation_map50_95"], 0.3)
        self.assertNotIn("clean_map50_95", result)

    def test_analyze_study_writes_comparison_and_criteria(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study_path = write_study(root)
            raw = yaml.safe_load(study_path.read_text())
            raw["analysis"]["criteria"] = [
                {
                    "name": "region_beats_supervised",
                    "left": "region",
                    "right": "supervised",
                    "metric": "clean_map50_95",
                    "operator": ">=",
                    "threshold": 0.01,
                }
            ]
            study_path.write_text(yaml.safe_dump(raw, sort_keys=False))
            config = load_study(study_path)
            for method, clean in (("supervised", 0.10), ("region", 0.12)):
                write_results(
                    config.artifact_root / f"finetuning/{method}/results.csv",
                    [(1, clean * 2, clean)],
                )
                evaluation = config.artifact_root / f"evaluations/shape_reliance/{method}"
                write_report(
                    evaluation / "report.json",
                    clean=clean,
                    shape=0.25,
                    color=0.03,
                    texture=0.1,
                )
                (evaluation / "sample_images.txt").write_text("same.jpg\n")

            result = analyze_study(config)

            self.assertTrue(result["overall_success"])
            self.assertTrue((config.artifact_root / "analysis/summary.md").is_file())


if __name__ == "__main__":
    unittest.main()
