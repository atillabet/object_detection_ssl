from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from helpers import write_report, write_results, write_study
from ssl_experiments.commands import finetune_results, shape_evaluation_dir
from ssl_experiments.matrix import (
    analyze_matrix,
    confidence_interval,
    load_matrix,
    render_matrix,
)


def write_matrix(root: Path, base: Path) -> Path:
    raw = {
        "schema_version": 1,
        "name": "low_label",
        "base_study": str(base),
        "axes": {"seed": [0, 1], "fraction": [0.01, 0.05]},
        "cell_name": "{matrix}_seed{seed}_{fraction_slug}",
        "pretraining_group": "seed_{seed}",
        "overrides": {
            "pretraining.args.seed": "{seed}",
            "finetuning.args.seed": "{seed}",
        },
        "aggregate_by": ["fraction"],
        "baseline": "supervised",
    }
    path = root / "matrix.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


class MatrixTests(unittest.TestCase):
    def test_rendering_shares_pretraining_within_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = write_study(root)
            matrix = load_matrix(write_matrix(root, base))
            cells = render_matrix(matrix)

            seed_zero = [cell for cell in cells if cell.axes["seed"] == 0]
            seed_one = [cell for cell in cells if cell.axes["seed"] == 1]

        self.assertEqual(len(cells), 4)
        self.assertEqual(seed_zero[0].config.pretraining_root, seed_zero[1].config.pretraining_root)
        self.assertNotEqual(seed_zero[0].config.pretraining_root, seed_one[0].config.pretraining_root)
        self.assertNotEqual(seed_zero[0].config.artifact_root, seed_zero[1].config.artifact_root)

    def test_rendering_rebases_evaluation_paths_from_base_study(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = write_study(root)
            matrix = load_matrix(write_matrix(root, base))
            cells = render_matrix(matrix)

        for cell in cells:
            shape = cell.config.evaluations["shape_reliance"]
            self.assertEqual(Path(shape["images"]), root / "images")
            self.assertEqual(Path(shape["labels"]), root / "labels")

    def test_shared_pretraining_rejects_command_differences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = write_study(root)
            matrix_path = write_matrix(root, base)
            raw = yaml.safe_load(matrix_path.read_text())
            raw["pretraining_group"] = "one_group"
            raw["overrides"]["pretraining.args.seed"] = "{seed}"
            matrix_path.write_text(yaml.safe_dump(raw, sort_keys=False))
            matrix = load_matrix(matrix_path)

            with self.assertRaisesRegex(ValueError, "different pretraining commands"):
                render_matrix(matrix)

    def test_matrix_analysis_aggregates_seeds_and_paired_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = write_study(root)
            matrix_path = write_matrix(root, base)
            raw = yaml.safe_load(matrix_path.read_text())
            raw["axes"]["fraction"] = [0.01]
            matrix_path.write_text(yaml.safe_dump(raw, sort_keys=False))
            matrix = load_matrix(matrix_path)
            cells = render_matrix(matrix)
            for cell in cells:
                seed = int(cell.axes["seed"])
                for method in cell.config.methods:
                    bonus = 0.02 if method.name == "region" else 0.0
                    score = 0.10 + 0.01 * seed + bonus
                    write_results(
                        finetune_results(cell.config, method),
                        [(1, score * 2, score)],
                    )
                    write_report(
                        shape_evaluation_dir(cell.config, method) / "report.json",
                        clean=score,
                        shape=0.2 + bonus,
                        color=0.1,
                        texture=0.1,
                    )

            result = analyze_matrix(matrix)

        region = next(row for row in result["aggregates"] if row["method"] == "region")
        contrast = next(
            row
            for row in result["paired_contrasts"]
            if row["method"] == "region" and row["metric"] == "clean_map50_95"
        )
        self.assertEqual(region["n"], 2)
        self.assertAlmostEqual(region["clean_map50_95_mean"], 0.125)
        self.assertAlmostEqual(contrast["mean_delta"], 0.02)

    def test_confidence_interval_uses_student_t(self) -> None:
        mean, standard_deviation, lower, upper = confidence_interval([1.0, 2.0, 3.0])
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(standard_deviation, 1.0)
        self.assertLess(lower, 0.0)
        self.assertGreater(upper, 4.0)


if __name__ == "__main__":
    unittest.main()
