from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from helpers import write_study
from ssl_experiments.commands import (
    args_to_cli,
    build_evaluation_commands,
    build_finetune_command,
    build_pretrain_command,
    resolve_project_argument_paths,
)
from ssl_experiments.config import load_study


class CommandTests(unittest.TestCase):
    def test_object_and_mixed_methods_select_explicit_sampling(self) -> None:
        methods = [
            {"name": "object", "kind": "vicreg_object"},
            {"name": "mixed", "kind": "vicreg_mixed"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            config = load_study(write_study(Path(directory), methods=methods))
            object_command = build_pretrain_command(config, config.method("object"))
            mixed_command = build_pretrain_command(config, config.method("mixed"))
            assert object_command is not None and mixed_command is not None

        self.assertIn("--sampling object", " ".join(object_command.command))
        self.assertIn("--sampling mixed", " ".join(mixed_command.command))
        self.assertIn("--object-probability 0.75", " ".join(mixed_command.command))

    def test_region_method_injects_contour_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_study(write_study(Path(directory)))
            command = build_pretrain_command(config, config.method("region"))
            assert command is not None
            rendered = " ".join(command.command)

        self.assertIn("--shape-region-weighting contour", rendered)
        self.assertIn("--shape-pairing aligned", rendered)
        self.assertIn("--shape-spatial-levels 2 4", rendered)
        self.assertIn("--output", rendered)

    def test_pretraining_command_uses_configured_shared_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = write_study(root)
            raw = yaml.safe_load(study.read_text())
            shared = root / "shared_pretraining"
            raw["paths"]["pretraining_root"] = str(shared)
            study.write_text(yaml.safe_dump(raw, sort_keys=False))
            config = load_study(study)
            command = build_pretrain_command(config, config.method("region"))
            assert command is not None

        rendered = " ".join(command.command)
        self.assertIn(f"--output {shared}", rendered)
        self.assertEqual(command.completion, shared / "region/latest.pt")

    def test_supervised_finetune_omits_ssl_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_study(write_study(Path(directory)))
            command = build_finetune_command(config, config.method("supervised"))

        self.assertNotIn("--ssl-checkpoint", command.command)

    def test_shape_evaluation_uses_method_weights_and_fixed_run_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_study(write_study(Path(directory)))
            commands = build_evaluation_commands(config, config.method("region"))
            rendered = " ".join(commands[0].command)

        self.assertIn("--run-name region", rendered)
        self.assertIn(str(Path("weights") / "best.pt"), rendered)
        self.assertEqual(commands[0].completion.parts[-2:], ("region", "report.json"))

    def test_clean_validation_is_a_first_class_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = write_study(root)
            raw = yaml.safe_load(study.read_text())
            raw["evaluations"]["clean_validation"] = {
                "module": "ssl_experiments.evaluation.clean",
                "data": str(root / "project/data.yaml"),
                "args": {"imgsz": 320},
            }
            study.write_text(yaml.safe_dump(raw, sort_keys=False))
            config = load_study(study)
            commands = build_evaluation_commands(config, config.method("region"))
            clean = next(command for command in commands if command.name == "clean_validation")

        rendered = " ".join(clean.command)
        self.assertIn("-m ssl_experiments.evaluation.clean", rendered)
        self.assertIn("--checkpoint region=", rendered)
        self.assertTrue(str(clean.completion).endswith("clean_validation.json"))

    def test_boolean_and_list_arguments(self) -> None:
        self.assertEqual(
            args_to_cli({"amp": True, "multi_scale": False, "levels": [2, 4]}),
            ["--amp", "--no-multi-scale", "--levels", "2", "4"],
        )

    def test_training_input_paths_resolve_from_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_study(write_study(Path(directory)))
            args = resolve_project_argument_paths(
                config,
                {"data": "data.yaml", "model": "yolo26n.yaml"},
                ("data",),
            )

        self.assertEqual(Path(args["data"]).resolve(), (config.project_root / "data.yaml").resolve())
        self.assertEqual(args["model"], "yolo26n.yaml")


if __name__ == "__main__":
    unittest.main()
