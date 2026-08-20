from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from helpers import write_study
from ssl_experiments.config import load_study
from ssl_experiments.pipeline import Pipeline


PRETRAIN_SCRIPT = r'''
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--epochs", type=int, default=1)
args, _ = parser.parse_known_args()
run = Path(args.output) / args.name
run.mkdir(parents=True, exist_ok=True)
(run / "latest.pt").write_text("checkpoint")
(run / "metrics.jsonl").write_text("".join("{}\n" for _ in range(args.epochs)))
'''

FINETUNE_SCRIPT = r'''
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--epochs", type=int, default=1)
args, _ = parser.parse_known_args()
run = Path(args.project) / args.name
(run / "weights").mkdir(parents=True, exist_ok=True)
(run / "weights/best.pt").write_text("best")
(run / "weights/last.pt").write_text("last")
rows = ["epoch,metrics/mAP50(B),metrics/mAP50-95(B)"]
rows.extend(f"{epoch},0.2,0.1" for epoch in range(1, args.epochs + 1))
(run / "results.csv").write_text("\n".join(rows) + "\n")
'''

EVALUATION_SCRIPT = r'''
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("images")
parser.add_argument("--output", required=True)
parser.add_argument("--run-name", required=True)
args, _ = parser.parse_known_args()
run = Path(args.output) / args.run_name
run.mkdir(parents=True, exist_ok=True)
(run / "report.json").write_text(json.dumps({"ok": True}))
'''


class PipelineTests(unittest.TestCase):
    def test_pipeline_runs_all_stages_and_records_resumable_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = write_study(
                root,
                methods=[{"name": "global", "kind": "vicreg_global"}],
            )
            repo = root / "project"
            (repo / "pretrain.py").write_text(PRETRAIN_SCRIPT)
            (repo / "train.py").write_text(FINETUNE_SCRIPT)
            (repo / "shape.py").write_text(EVALUATION_SCRIPT)
            config = load_study(study)
            pipeline = Pipeline(config)

            pipeline.run()
            first_status = pipeline.status()
            pipeline.run()

            self.assertEqual(
                first_status,
                [
                    {
                        "method": "global",
                        "pretrain": "complete",
                        "finetune": "complete",
                        "evaluate": "complete",
                    }
                ],
            )
            self.assertTrue((config.artifact_root / "study.json").is_file())
            self.assertTrue((config.artifact_root / "state/global/pretrain.json").is_file())
            self.assertTrue((config.artifact_root / "state/global/finetune.json").is_file())
            self.assertTrue((config.artifact_root / "state/global/shape_reliance.json").is_file())


if __name__ == "__main__":
    unittest.main()
