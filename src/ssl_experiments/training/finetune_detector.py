#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch import nn
from ultralytics import YOLO

from ssl_experiments.methods.vicreg import load_ssl_backbone

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO detector with CLI-configurable hyperparameters."
    )
    parser.add_argument(
        "--data",
        default="/media/datasets/object_detection/cur_train_3_natural_disasters/data.yaml",
        help="Path to YOLO-format data.yaml.",
    )
    parser.add_argument(
        "--model",
        default="yolo26n",
        help="YOLO model checkpoint, yaml, or weights path.",
    )
    parser.add_argument(
        "--ssl-checkpoint",
        default=None,
        help=(
            "Object-detector VICReg checkpoint whose backbone is loaded before "
            "supervised detection training. Use the matching model yaml."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume an interrupted Ultralytics run from its last.pt checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", default="1")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the training dataset to use.",
    )
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--project", default="artifacts/finetuning")
    parser.add_argument(
        "--name",
        default=None,
        help="Optional run name. Defaults to Ultralytics auto-naming.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument(
        "--freeze", type=int, default=None, help="How many early layers to freeze."
    )
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-bias-lr", type=float, default=0.01)
    parser.add_argument("--warmup-momentum", type=float, default=0.8)
    parser.add_argument("--cls", type=float, default=1.0)
    parser.add_argument("--dfl", type=float, default=2.0)
    parser.add_argument("--box", type=float, default=7.5)
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--degrees", type=float, default=5.0)
    parser.add_argument("--scale", type=float, default=0.3)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--erasing", type=float, default=0.1)
    parser.add_argument("--hsv-h", type=float, default=0.02)
    parser.add_argument("--hsv-s", type=float, default=0.5)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--multi-scale", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--log-dataset", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument(
        "--mlflow-experiment",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "ssl_experiments"),
    )
    return parser.parse_args()


def configure_mlflow(args: argparse.Namespace) -> None:
    if args.mlflow_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri
    if args.mlflow_experiment:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = args.mlflow_experiment


def log_dataset(trainer) -> None:
    import mlflow
    import mlflow.data
    from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource

    if not mlflow.active_run():
        return

    data_yaml = Path(trainer.args.data)
    mlflow.log_artifact(str(data_yaml), artifact_path="dataset")
    dataset = mlflow.data.meta_dataset(
        source=FileSystemDatasetSource(uri=str(data_yaml)),
        name=data_yaml.parent.name,
    )
    mlflow.log_input(
        dataset,
        context="training",
        tags={"description": "YOLO fine-tuning run"},
    )


def make_ssl_transfer_verifier(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = checkpoint["backbone_state_dict"]
    backbone_layers = int(checkpoint["backbone_layers"])

    def verify(trainer) -> None:
        target = nn.Sequential(
            *list(trainer.model.model.children())[:backbone_layers]
        )
        target_state = target.state_dict()
        mismatches = [
            key
            for key, value in source_state.items()
            if key not in target_state
            or target_state[key].shape != value.shape
            or not torch.equal(
                target_state[key].detach().cpu(),
                value.detach().cpu(),
            )
        ]
        if mismatches:
            raise RuntimeError(
                "SSL backbone was not preserved during trainer construction: "
                f"{len(mismatches)} mismatched tensors"
            )
        print(
            f"Verified {len(source_state)} SSL backbone tensors after detector "
            "class-count initialization"
        )

    return verify


def main() -> None:
    args = parse_args()
    configure_mlflow(args)

    if args.resume and args.ssl_checkpoint:
        raise ValueError("--resume and --ssl-checkpoint cannot be used together")

    model = YOLO(str(args.resume.resolve()) if args.resume else args.model)
    if args.ssl_checkpoint and not args.resume:
        report = load_ssl_backbone(model, args.ssl_checkpoint, strict=True)
        model.ckpt = {"ssl_checkpoint": str(Path(args.ssl_checkpoint).resolve())}
        model.add_callback(
            "on_pretrain_routine_end",
            make_ssl_transfer_verifier(args.ssl_checkpoint),
        )
        print(
            f"Loaded {report['loaded_tensors']} SSL backbone tensors across "
            f"{report['backbone_layers']} layers from {args.ssl_checkpoint} "
            f"(epoch={report['checkpoint_epoch']}, sampling={report['sampling']})"
        )
    if args.log_dataset:
        model.add_callback("on_pretrain_routine_end", log_dataset)

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "cos_lr": args.cos_lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "warmup_bias_lr": args.warmup_bias_lr,
        "warmup_momentum": args.warmup_momentum,
        "cls": args.cls,
        "dfl": args.dfl,
        "box": args.box,
        "multi_scale": args.multi_scale,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        "degrees": args.degrees,
        "scale": args.scale,
        "translate": args.translate,
        "erasing": args.erasing,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "patience": args.patience,
        "seed": args.seed,
        "workers": args.workers,
        "fraction": args.fraction,
        "val": args.val,
        "project": args.project,
    }
    if args.name:
        train_kwargs["name"] = args.name
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze
    if args.resume:
        train_kwargs["resume"] = True

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
