import argparse
import os
from pathlib import Path

from ultralytics import RTDETR

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune RT-DETR as an alternative detector baseline."
    )
    parser.add_argument(
        "--data",
        default="/media/datasets/object_detection/cur_train_3_natural_disasters/data.yaml",
        help="Path to YOLO-format data.yaml.",
    )
    parser.add_argument(
        "--model",
        default="rtdetr-l.pt",
        help="RT-DETR checkpoint or model yaml to fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="artifacts/baselines")
    parser.add_argument("--name", default="rtdetr_baseline")
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="How many early layers to freeze. Leave at 0 unless you know the layer layout you want.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-dataset",
        action=argparse.BooleanOptionalAction,
        default=False,
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

    dataset_name = data_yaml.parent.name
    source = FileSystemDatasetSource(uri=str(data_yaml))
    dataset = mlflow.data.meta_dataset(source=source, name=dataset_name)
    mlflow.log_input(
        dataset,
        context="training",
        tags={"description": "RT-DETR alternative detector baseline"},
    )


def main() -> None:
    args = parse_args()
    configure_mlflow(args)

    model = RTDETR(args.model)
    if args.log_dataset:
        model.add_callback("on_pretrain_routine_end", log_dataset)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        freeze=args.freeze,
        optimizer="AdamW",
        lr0=1e-4,
        lrf=1e-2,
        cos_lr=True,
        momentum=0.9,
        weight_decay=1e-4,
        warmup_epochs=3,
        warmup_bias_lr=0.0,
        warmup_momentum=0.8,
        patience=30,
        seed=args.seed,
        workers=args.workers,
        multi_scale=False,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=5.0,
        scale=0.2,
        translate=0.1,
        erasing=0.0,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
