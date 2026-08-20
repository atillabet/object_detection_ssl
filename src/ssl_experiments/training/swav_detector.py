import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

OFFICIAL_SWAV_URL = "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar"
DEFAULT_MODEL = Path(__file__).parents[1] / "models/swav_resnet50_yolo11.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a YOLO detector with a ResNet backbone initialized from a SwAV checkpoint."
    )
    parser.add_argument(
        "--data",
        default="/media/datasets/object_detection/cur_train_3_natural_disasters/data.yaml",
        help="Path to YOLO-format data.yaml.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Model yaml or checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--swav-ckpt",
        default=None,
        help="Path to a SwAV-pretrained ResNet checkpoint. If omitted, torchvision DEFAULT weights are used.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--project", default="artifacts/baselines")
    parser.add_argument("--name", default="swav_resnet50_yolo11")
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
    dataset = mlflow.data.meta_dataset(
        source=FileSystemDatasetSource(uri=str(data_yaml)),
        name=data_yaml.parent.name,
    )
    mlflow.log_input(
        dataset,
        context="training",
        tags={"description": "YOLO fine-tuning from a SwAV-initialized ResNet backbone"},
    )


def unwrap_checkpoint(checkpoint: dict) -> tuple[dict, str]:
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"], "nested_state_dict"
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"], "nested_model"
    return checkpoint, "flat_state_dict"


def extract_backbone_state_dict(checkpoint: dict) -> tuple[dict[str, torch.Tensor], str]:
    state, source_format = unwrap_checkpoint(checkpoint)
    extracted = {}
    prefixes = (
        "module.",
        "backbone.",
        "encoder.",
        "encoder_q.",
        "model.",
    )

    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue

        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True

        if new_key.startswith(("projection_head", "prototypes", "fc.")):
            continue
        extracted[new_key] = value

    return extracted, source_format


def remap_resnet_keys_for_ultralytics(backbone_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    stage_map = {
        "conv1": "0",
        "bn1": "1",
        "layer1": "4",
        "layer2": "5",
        "layer3": "6",
        "layer4": "7",
    }
    remapped = {}

    for key, value in backbone_state.items():
        root = key.split(".", 1)[0]
        target_root = stage_map.get(root)
        if target_root is None:
            continue
        suffix = key[len(root) :]
        remapped[f"{target_root}{suffix}"] = value

    return remapped


def load_swav_backbone(model: YOLO, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    backbone_state, source_format = extract_backbone_state_dict(checkpoint)
    remapped = remap_resnet_keys_for_ultralytics(backbone_state)

    backbone = model.model.model[0]
    missing, unexpected = backbone.m.load_state_dict(remapped, strict=False)

    loaded_tensors = len(remapped) - len(unexpected)
    if loaded_tensors < 100:
        raise RuntimeError(
            f"Loaded only {loaded_tensors} tensors from {checkpoint_path}. "
            "This does not look like an official or compatible SwAV ResNet50 checkpoint."
        )

    print(f"Loaded {loaded_tensors} backbone tensors from {checkpoint_path} ({source_format})")
    print(f"Expected official SwAV layout: flat torchvision-style state_dict from {OFFICIAL_SWAV_URL}")
    if missing:
        print(f"Missing backbone keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected checkpoint keys after remap: {len(unexpected)}")


def main() -> None:
    args = parse_args()
    configure_mlflow(args)

    model = YOLO(args.model)
    if args.swav_ckpt:
        load_swav_backbone(model, args.swav_ckpt)
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
        lr0=3e-4,
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
        mosaic=0.2,
        close_mosaic=10,
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
