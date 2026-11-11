#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import track
from torch.utils.data import DataLoader

from ssl_experiments.methods.vicreg import (
    AlignedFeatureConsistencyLoss,
    MultiOutputVICRegModel,
    ObjectCentricDataset,
    SpatialVICRegLoss,
    VICRegModel,
    contour_spatial_weights,
    create_yolo_backbone,
    create_yolo_trunk,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain a YOLO backbone with object-centric VICReg."
    )
    parser.add_argument(
        "--images",
        default="/media/datasets/object_detection/cur_train_3_natural_disasters/train/images",
    )
    parser.add_argument(
        "--image-list",
        default=None,
        help="Optional newline-delimited image manifest rooted at --images.",
    )
    parser.add_argument(
        "--labels",
        default="/media/datasets/object_detection/cur_train_3_natural_disasters/train/labels",
    )
    parser.add_argument("--model", default="yolo26n.yaml")
    parser.add_argument(
        "--encoder-scope",
        choices=("backbone", "trunk"),
        default="backbone",
        help="Pretrain only the backbone or the backbone-plus-neck trunk.",
    )
    parser.add_argument(
        "--sampling",
        choices=("global", "object", "mixed"),
        default="mixed",
    )
    parser.add_argument("--object-probability", type=float, default=0.75)
    parser.add_argument("--context-min", type=float, default=1.25)
    parser.add_argument("--context-max", type=float, default=2.5)
    parser.add_argument("--center-jitter", type=float, default=0.1)
    parser.add_argument(
        "--view-pairing",
        choices=("independent", "aligned", "opposed"),
        default="independent",
    )
    parser.add_argument("--shape-view-probability", type=float, default=0.0)
    parser.add_argument(
        "--auxiliary-shape-probability",
        type=float,
        default=0.0,
        help="Fraction of RGB pairs that also receive an aligned contour view.",
    )
    parser.add_argument(
        "--shape-loss-weight",
        type=float,
        default=0.0,
        help="Weight of aligned RGB-to-contour backbone consistency.",
    )
    parser.add_argument(
        "--shape-spatial-levels",
        type=int,
        nargs="+",
        default=[2, 4],
        help="Pooled backbone grids used by the auxiliary contour loss.",
    )
    parser.add_argument(
        "--shape-region-weighting",
        choices=("uniform", "contour"),
        default="uniform",
        help="Uniform cells or label-free high-contour cells for the shape loss.",
    )
    parser.add_argument(
        "--shape-region-fraction",
        type=float,
        default=0.25,
        help="Fraction of high-contour cells retained at each spatial level.",
    )
    parser.add_argument(
        "--shape-pairing",
        choices=("aligned", "mismatched"),
        default="aligned",
        help="Use the matching contour or a within-batch mismatched control.",
    )
    parser.add_argument(
        "--spatial-levels",
        type=int,
        nargs="+",
        default=[1],
        help="Aligned feature-grid levels; level 1 preserves global VICReg.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="0")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--projection-dim", type=int, default=1024)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Optional cap for smoke tests; omit for full epochs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help="BF16 is the safer default for VICReg covariance gradients.",
    )
    parser.add_argument("--output", default="artifacts/pretraining")
    parser.add_argument("--name", default="yolo26n_vicreg_mixed")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume an interrupted run from its latest.pt checkpoint.",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value.isdigit():
        value = f"cuda:{value}"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def validate_args(args: argparse.Namespace) -> None:
    if args.view_pairing == "opposed" and (
        args.shape_loss_weight != 0 or args.shape_view_probability != 0
        or args.auxiliary_shape_probability != 0 or args.spatial_levels != [1]
    ):
        raise ValueError("Opposed views require ordinary global VICReg without auxiliary losses")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.batch < 2:
        raise ValueError("--batch must be at least 2 for VICReg")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("Optimizer parameters are invalid")
    if args.context_min < 1 or args.context_max < args.context_min:
        raise ValueError("Context scale must be ordered and start at 1 or greater")
    if not 0.0 <= args.shape_view_probability <= 1.0:
        raise ValueError("--shape-view-probability must be in [0, 1]")
    if args.shape_view_probability > 0.0 and args.view_pairing != "aligned":
        raise ValueError("Shape-dominant views require --view-pairing aligned")
    if not 0.0 <= args.auxiliary_shape_probability <= 1.0:
        raise ValueError("--auxiliary-shape-probability must be in [0, 1]")
    if args.shape_loss_weight < 0.0:
        raise ValueError("--shape-loss-weight must be non-negative")
    auxiliary_shape_enabled = args.auxiliary_shape_probability > 0.0
    if auxiliary_shape_enabled != (args.shape_loss_weight > 0.0):
        raise ValueError(
            "--auxiliary-shape-probability and --shape-loss-weight must both be positive"
        )
    if auxiliary_shape_enabled and args.shape_view_probability > 0.0:
        raise ValueError("Auxiliary contours cannot replace an RGB VICReg view")
    if auxiliary_shape_enabled and args.encoder_scope != "backbone":
        raise ValueError("Auxiliary contour consistency currently requires backbone scope")
    if not args.shape_spatial_levels or any(
        level <= 1 for level in args.shape_spatial_levels
    ):
        raise ValueError("--shape-spatial-levels must contain levels greater than 1")
    if len(set(args.shape_spatial_levels)) != len(args.shape_spatial_levels):
        raise ValueError("--shape-spatial-levels must not contain duplicates")
    if not 0.0 < args.shape_region_fraction <= 1.0:
        raise ValueError("--shape-region-fraction must be in (0, 1]")
    if not auxiliary_shape_enabled and (
        args.shape_region_weighting != "uniform" or args.shape_pairing != "aligned"
    ):
        raise ValueError("Shape region and pairing options require the auxiliary loss")
    if not args.spatial_levels or 1 not in args.spatial_levels:
        raise ValueError("--spatial-levels must include 1")
    if any(level < 1 for level in args.spatial_levels):
        raise ValueError("--spatial-levels must be positive")
    if len(set(args.spatial_levels)) != len(args.spatial_levels):
        raise ValueError("--spatial-levels must not contain duplicates")
    if len(args.spatial_levels) > 1 and args.view_pairing != "aligned":
        raise ValueError("Spatial VICReg levels require --view-pairing aligned")
    if args.encoder_scope == "trunk" and args.spatial_levels != [1]:
        raise ValueError("Trunk VICReg uses its three neck scales; set --spatial-levels 1")
    if args.steps_per_epoch is not None and args.steps_per_epoch < 1:
        raise ValueError("--steps-per-epoch must be positive")
    if args.image_list is not None and args.max_images is not None:
        raise ValueError("--image-list and --max-images are mutually exclusive")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    path: Path,
    *,
    model: VICRegModel | MultiOutputVICRegModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    data_generator: torch.Generator,
    epoch: int,
    global_step: int,
    optimizer_step: int,
    backbone_layers: int,
    config: dict,
    metrics: dict[str, float],
) -> None:
    checkpoint = {
        "schema_version": 2,
        "method": "vicreg",
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "backbone_layers": backbone_layers,
        "backbone_state_dict": model.backbone.state_dict(),
        "projector_state_dict": model.projector.state_dict(),
        "spatial_projectors_state_dict": model.spatial_projectors.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "data_generator_state": data_generator.get_state(),
        "config": config,
        "metrics": metrics,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def resume_training(
    path: Path,
    *,
    args: argparse.Namespace,
    model: VICRegModel | MultiOutputVICRegModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    data_generator: torch.Generator,
) -> tuple[int, int, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 2 or checkpoint.get("method") != "vicreg":
        raise ValueError("Resume requires a schema-version 2 VICReg checkpoint")

    saved_config = checkpoint.get("config", {})
    compatibility_fields = (
        "model", "encoder_scope", "sampling", "object_probability", "context_min",
        "context_max", "center_jitter", "view_pairing",
        "shape_view_probability", "auxiliary_shape_probability",
        "shape_loss_weight", "shape_spatial_levels", "spatial_levels",
        "epochs", "batch",
        "imgsz", "lr", "weight_decay", "hidden_dim", "projection_dim",
        "seed", "amp", "amp_dtype",
    )
    compatibility_defaults = {
        "encoder_scope": "backbone",
        "view_pairing": "independent",
        "shape_view_probability": 0.0,
        "auxiliary_shape_probability": 0.0,
        "shape_loss_weight": 0.0,
        "shape_spatial_levels": [2, 4],
        "spatial_levels": [1],
    }
    mismatches = [
        field
        for field in compatibility_fields
        if saved_config.get(field, compatibility_defaults.get(field))
        != getattr(args, field)
    ]
    if mismatches:
        raise ValueError(
            f"Resume arguments differ from checkpoint: {', '.join(mismatches)}"
        )

    model.backbone.load_state_dict(checkpoint["backbone_state_dict"], strict=True)
    model.projector.load_state_dict(checkpoint["projector_state_dict"], strict=True)
    spatial_state = checkpoint.get("spatial_projectors_state_dict", {})
    model.spatial_projectors.load_state_dict(spatial_state, strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    random.setstate(checkpoint["random_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint["cuda_rng_state_all"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    data_generator.set_state(checkpoint["data_generator_state"])

    completed_epoch = int(checkpoint["epoch"])
    return (
        completed_epoch + 1,
        int(checkpoint["global_step"]),
        int(checkpoint["optimizer_step"]),
    )


def main() -> None:
    args = parse_args()
    args.images = str(Path(args.images).expanduser().resolve())
    if args.image_list is not None:
        args.image_list = str(Path(args.image_list).expanduser().resolve())
    validate_args(args)
    auxiliary_shape_enabled = args.auxiliary_shape_probability > 0.0
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    amp_enabled = args.amp and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    labels = Path(args.labels) if args.sampling != "global" or args.view_pairing == "opposed" else None
    dataset = ObjectCentricDataset(
        Path(args.images),
        image_list=Path(args.image_list) if args.image_list is not None else None,
        labels=labels,
        sampling=args.sampling,
        object_probability=args.object_probability,
        image_size=args.imgsz,
        context_scale=(args.context_min, args.context_max),
        center_jitter=args.center_jitter,
        view_pairing=args.view_pairing,
        shape_view_probability=args.shape_view_probability,
        auxiliary_shape_view=auxiliary_shape_enabled,
        max_images=args.max_images,
        seed=args.seed,
    )
    if len(dataset) < args.batch:
        raise ValueError(
            f"Dataset has {len(dataset)} images but batch size is {args.batch}; "
            "VICReg drops incomplete batches"
        )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model_spatial_levels = tuple(args.spatial_levels)
    if args.encoder_scope == "backbone":
        backbone, feature_dim, backbone_layers = create_yolo_backbone(
            args.model,
            probe_size=args.imgsz,
        )
        feature_dims = None
        output_layers = None
        model: VICRegModel | MultiOutputVICRegModel = VICRegModel(
            backbone,
            feature_dim,
            hidden_dim=args.hidden_dim,
            projection_dim=args.projection_dim,
            spatial_levels=model_spatial_levels,
        )
    else:
        trunk, feature_dims, backbone_layers, output_layers = create_yolo_trunk(
            args.model,
            probe_size=args.imgsz,
        )
        feature_dim = None
        model = MultiOutputVICRegModel(
            trunk,
            feature_dims,
            hidden_dim=args.hidden_dim,
            projection_dim=args.projection_dim,
        )
    model = model.to(device)
    criterion = SpatialVICRegLoss()
    shape_criterion = AlignedFeatureConsistencyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )
    scaler_enabled = amp_enabled and autocast_dtype == torch.float16
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=scaler_enabled,
        init_scale=1024.0,
    )

    run_dir = Path(args.output) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "images": str(Path(args.images).resolve()),
            "image_list": args.image_list,
            "image_list_sha256": (
                sha256_file(Path(args.image_list)) if args.image_list is not None else None
            ),
            "labels": str(labels.resolve()) if labels is not None else None,
            "device_resolved": str(device),
            "amp_enabled": amp_enabled,
            "autocast_dtype": str(autocast_dtype) if amp_enabled else None,
            "dataset_size": len(dataset),
            "feature_dim": feature_dim,
            "feature_dims": feature_dims,
            "output_layers": output_layers,
            "backbone_layers": backbone_layers,
        }
    )
    config_path = run_dir / "config.json"
    if not args.resume:
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    console.print(
        f"Pretraining [bold]{args.model}[/bold] on {len(dataset):,} images "
        f"with sampling=[bold]{args.sampling}[/bold], "
        f"encoder_scope=[bold]{args.encoder_scope}[/bold], device={device}"
    )
    start_epoch = 1
    global_step = 0
    optimizer_step = 0
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if resume_path.parent != run_dir.resolve():
            raise ValueError(
                "--resume checkpoint must be inside the selected run directory"
            )
        start_epoch, global_step, optimizer_step = resume_training(
            resume_path,
            args=args,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            data_generator=generator,
        )
        console.print(
            f"Resuming from epoch {start_epoch} at optimizer step {optimizer_step:,}"
        )
    metrics_path = run_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "vicreg_loss": 0.0,
            "shape_loss": 0.0,
            "shape_pair_fraction": 0.0,
            "invariance": 0.0,
            "variance": 0.0,
            "covariance": 0.0,
            "embedding_std": 0.0,
        }
        steps = min(len(loader), args.steps_per_epoch or len(loader))
        iterator = track(loader, total=steps, description=f"Epoch {epoch}/{args.epochs}")
        completed_steps = 0
        for views in iterator:
            if completed_steps >= steps:
                break
            first, second, *auxiliary_views = views
            if auxiliary_shape_enabled != bool(auxiliary_views):
                raise RuntimeError("Dataset views do not match the shape objective")
            shape_mask = None
            shape_input = None
            if auxiliary_shape_enabled:
                shape_mask = (
                    torch.rand(auxiliary_views[0].shape[0])
                    < args.auxiliary_shape_probability
                )
                shape_input = auxiliary_views[0][shape_mask].to(
                    device, non_blocking=True
                )
            first = first.to(device, non_blocking=True)
            second = second.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=amp_enabled,
            ):
                if auxiliary_shape_enabled:
                    if not isinstance(model, VICRegModel):
                        raise RuntimeError("Shape consistency requires VICRegModel")
                    first_features = model.encode(first)
                    second_features = model.encode(second)
                    first_projection = {
                        level: projection.float()
                        for level, projection in model.project_features(
                            first_features
                        ).items()
                    }
                    second_projection = {
                        level: projection.float()
                        for level, projection in model.project_features(
                            second_features
                        ).items()
                    }
                else:
                    first_projection = {
                        level: projection.float()
                        for level, projection in model.forward_spatial(first).items()
                    }
                    second_projection = {
                        level: projection.float()
                        for level, projection in model.forward_spatial(second).items()
                    }
                losses = criterion(first_projection, second_projection)
                vicreg_loss = losses["loss"]
                shape_loss = vicreg_loss.new_zeros(())
                shape_pair_fraction = 0.0
                if (
                    auxiliary_shape_enabled
                    and shape_mask is not None
                    and shape_input is not None
                    and bool(shape_mask.any())
                ):
                    selected = shape_mask.to(device)
                    use_shape_pair = True
                    if args.shape_pairing == "mismatched":
                        use_shape_pair = shape_input.shape[0] > 1
                        if use_shape_pair:
                            shape_input = torch.roll(shape_input, shifts=1, dims=0)
                    if use_shape_pair:
                        shape_features = model.encode(shape_input)
                        levels = tuple(args.shape_spatial_levels)
                        appearance_spatial = model.spatial_features(
                            second_features[selected], levels
                        )
                        shape_spatial = model.spatial_features(shape_features, levels)
                        spatial_weights = (
                            contour_spatial_weights(
                                shape_input,
                                levels,
                                region_fraction=args.shape_region_fraction,
                            )
                            if args.shape_region_weighting == "contour"
                            else None
                        )
                        shape_loss = shape_criterion(
                            appearance_spatial,
                            shape_spatial,
                            spatial_weights,
                        )
                        shape_pair_fraction = float(shape_mask.float().mean())
                losses["vicreg_loss"] = vicreg_loss
                losses["shape_loss"] = shape_loss
                losses["shape_pair_fraction"] = vicreg_loss.new_tensor(
                    shape_pair_fraction
                )
                losses["loss"] = vicreg_loss + args.shape_loss_weight * shape_loss
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"Non-finite SSL loss at step {global_step}")
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_updated = not scaler_enabled or scaler.get_scale() >= scale_before

            completed_steps += 1
            global_step += 1
            optimizer_step += int(optimizer_updated)
            for key in totals:
                totals[key] += float(losses[key].detach())

        scheduler.step()
        epoch_metrics = {key: value / completed_steps for key, value in totals.items()}
        epoch_metrics.update(
            {
                "epoch": epoch,
                "steps": completed_steps,
                "global_step": global_step,
                "optimizer_step": optimizer_step,
                "skipped_optimizer_steps": global_step - optimizer_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if not all(math.isfinite(float(value)) for value in epoch_metrics.values()):
            raise FloatingPointError("Non-finite epoch metrics")
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(epoch_metrics) + "\n")
        save_checkpoint(
            run_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            data_generator=generator,
            epoch=epoch,
            global_step=global_step,
            optimizer_step=optimizer_step,
            backbone_layers=backbone_layers,
            config=config,
            metrics=epoch_metrics,
        )
        console.print(
            f"epoch={epoch} loss={epoch_metrics['loss']:.4f} "
            f"std={epoch_metrics['embedding_std']:.4f} "
            f"lr={epoch_metrics['learning_rate']:.6g}"
        )

    console.print(f"Checkpoint saved to: [bold]{run_dir / 'latest.pt'}[/bold]")


if __name__ == "__main__":
    main()
