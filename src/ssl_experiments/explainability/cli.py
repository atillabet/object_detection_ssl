#!/usr/bin/env python3

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from ultralytics import YOLO

from ssl_experiments.explainability.core import (
    explain_image,
    find_images,
    write_run_outputs,
)

app = typer.Typer()
console = Console()


@app.command()
def main(
    source: Annotated[Path, typer.Argument(help="Image file or directory of images to analyze.")],
    weights: Annotated[Path, typer.Option("--weights", "-w", help="YOLO .pt weights path.")] = Path("yolo26n.pt"),
    output: Annotated[Path, typer.Option("--output", "-o", help="Output root for explainability runs.")] = Path(
        "artifacts/explainability"
    ),
    run_name: Annotated[str | None, typer.Option(help="Optional run name. Defaults to a timestamped folder.")] = None,
    imgsz: Annotated[int, typer.Option(help="Inference image size.")] = 320,
    conf: Annotated[float, typer.Option(help="Confidence threshold.")] = 0.05,
    iou: Annotated[float, typer.Option(help="NMS IoU threshold.")] = 0.5,
    device: Annotated[str | None, typer.Option(help="Torch device, e.g. 0 or cpu.")] = None,
    max_images: Annotated[int | None, typer.Option(help="Maximum number of images to process.")] = None,
    max_det: Annotated[int, typer.Option(help="Max detections per inference pass.")] = 300,
    max_detections: Annotated[int, typer.Option(help="Analyze up to N top detections per image.")] = 5,
    selection: Annotated[
        str,
        typer.Option("--selection", help="Detection selection: top_conf, false_positive, mixed, or low_conf."),
    ] = "top_conf",
    max_candidates: Annotated[
        int | None,
        typer.Option("--max-candidates", help="Maximum raw candidates to log per branch per image. Defaults to all."),
    ] = None,
    sahi: Annotated[bool, typer.Option("--sahi/--no-sahi", help="Use SAHI tiled inference for source images.")] = True,
    counterfactual_sahi: Annotated[
        bool,
        typer.Option(
            "--counterfactual-sahi/--no-counterfactual-sahi",
            help="Use SAHI for crop/object-mask/occlusion re-predictions. Slower, but matches the source pipeline.",
        ),
    ] = False,
    slice_size: Annotated[int, typer.Option("--slice-size", help="SAHI tile size.")] = 320,
    overlap: Annotated[int, typer.Option(help="SAHI tile overlap in pixels.")] = 64,
    sahi_batch_size: Annotated[int, typer.Option("--sahi-batch-size", help="Batch size for SAHI tile inference.")] = 128,
    large_conf: Annotated[
        float,
        typer.Option("--large-conf", help="SAHI confidence threshold for objects larger than --size-threshold."),
    ] = 0.25,
    size_threshold: Annotated[
        int,
        typer.Option("--size-threshold", help="Min max(w,h) px to apply --large-conf in SAHI."),
    ] = 16,
    labels: Annotated[
        Path | None,
        typer.Option("--labels", "-l", help="YOLO labels directory. Defaults to a sibling labels/ directory."),
    ] = None,
    gt_iou_threshold: Annotated[float, typer.Option("--gt-iou", help="IoU threshold for prediction/GT matching.")] = 0.5,
    crop_margin: Annotated[float, typer.Option(help="Relative margin around the target box for the crop view.")] = 0.15,
    occlusion_patch: Annotated[int, typer.Option(help="Square occlusion patch size in pixels.")] = 48,
    occlusion_stride: Annotated[int, typer.Option(help="Occlusion patch stride in pixels.")] = 24,
    occlusion_scope: Annotated[
        str,
        typer.Option("--occlusion-scope", help="Occlusion grid scope: target or full."),
    ] = "target",
    occlusion_roi_margin: Annotated[
        float,
        typer.Option("--occlusion-roi-margin", help="Margin around detection box when --occlusion-scope target."),
    ] = 0.5,
    predict_batch_size: Annotated[
        int,
        typer.Option("--predict-batch-size", help="Batch size for counterfactual and occlusion inference."),
    ] = 32,
    drise_masks: Annotated[
        int,
        typer.Option("--drise-masks", help="Number of D-RISE random masks per detection. Use 0 to disable."),
    ] = 64,
    drise_min_masks: Annotated[
        int,
        typer.Option("--drise-min-masks", help="Minimum D-RISE masks to process before adaptive early stop."),
    ] = 64,
    drise_target_positive_masks: Annotated[
        int,
        typer.Option(
            "--drise-target-positive-masks",
            help="Stop D-RISE early after this many masks preserve the detection. Use 0 to disable.",
        ),
    ] = 24,
    drise_stop_if_no_signal_after: Annotated[
        int,
        typer.Option(
            "--drise-stop-if-no-signal-after",
            help="Stop D-RISE early after this many masks if none preserve the detection. Use 0 to disable.",
        ),
    ] = 64,
    drise_grid_size: Annotated[int, typer.Option("--drise-grid-size", help="D-RISE low-resolution mask grid size.")] = 12,
    drise_mask_prob: Annotated[
        float,
        typer.Option("--drise-mask-prob", help="Probability that a D-RISE mask cell remains visible."),
    ] = 0.5,
    drise_conf: Annotated[
        float | None,
        typer.Option(
            "--drise-conf",
            help="Confidence threshold for masked D-RISE predictions. Defaults to min(--conf, 0.01).",
        ),
    ] = None,
    drise_seed: Annotated[int, typer.Option("--drise-seed", help="Base random seed for D-RISE masks.")] = 0,
    robustness: Annotated[
        bool,
        typer.Option("--robustness/--no-robustness", help="Run blur/noise/compression/shift/scale robustness tests."),
    ] = True,
    sanity_checks: Annotated[
        bool,
        typer.Option("--sanity-checks/--no-sanity-checks", help="Run explanation sanity checks such as important-vs-random deletion."),
    ] = True,
    randomized_weights_sanity: Annotated[
        bool,
        typer.Option(
            "--randomized-weights-sanity/--no-randomized-weights-sanity",
            help="Also compare Grad-CAM with a temporarily randomized detection head. Slower.",
        ),
    ] = False,
    audit_data: Annotated[
        Path | None,
        typer.Option("--audit-data", help="Optional YOLO train dataset root or data.yaml for dataset audit/leakage checks."),
    ] = None,
    audit_max_images: Annotated[
        int | None,
        typer.Option("--audit-max-images", help="Maximum images to inspect for dataset audit. Defaults to the run image count."),
    ] = None,
):
    images = find_images(source, max_images=max_images)
    if not images:
        raise typer.BadParameter(f"No supported images found under {source}")

    run_name = run_name or f"{weights.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output / run_name

    console.print(f"[cyan]Loading model:[/cyan] {weights}")
    model = YOLO(str(weights))

    image_rows = []
    detection_rows = []
    candidate_rows = []
    postprocess_rows = []
    counterfactual_rows = []
    robustness_rows = []
    sanity_rows = []
    for image_path in images:
        console.print(f"[cyan]Explaining:[/cyan] {image_path}")
        image_summary, rows, candidates, postprocess, counterfactuals, robust_rows, sanity = explain_image(
            model=model,
            image_path=image_path,
            run_dir=run_dir,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            max_det=max_det,
            max_detections=max_detections,
            selection=selection,
            max_candidates=max_candidates,
            crop_margin=crop_margin,
            occlusion_patch=occlusion_patch,
            occlusion_stride=occlusion_stride,
            labels_dir=labels,
            gt_iou_threshold=gt_iou_threshold,
            predict_batch_size=predict_batch_size,
            occlusion_scope=occlusion_scope,
            occlusion_roi_margin=occlusion_roi_margin,
            use_sahi=sahi,
            counterfactual_sahi=counterfactual_sahi,
            slice_size=slice_size,
            overlap=overlap,
            sahi_batch_size=sahi_batch_size,
            large_conf=large_conf,
            size_threshold=size_threshold,
            drise_masks=drise_masks,
            drise_min_masks=drise_min_masks,
            drise_target_positive_masks=drise_target_positive_masks,
            drise_stop_if_no_signal_after=drise_stop_if_no_signal_after,
            drise_grid_size=drise_grid_size,
            drise_mask_prob=drise_mask_prob,
            drise_conf=drise_conf,
            drise_seed=drise_seed,
            robustness=robustness,
            sanity_checks=sanity_checks,
            randomized_weights_sanity=randomized_weights_sanity,
        )
        image_rows.append(image_summary)
        detection_rows.extend(rows)
        candidate_rows.extend(candidates)
        postprocess_rows.extend(postprocess)
        counterfactual_rows.extend(counterfactuals)
        robustness_rows.extend(robust_rows)
        sanity_rows.extend(sanity)

    model_names = getattr(model, "names", {})
    if isinstance(model_names, dict):
        class_names = [str(model_names[idx]) for idx in sorted(model_names)]
    else:
        class_names = [str(name) for name in model_names]
    manifest = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "weights": str(weights.resolve()),
        "source": str(source.resolve()),
        "labels": str(labels.resolve()) if labels is not None else None,
        "gt_iou_threshold": gt_iou_threshold,
        "num_images": len(image_rows),
        "num_detections": len(detection_rows),
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "sahi": sahi,
        "counterfactual_sahi": counterfactual_sahi,
        "slice_size": slice_size if sahi else None,
        "overlap": overlap if sahi else None,
        "sahi_batch_size": sahi_batch_size if sahi else None,
        "large_conf": large_conf if sahi else None,
        "size_threshold": size_threshold if sahi else None,
        "max_detections_per_image": max_detections,
        "selection": selection,
        "max_candidates_per_branch": max_candidates,
        "occlusion_patch": occlusion_patch,
        "occlusion_stride": occlusion_stride,
        "occlusion_scope": occlusion_scope,
        "occlusion_roi_margin": occlusion_roi_margin,
        "predict_batch_size": predict_batch_size,
        "drise_masks": drise_masks,
        "drise_min_masks": drise_min_masks,
        "drise_target_positive_masks": drise_target_positive_masks,
        "drise_stop_if_no_signal_after": drise_stop_if_no_signal_after,
        "drise_grid_size": drise_grid_size,
        "drise_mask_prob": drise_mask_prob,
        "drise_conf": drise_conf if drise_conf is not None else min(conf, 0.01),
        "drise_seed": drise_seed,
        "robustness": robustness,
        "sanity_checks": sanity_checks,
        "randomized_weights_sanity": randomized_weights_sanity,
        "audit_data": str(audit_data.resolve()) if audit_data is not None else None,
        "audit_max_images": audit_max_images if audit_max_images is not None else len(image_rows),
        "class_names": class_names,
    }
    write_run_outputs(
        run_dir=run_dir,
        run_manifest=manifest,
        images_rows=image_rows,
        detection_rows=detection_rows,
        candidate_rows=candidate_rows,
        postprocess_rows=postprocess_rows,
        counterfactual_rows=counterfactual_rows,
        robustness_rows=robustness_rows,
        sanity_rows=sanity_rows,
    )

    table = Table(title="Explainability Run Summary")
    table.add_column("Run")
    table.add_column("Images", justify="right")
    table.add_column("Detections", justify="right")
    table.add_column("Output")
    table.add_row(run_name, str(len(image_rows)), str(len(detection_rows)), str(run_dir))
    console.print(table)


if __name__ == "__main__":
    app()
