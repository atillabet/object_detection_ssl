import marimo

__generated_with = "0.24.0"
app = marimo.App(width="wide", app_title="YOLO Explainability Explorer")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from PIL import Image

    return Image, Path, json, mo, np, pl, plt


@app.cell
def _(mo):
    mo.md("""
    # YOLO Explainability Explorer

    Browse saved runs from `artifacts/explainability/`.
    Use the visible-section selector to show only the parts of the analysis you want to inspect.
    """)
    return


@app.cell
def _(Path, json, mo):
    root = (Path(__file__).parent / ".." / "artifacts" / "explainability").resolve()

    def discover_runs(base: Path):
        runs = {}
        if not base.exists():
            return runs
        for run_dir in sorted(base.iterdir()):
            manifest = run_dir / "manifest.json"
            images = run_dir / "images.parquet"
            detections = run_dir / "detections.parquet"
            if run_dir.is_dir() and manifest.exists() and images.exists() and detections.exists():
                runs[run_dir.name] = {
                    "path": run_dir,
                    "manifest": json.loads(manifest.read_text()),
                }
        return runs

    all_runs = discover_runs(root)
    if not all_runs:
        mo.stop(
            True,
            mo.callout(
                mo.md(f"No explainability runs found in **{root}**. Run `ssl-explain` first."),
                kind="warn",
            ),
        )

    run_selector = mo.ui.dropdown(
        options=sorted(all_runs.keys()),
        value=sorted(all_runs.keys())[-1],
        label="Run",
    )
    run_selector
    return all_runs, run_selector


@app.cell
def _(mo):
    section_options = [
        "Overview",
        "Artifact status",
        "Candidate logging",
        "Dataset audit",
        "Calibration",
        "Robustness",
        "D-RISE / heatmaps",
        "Sanity checks",
        "Failure clustering",
        "Recommendations",
        "Detection explorer",
        "Raw tables",
    ]
    section_selector = mo.ui.multiselect(
        options=section_options,
        value=section_options,
        label="Visible sections",
        full_width=True,
    )
    mo.vstack(
        [
            section_selector,
            mo.md(
                "Select one or more analysis sections. Empty selection hides the analysis output but keeps the run controls visible."
            ),
        ]
    )
    return (section_selector,)


@app.cell
def _(section_selector):
    selected_sections = set(section_selector.value or [])

    def section_is_visible(name: str) -> bool:
        return name in selected_sections

    def detection_detail_visible() -> bool:
        return section_is_visible("Detection explorer") or section_is_visible("D-RISE / heatmaps")

    return detection_detail_visible, section_is_visible


@app.cell
def _(all_runs, json, pl, run_selector):
    run = all_runs[run_selector.value]
    run_dir = run["path"]
    manifest = run["manifest"]
    df_images = pl.read_parquet(run_dir / "images.parquet")
    df_detections = pl.read_parquet(run_dir / "detections.parquet")
    df_candidates = (
        pl.read_parquet(run_dir / "raw_candidates.parquet")
        if (run_dir / "raw_candidates.parquet").exists()
        else pl.DataFrame()
    )
    df_postprocess = (
        pl.read_parquet(run_dir / "postprocess.parquet")
        if (run_dir / "postprocess.parquet").exists()
        else pl.DataFrame()
    )
    df_counterfactuals = (
        pl.read_parquet(run_dir / "counterfactuals.parquet")
        if (run_dir / "counterfactuals.parquet").exists()
        else pl.DataFrame()
    )
    df_dataset = (
        pl.read_parquet(run_dir / "dataset_audit.parquet")
        if (run_dir / "dataset_audit.parquet").exists()
        else pl.DataFrame()
    )
    df_dataset_boxes = (
        pl.read_parquet(run_dir / "dataset_boxes.parquet")
        if (run_dir / "dataset_boxes.parquet").exists()
        else pl.DataFrame()
    )
    df_calibration = (
        pl.read_parquet(run_dir / "calibration.parquet")
        if (run_dir / "calibration.parquet").exists()
        else pl.DataFrame()
    )
    df_thresholds = (
        pl.read_parquet(run_dir / "threshold_sweep.parquet")
        if (run_dir / "threshold_sweep.parquet").exists()
        else pl.DataFrame()
    )
    df_robustness = (
        pl.read_parquet(run_dir / "robustness.parquet")
        if (run_dir / "robustness.parquet").exists()
        else pl.DataFrame()
    )
    df_sanity = (
        pl.read_parquet(run_dir / "sanity_checks.parquet")
        if (run_dir / "sanity_checks.parquet").exists()
        else pl.DataFrame()
    )
    df_failure_clusters = (
        pl.read_parquet(run_dir / "failure_cluster_summary.parquet")
        if (run_dir / "failure_cluster_summary.parquet").exists()
        else pl.DataFrame()
    )
    df_failure_rows = (
        pl.read_parquet(run_dir / "failure_clusters.parquet")
        if (run_dir / "failure_clusters.parquet").exists()
        else pl.DataFrame()
    )
    df_leakage = (
        pl.read_parquet(run_dir / "train_val_leakage.parquet")
        if (run_dir / "train_val_leakage.parquet").exists()
        else pl.DataFrame()
    )
    dataset_summary = (
        json.loads((run_dir / "dataset_audit_summary.json").read_text())
        if (run_dir / "dataset_audit_summary.json").exists()
        else {}
    )
    calibration_summary = (
        json.loads((run_dir / "calibration_summary.json").read_text())
        if (run_dir / "calibration_summary.json").exists()
        else {}
    )
    recommendations = (
        json.loads((run_dir / "fix_recommendations.json").read_text())
        if (run_dir / "fix_recommendations.json").exists()
        else []
    )
    artifact_specs = [
        ("images", "images.parquet", df_images),
        ("detections", "detections.parquet", df_detections),
        ("raw candidates", "raw_candidates.parquet", df_candidates),
        ("postprocess", "postprocess.parquet", df_postprocess),
        ("counterfactuals", "counterfactuals.parquet", df_counterfactuals),
        ("robustness", "robustness.parquet", df_robustness),
        ("sanity checks", "sanity_checks.parquet", df_sanity),
        ("dataset audit images", "dataset_audit.parquet", df_dataset),
        ("dataset audit boxes", "dataset_boxes.parquet", df_dataset_boxes),
        ("train/val leakage", "train_val_leakage.parquet", df_leakage),
        ("calibration", "calibration.parquet", df_calibration),
        ("threshold sweep", "threshold_sweep.parquet", df_thresholds),
        ("failure clusters", "failure_clusters.parquet", df_failure_rows),
        ("failure cluster summary", "failure_cluster_summary.parquet", df_failure_clusters),
    ]
    artifact_rows = [
        {
            "artifact": name,
            "file": filename,
            "exists": (run_dir / filename).exists(),
            "rows": frame.height,
            "columns": len(frame.columns),
        }
        for name, filename, frame in artifact_specs
    ]
    artifact_rows.extend(
        [
            {
                "artifact": "dataset audit summary",
                "file": "dataset_audit_summary.json",
                "exists": (run_dir / "dataset_audit_summary.json").exists(),
                "rows": None,
                "columns": None,
            },
            {
                "artifact": "calibration summary",
                "file": "calibration_summary.json",
                "exists": (run_dir / "calibration_summary.json").exists(),
                "rows": None,
                "columns": None,
            },
            {
                "artifact": "fix recommendations",
                "file": "fix_recommendations.json",
                "exists": (run_dir / "fix_recommendations.json").exists(),
                "rows": len(recommendations),
                "columns": None,
            },
            {
                "artifact": "analysis report",
                "file": "analysis_report.md",
                "exists": (run_dir / "analysis_report.md").exists(),
                "rows": None,
                "columns": None,
            },
        ]
    )
    return (
        artifact_rows,
        calibration_summary,
        dataset_summary,
        df_calibration,
        df_candidates,
        df_counterfactuals,
        df_dataset,
        df_dataset_boxes,
        df_detections,
        df_failure_clusters,
        df_failure_rows,
        df_images,
        df_leakage,
        df_postprocess,
        df_robustness,
        df_sanity,
        df_thresholds,
        manifest,
        recommendations,
        run_dir,
    )


@app.cell
def _(manifest, mo, section_is_visible):
    mo.stop(not section_is_visible("Overview"), mo.md(""))

    mo.md(
        f"""
        ## Run Summary

        - **Run**: `{manifest["run_name"]}`
        - **Weights**: `{manifest["weights"]}`
        - **Source**: `{manifest["source"]}`
        - **Images**: `{manifest["num_images"]}`
        - **Detections**: `{manifest["num_detections"]}`
        - **imgsz/conf/iou**: `{manifest["imgsz"]}` / `{manifest["conf"]}` / `{manifest["iou"]}`
        - **Analysis report**: `analysis_report.md`
        """
    )
    return


@app.cell
def _(artifact_rows, mo, pl, run_dir, section_is_visible):
    mo.stop(not section_is_visible("Artifact status"), mo.md(""))

    reports = sorted((run_dir / "reports").glob("*.png")) if (run_dir / "reports").exists() else []
    asset_count = sum(1 for _ in (run_dir / "assets").rglob("*") if _.is_file()) if (run_dir / "assets").exists() else 0
    mo.md(
        f"""
        ## Artifact Status

        - **Run directory**: `{run_dir}`
        - **Report plots**: `{len(reports)}`
        - **Asset files**: `{asset_count}`
        """
    )
    mo.ui.table(pl.DataFrame(artifact_rows))
    return


@app.cell
def _(df_candidates, df_postprocess, mo, pl, section_is_visible):
    mo.stop(not section_is_visible("Candidate logging"), mo.md(""))
    mo.stop(df_candidates.is_empty(), mo.md("## Candidate Logging\n\nNo raw candidate log is available for this run."))

    _summary = df_candidates.select(
        [
            pl.len().alias("raw_candidates"),
            pl.col("kept_after_threshold").sum().alias("above_threshold"),
            pl.col("kept_after_nms").sum().alias("kept_after_postprocess"),
            pl.col("source_detection_id").is_not_null().sum().alias("matched_to_source"),
        ]
    ).to_dicts()[0]
    by_branch = df_candidates.group_by("branch").len().sort("len", descending=True)
    by_pipeline = (
        df_postprocess.group_by("pipeline").len().sort("len", descending=True)
        if not df_postprocess.is_empty()
        else pl.DataFrame()
    )
    mo.md(
        f"""
        ## Candidate Logging

        - **Raw candidates**: `{_summary["raw_candidates"]}`
        - **Above threshold**: `{_summary["above_threshold"]}`
        - **Kept after adapter postprocess**: `{_summary["kept_after_postprocess"]}`
        - **Matched to source detections**: `{_summary["matched_to_source"]}`
        """
    )
    mo.vstack([mo.ui.table(by_branch), mo.ui.table(by_pipeline) if not by_pipeline.is_empty() else mo.md("")])
    return


@app.cell
def _(df_candidates, mo, pl, section_is_visible):
    mo.stop(not section_is_visible("Candidate logging"), mo.md(""))
    if df_candidates.is_empty():
        mo.stop(True)

    interesting = (
        df_candidates.filter(
            pl.col("kept_after_nms") | pl.col("source_detection_id").is_not_null() | (pl.col("class_score") >= 0.25)
        )
        .select(
            [
                "image_name",
                "branch",
                "candidate_id",
                "class_name",
                "class_score",
                "kept_after_threshold",
                "kept_after_nms",
                "source_detection_id",
                "source_gt_status",
                "source_fp_type",
                "box_area",
                "aspect_ratio",
                "image_region",
            ]
        )
        .sort("class_score", descending=True)
        .head(200)
    )
    mo.md("## Candidate/Postprocess Samples")
    mo.ui.table(interesting)
    return


@app.cell
def _(
    dataset_summary,
    df_dataset,
    df_dataset_boxes,
    df_leakage,
    mo,
    pl,
    section_is_visible,
):
    mo.stop(not section_is_visible("Dataset audit"), mo.md(""))
    mo.stop(not dataset_summary, mo.md("## Dataset Audit\n\nNo dataset audit artifacts are available for this run."))

    role_rows = []
    for role, _values in dataset_summary.get("roles", {}).items():
        role_rows.append(
            {
                "role": role,
                "images": _values.get("images"),
                "boxes": _values.get("boxes"),
                "hard_negative_images": _values.get("hard_negative_images"),
                "missing_label_files": _values.get("missing_label_files"),
                "invalid_label_lines": _values.get("invalid_label_lines"),
                "tiny_boxes_lt8": _values.get("tiny_boxes_lt8"),
                "small_boxes_lt16": _values.get("small_boxes_lt16"),
                "duplicate_labels": _values.get("duplicate_labels"),
                "loose_box_proxy": _values.get("loose_box_proxy"),
                "blurred_images_proxy": _values.get("blurred_images_proxy"),
            }
        )
    role_table = pl.DataFrame(role_rows)
    class_table = (
        df_dataset_boxes.group_by(["dataset_role", "class_name"])
        .len()
        .sort(["dataset_role", "len"], descending=[False, True])
        if not df_dataset_boxes.is_empty()
        else pl.DataFrame()
    )
    bad_boxes = (
        df_dataset_boxes.filter(
            pl.col("tiny_box")
            | pl.col("small_box")
            | pl.col("invalid_label")
            | pl.col("duplicate_label")
            | pl.col("loose_box_proxy")
            | pl.col("extreme_aspect")
        )
        .select(
            [
                "dataset_role",
                "image_name",
                "class_name",
                "min_side",
                "area_ratio",
                "tiny_box",
                "small_box",
                "invalid_label",
                "duplicate_label",
                "loose_box_proxy",
                "extreme_aspect",
            ]
        )
        .head(200)
        if not df_dataset_boxes.is_empty()
        else pl.DataFrame()
    )
    image_issues = (
        df_dataset.filter(
            pl.col("missing_label_file")
            | pl.col("empty_label_file")
            | (pl.col("invalid_label_lines") > 0)
            | pl.col("blurred_image_proxy")
            | pl.col("low_contrast_proxy")
        )
        .select(
            [
                "dataset_role",
                "image_name",
                "num_labels",
                "missing_label_file",
                "empty_label_file",
                "invalid_label_lines",
                "blur_laplacian",
                "brightness",
                "contrast",
                "blurred_image_proxy",
                "low_contrast_proxy",
            ]
        )
        .head(200)
        if not df_dataset.is_empty()
        else pl.DataFrame()
    )
    mo.md(
        f"""
        ## Dataset Audit

        - **Potential leakage matches**: `{dataset_summary.get("leakage_count", 0)}`
        - **Image rows audited**: `{df_dataset.height if not df_dataset.is_empty() else 0}`
        - **Box rows audited**: `{df_dataset_boxes.height if not df_dataset_boxes.is_empty() else 0}`
        """
    )
    mo.vstack(
        [
            mo.ui.table(role_table),
            mo.md("### Class Balance"),
            mo.ui.table(class_table) if not class_table.is_empty() else mo.md("No box rows."),
            mo.md("### Image Issues"),
            mo.ui.table(image_issues) if not image_issues.is_empty() else mo.md("No image-level issue proxies found."),
            mo.md("### Labels To Review"),
            mo.ui.table(bad_boxes) if not bad_boxes.is_empty() else mo.md("No obvious bad-label proxies found."),
            mo.md("### Train/Val Leakage Candidates"),
            mo.ui.table(df_leakage.head(200)) if not df_leakage.is_empty() else mo.md("No leakage candidates found."),
        ]
    )
    return


@app.cell
def _(
    Image,
    calibration_summary,
    df_calibration,
    df_thresholds,
    mo,
    pl,
    run_dir,
    section_is_visible,
):
    mo.stop(not section_is_visible("Calibration"), mo.md(""))
    mo.stop(
        not calibration_summary,
        mo.md("## Confidence Calibration\n\nNo calibration artifacts are available for this run."),
    )

    best = calibration_summary.get("best_all_threshold")
    cal_table = (
        df_calibration.filter(pl.col("class_name") == "all").sort("bin_start")
        if not df_calibration.is_empty()
        else pl.DataFrame()
    )
    best_by_class = (
        df_thresholds.filter(pl.col("f1").is_not_null())
        .sort(["class_name", "f1"], descending=[False, True])
        .group_by("class_name")
        .first()
        .select(["class_name", "threshold", "precision", "recall", "f1"])
        if not df_thresholds.is_empty()
        else pl.DataFrame()
    )
    _panels = []
    for asset in ["reliability_diagram.png", "precision_recall_curve.png"]:
        path = run_dir / "reports" / asset
        if path.exists():
            _panels.append((asset, Image.open(path)))
    mo.md(
        f"""
        ## Confidence Calibration

        - **Expected calibration error**: `{calibration_summary.get("ece")}`
        - **Best all-class threshold row**: `{best}`
        """
    )
    views = [mo.ui.table(cal_table), mo.md("### Best Thresholds"), mo.ui.table(best_by_class)]
    for _title, _image in _panels:
        views.extend([mo.md(f"### {_title}"), mo.image(_image)])
    mo.vstack(views)
    return


@app.cell
def _(df_robustness, mo, pl, section_is_visible):
    mo.stop(not section_is_visible("Robustness"), mo.md(""))
    mo.stop(df_robustness.is_empty(), mo.md("## Robustness Tests\n\nNo robustness rows are available for this run."))

    by_transform = (
        df_robustness.group_by("transform")
        .agg(
            [
                pl.len().alias("n"),
                pl.col("score_retention").mean().alias("mean_score_retention"),
                pl.col("score_drop").mean().alias("mean_score_drop"),
                pl.col("matched_iou").mean().alias("mean_matched_iou"),
                pl.col("detection_present").mean().alias("present_rate"),
            ]
        )
        .sort("mean_score_retention")
    )
    worst = (
        df_robustness.select(
            [
                "image_name",
                "detection_id",
                "class_name",
                "transform",
                "original_score",
                "score",
                "score_retention",
                "matched_iou",
                "detection_present",
            ]
        )
        .sort("score_retention")
        .head(100)
    )
    mo.md("## Robustness Tests")
    mo.vstack([mo.ui.table(by_transform), mo.md("### Weakest Transform Cases"), mo.ui.table(worst)])
    return


@app.cell
def _(df_detections, df_sanity, mo, pl, section_is_visible):
    mo.stop(not section_is_visible("D-RISE / heatmaps"), mo.md(""))
    mo.stop(
        df_detections.is_empty() or "drise_num_masks" not in df_detections.columns,
        mo.md("## D-RISE / Heatmaps\n\nNo D-RISE metadata is available for this run."),
    )

    drise_rows = df_detections.filter(pl.col("drise_num_masks") > 0)
    mo.stop(drise_rows.is_empty(), mo.md("## D-RISE / Heatmaps\n\nD-RISE was disabled or produced no rows."))

    summary_columns = [
        pl.len().alias("detections_with_drise"),
        (pl.col("drise_weight_sum") <= 0).sum().alias("no_signal_detections"),
        (pl.col("drise_weight_sum") > 0).sum().alias("signal_detections"),
        pl.col("drise_requested_masks").mean().alias("mean_requested_masks")
        if "drise_requested_masks" in drise_rows.columns
        else pl.col("drise_num_masks").mean().alias("mean_requested_masks"),
        pl.col("drise_num_masks").mean().alias("mean_masks"),
        pl.col("drise_grid_size").mean().alias("mean_grid_size"),
        pl.col("drise_mask_probability").mean().alias("mean_mask_probability"),
        pl.col("drise_weight_sum").mean().alias("mean_weight_sum"),
        pl.col("drise_weight_sum").median().alias("median_weight_sum"),
        pl.col("drise_best_weight").mean().alias("mean_best_weight"),
        pl.col("occlusion_drise_corr").mean().alias("mean_occlusion_drise_corr"),
        pl.col("gradcam_drise_corr").mean().alias("mean_gradcam_drise_corr"),
    ]
    drise_summary = drise_rows.select(summary_columns)
    by_status = (
        drise_rows.group_by(["class_name", "gt_status", "fp_type"])
        .agg(
            [
                pl.len().alias("n"),
                pl.col("original_conf").mean().alias("mean_conf"),
                pl.col("drise_weight_sum").mean().alias("mean_weight_sum"),
                pl.col("drise_best_weight").mean().alias("mean_best_weight"),
                pl.col("occlusion_drise_corr").mean().alias("mean_occlusion_drise_corr"),
            ]
        )
        .sort("n", descending=True)
    )
    drise_table_columns = [
        "image_name",
        "detection_id",
        "class_name",
        "original_conf",
        "gt_status",
        "fp_type",
        "drise_requested_masks",
        "drise_num_masks",
        "drise_min_masks",
        "drise_target_positive_masks",
        "drise_stop_if_no_signal_after",
        "drise_early_stop_reason",
        "drise_grid_size",
        "drise_mask_probability",
        "drise_prediction_conf",
        "drise_prediction_large_conf",
        "drise_weight_sum",
        "drise_best_weight",
        "drise_best_mask_index",
        "drise_positive_masks",
        "drise_has_signal",
        "occlusion_drise_corr",
        "gradcam_drise_corr",
        "drise_sahi",
        "drise_asset",
    ]
    drise_available = [column for column in drise_table_columns if column in drise_rows.columns]
    drise_examples = (
        drise_rows.select(drise_available)
        .sort(["drise_weight_sum", "original_conf"], descending=[True, True])
        .head(200)
    )
    no_signal_examples = (
        drise_rows.filter(pl.col("drise_weight_sum") <= 0)
        .select(drise_available)
        .sort("original_conf", descending=True)
        .head(200)
    )
    sanity_columns = [
        "image_name",
        "detection_id",
        "class_name",
        "occlusion_drise_corr",
        "gradcam_drise_corr",
        "deletion_check_pass",
        "deletion_score_margin",
    ]
    sanity_available = [column for column in sanity_columns if column in df_sanity.columns]
    sanity_view = (
        df_sanity.select(sanity_available).sort("occlusion_drise_corr", descending=True).head(200)
        if not df_sanity.is_empty() and sanity_available
        else pl.DataFrame()
    )

    mo.md("## D-RISE / Heatmaps")
    mo.vstack(
        [
            mo.md("### D-RISE Summary"),
            mo.ui.table(drise_summary),
            mo.md("### D-RISE By GT Status"),
            mo.ui.table(by_status),
            mo.md("### D-RISE Detection Rows"),
            mo.ui.table(drise_examples),
            mo.md("### D-RISE No-Signal Rows"),
            mo.ui.table(no_signal_examples) if not no_signal_examples.is_empty() else mo.md("No zero-signal D-RISE rows."),
            mo.md("### D-RISE Sanity Agreement"),
            mo.ui.table(sanity_view) if not sanity_view.is_empty() else mo.md("No D-RISE sanity rows."),
        ]
    )
    return


@app.cell
def _(df_detections, mo, pl, section_is_visible):
    mo.stop(not section_is_visible("D-RISE / heatmaps"), mo.md(""))
    mo.stop(
        df_detections.is_empty() or "drise_weight_sum" not in df_detections.columns,
        mo.md("## D-RISE Image Viewer\n\nNo D-RISE metadata is available for this run."),
    )

    _signal_rows = (
        df_detections.filter((pl.col("drise_weight_sum") > 0) & pl.col("drise_asset").is_not_null())
        .select(
            [
                "image_key",
                "image_name",
                "detection_id",
                "class_name",
                "original_conf",
                "gt_status",
                "fp_type",
                "drise_weight_sum",
                "drise_best_weight",
                "drise_best_mask_index",
                "occlusion_drise_corr",
                "target_asset",
                "occlusion_asset",
                "drise_asset",
                "gradcam_asset",
            ]
        )
        .sort(["drise_weight_sum", "original_conf"], descending=[True, True])
        .to_dicts()
    )
    mo.stop(
        not _signal_rows,
        mo.md(
            "## D-RISE Image Viewer\n\nNo detections in this run have positive D-RISE signal. Check the no-signal table."
        ),
    )

    drise_signal_records = {}
    for _index, _row in enumerate(_signal_rows, start=1):
        _label = (
            f'{_index:03d} | {_row["image_name"]} | det {_row["detection_id"]} | '
            f'{_row["class_name"]} | conf {_row["original_conf"]:.3f} | '
            f'D-RISE {_row["drise_weight_sum"]:.2f}'
        )
        drise_signal_records[_label] = _row

    drise_signal_selector = mo.ui.dropdown(
        options=list(drise_signal_records.keys()),
        value=next(iter(drise_signal_records)),
        label="D-RISE signal detection",
    )
    drise_compare_selector = mo.ui.dropdown(
        options=["D-RISE", "Target", "Occlusion", "Grad-CAM"],
        value="D-RISE",
        label="Image to display",
    )
    drise_image_width_slider = mo.ui.slider(
        start=600,
        stop=2400,
        step=100,
        value=1600,
        show_value=True,
        include_input=True,
        full_width=True,
        label="D-RISE viewer width",
    )

    mo.md("## D-RISE Image Viewer")
    mo.vstack(
        [
            drise_signal_selector,
            mo.hstack([drise_compare_selector, drise_image_width_slider], justify="start", gap=2),
        ]
    )
    return (
        drise_compare_selector,
        drise_image_width_slider,
        drise_signal_records,
        drise_signal_selector,
    )


@app.cell
def _(
    Image,
    drise_compare_selector,
    drise_image_width_slider,
    drise_signal_records,
    drise_signal_selector,
    mo,
    run_dir,
):
    _row = drise_signal_records[drise_signal_selector.value]
    _asset_by_label = {
        "D-RISE": _row.get("drise_asset"),
        "Target": _row.get("target_asset"),
        "Occlusion": _row.get("occlusion_asset"),
        "Grad-CAM": _row.get("gradcam_asset"),
    }
    _asset = _asset_by_label.get(drise_compare_selector.value)
    mo.stop(not _asset or not (run_dir / _asset).exists(), mo.md("Selected image asset does not exist."))

    _caption = (
        f'{drise_compare_selector.value} | {_row["image_name"]} | det {_row["detection_id"]} | '
        f'conf={_row["original_conf"]:.4f} | drise_weight_sum={_row["drise_weight_sum"]:.4f} | '
        f'best_weight={_row["drise_best_weight"]:.4f}'
    )
    mo.image(
        Image.open(run_dir / _asset),
        width=f"{int(drise_image_width_slider.value)}px",
        caption=_caption,
        style={"max-width": "100%", "height": "auto"},
    )
    return


@app.cell
def _(drise_signal_records, drise_signal_selector, mo, pl):
    _row = drise_signal_records[drise_signal_selector.value]
    mo.ui.table(pl.DataFrame([_row]))
    return


@app.cell
def _(df_sanity, mo, pl, section_is_visible):
    mo.stop(not section_is_visible("Sanity checks"), mo.md(""))
    mo.stop(
        df_sanity.is_empty(),
        mo.md("## Explanation Sanity Checks\n\nNo sanity-check rows are available for this run."),
    )

    _columns = [
        "image_name",
        "detection_id",
        "class_name",
        "occlusion_gradcam_corr",
        "occlusion_drise_corr",
        "gradcam_drise_corr",
        "gradcam_randomized_corr",
        "deletion_important_drop",
        "deletion_random_drop",
        "deletion_score_margin",
        "deletion_check_pass",
        "target_change_check",
        "randomized_weights_check",
    ]
    available = [column for column in _columns if column in df_sanity.columns]
    summary_exprs = [pl.len().alias("rows")]
    if "deletion_check_pass" in df_sanity.columns:
        summary_exprs.extend(
            [
                pl.col("deletion_check_pass").sum().alias("deletion_pass"),
                (~pl.col("deletion_check_pass")).sum().alias("deletion_fail"),
            ]
        )
    for column in ["occlusion_gradcam_corr", "occlusion_drise_corr", "gradcam_drise_corr", "gradcam_randomized_corr"]:
        if column in df_sanity.columns:
            summary_exprs.append(pl.col(column).mean().alias(f"mean_{column}"))
    sanity_summary = df_sanity.select(summary_exprs)
    mo.md("## Explanation Sanity Checks")
    mo.vstack([mo.ui.table(sanity_summary), mo.ui.table(df_sanity.select(available))])
    return


@app.cell
def _(df_failure_clusters, df_failure_rows, mo, section_is_visible):
    mo.stop(not section_is_visible("Failure clustering"), mo.md(""))
    mo.stop(
        df_failure_clusters.is_empty(),
        mo.md("## Failure Clustering\n\nNo failure clusters were generated for this run."),
    )

    mo.md("## Failure Clustering")
    mo.vstack(
        [
            mo.md("### Cluster Summary"),
            mo.ui.table(df_failure_clusters),
            mo.md("### Cluster Members"),
            mo.ui.table(df_failure_rows.head(500)) if not df_failure_rows.is_empty() else mo.md("No cluster-member rows."),
        ]
    )
    return


@app.cell
def _(mo, recommendations, section_is_visible):
    mo.stop(not section_is_visible("Recommendations"), mo.md(""))
    mo.stop(
        not recommendations,
        mo.md("## Fix Recommendation Matrix\n\nNo recommendation matrix is available for this run."),
    )

    _recommendation_rows = [
        {
            "priority": row.get("priority"),
            "diagnosis": row.get("diagnosis"),
            "evidence": row.get("evidence"),
            "action": row.get("action"),
        }
        for row in recommendations
    ]
    mo.md("## Fix Recommendation Matrix")
    mo.ui.table(_recommendation_rows)
    return


@app.cell
def _(mo, section_is_visible):
    mo.stop(not section_is_visible("Raw tables"), mo.md(""))

    raw_table_selector = mo.ui.dropdown(
        options=[
            "detections",
            "raw candidates",
            "postprocess",
            "counterfactuals",
            "robustness",
            "sanity checks",
            "dataset audit images",
            "dataset audit boxes",
            "train/val leakage",
            "calibration",
            "threshold sweep",
            "failure clusters",
            "failure cluster summary",
        ],
        value="detections",
        label="Raw table",
    )
    raw_table_selector
    return (raw_table_selector,)


@app.cell
def _(
    df_calibration,
    df_candidates,
    df_counterfactuals,
    df_dataset,
    df_dataset_boxes,
    df_detections,
    df_failure_clusters,
    df_failure_rows,
    df_leakage,
    df_postprocess,
    df_robustness,
    df_sanity,
    df_thresholds,
    mo,
    pl,
    raw_table_selector,
    section_is_visible,
):
    mo.stop(not section_is_visible("Raw tables"), mo.md(""))

    raw_tables = {
        "detections": df_detections,
        "raw candidates": df_candidates,
        "postprocess": df_postprocess,
        "counterfactuals": df_counterfactuals,
        "robustness": df_robustness,
        "sanity checks": df_sanity,
        "dataset audit images": df_dataset,
        "dataset audit boxes": df_dataset_boxes,
        "train/val leakage": df_leakage,
        "calibration": df_calibration,
        "threshold sweep": df_thresholds,
        "failure clusters": df_failure_rows,
        "failure cluster summary": df_failure_clusters,
    }
    selected_raw = raw_tables[raw_table_selector.value]
    schema_rows = [
        {"column": name, "dtype": str(dtype)}
        for name, dtype in zip(selected_raw.columns, selected_raw.dtypes, strict=False)
    ]
    preview = selected_raw.head(5000)
    mo.md(
        f"""
        ## Raw Tables

        - **Selected**: `{raw_table_selector.value}`
        - **Rows**: `{selected_raw.height}`
        - **Columns**: `{len(selected_raw.columns)}`
        - **Displayed preview rows**: `{preview.height}`
        """
    )
    mo.vstack(
        [
            mo.md("### Schema"),
            mo.ui.table(pl.DataFrame(schema_rows)),
            mo.md("### Data Preview"),
            mo.ui.dataframe(preview, page_size=25, limit=5000, lazy=True),
        ]
    )
    return


@app.cell
def _(df_detections, mo, pl, section_is_visible):
    mo.stop(not section_is_visible("Overview"), mo.md(""))
    if df_detections.is_empty():
        mo.stop(True)

    summary = df_detections.select(
        [
            pl.len().alias("detections"),
            pl.col("original_conf").mean().alias("mean_original_conf"),
            pl.col("crop_score_same_class").mean().alias("mean_crop_score"),
            pl.col("object_only_score").mean().alias("mean_object_only_score"),
            pl.col("object_masked_score").mean().alias("mean_object_masked_score"),
            pl.col("mask_drop").mean().alias("mean_mask_drop"),
            pl.col("context_drop").mean().alias("mean_context_drop"),
            pl.col("occlusion_max_drop").mean().alias("mean_occlusion_max_drop"),
            (pl.col("mask_drop") > pl.col("context_drop") + 0.1).sum().alias("object_driven_count"),
            (pl.col("context_drop") > pl.col("mask_drop") + 0.1).sum().alias("context_driven_count"),
            ((pl.col("context_drop") - pl.col("mask_drop")).abs() <= 0.1).sum().alias("balanced_count"),
        ]
    ).to_dicts()[0]

    if summary["mean_context_drop"] > summary["mean_mask_drop"] + 0.05:
        _interpretation = "Overall this run looks context-reliant: removing background hurts more than hiding the detected object."
    elif summary["mean_mask_drop"] > summary["mean_context_drop"] + 0.05:
        _interpretation = "Overall this run looks object-reliant: hiding the detected object hurts more than removing the background."
    else:
        _interpretation = "Overall this run looks mixed: object pixels and context contribute at similar levels."

    mo.md(
        f"""
        ## Run-Level Explainability Summary

        - **Detections analyzed**: `{summary["detections"]}`
        - **Mean original confidence**: `{summary["mean_original_conf"]:.4f}`
        - **Mean object-only score**: `{summary["mean_object_only_score"]:.4f}`
        - **Mean object-masked score**: `{summary["mean_object_masked_score"]:.4f}`
        - **Mean mask drop**: `{summary["mean_mask_drop"]:.4f}`
        - **Mean context drop**: `{summary["mean_context_drop"]:.4f}`
        - **Object-driven detections**: `{summary["object_driven_count"]}`
        - **Context-driven detections**: `{summary["context_driven_count"]}`
        - **Balanced detections**: `{summary["balanced_count"]}`

        **Interpretation**: {_interpretation}
        """
    )
    return


@app.cell
def _(df_detections, mo, pl, section_is_visible):
    mo.stop(not section_is_visible("Overview"), mo.md(""))
    if df_detections.is_empty():
        mo.stop(True)

    by_class = (
        df_detections.group_by("class_name")
        .agg(
            [
                pl.len().alias("n"),
                pl.col("original_conf").mean().alias("original_conf"),
                pl.col("object_only_score").mean().alias("object_only_score"),
                pl.col("object_masked_score").mean().alias("object_masked_score"),
                pl.col("mask_drop").mean().alias("mask_drop"),
                pl.col("context_drop").mean().alias("context_drop"),
            ]
        )
        .sort("n", descending=True)
    )
    mo.md("## By Class")
    mo.ui.table(by_class)
    return


@app.cell
def _(df_detections, mo, pl, section_is_visible):
    mo.stop(not section_is_visible("Overview"), mo.md(""))
    if df_detections.is_empty():
        mo.stop(True)

    by_image = (
        df_detections.group_by(["image_name", "image_key"])
        .agg(
            [
                pl.len().alias("n"),
                pl.col("original_conf").mean().alias("original_conf"),
                pl.col("mask_drop").mean().alias("mask_drop"),
                pl.col("context_drop").mean().alias("context_drop"),
                pl.col("object_only_score").mean().alias("object_only_score"),
                pl.col("object_masked_score").mean().alias("object_masked_score"),
            ]
        )
        .sort("context_drop", descending=True)
    )
    mo.md("## By Image")
    mo.ui.table(by_image)
    return


@app.cell
def _(df_detections, mo, section_is_visible):
    mo.stop(not section_is_visible("Overview"), mo.md(""))
    if df_detections.is_empty():
        mo.stop(True)

    top_context = (
        df_detections.select(
            [
                "image_name",
                "detection_id",
                "class_name",
                "original_conf",
                "object_only_score",
                "object_masked_score",
                "mask_drop",
                "context_drop",
            ]
        )
        .sort("context_drop", descending=True)
        .head(10)
    )
    mo.md("## Most Context-Reliant Detections")
    mo.ui.table(top_context)
    return


@app.cell
def _(df_detections, mo, section_is_visible):
    mo.stop(not section_is_visible("Overview"), mo.md(""))
    if df_detections.is_empty():
        mo.stop(True)

    top_object = (
        df_detections.select(
            [
                "image_name",
                "detection_id",
                "class_name",
                "original_conf",
                "object_only_score",
                "object_masked_score",
                "mask_drop",
                "context_drop",
            ]
        )
        .sort("mask_drop", descending=True)
        .head(10)
    )
    mo.md("## Most Object-Reliant Detections")
    mo.ui.table(top_object)
    return


@app.cell
def _(detection_detail_visible, df_images, mo):
    mo.stop(not detection_detail_visible(), mo.md(""))

    image_options = df_images["image_key"].to_list()
    image_selector = mo.ui.dropdown(
        options=image_options,
        value=image_options[0],
        label="Image",
    )
    image_selector
    return (image_selector,)


@app.cell
def _(detection_detail_visible, df_detections, image_selector, mo, pl):
    mo.stop(not detection_detail_visible(), mo.md(""))

    image_rows = df_detections.filter(pl.col("image_key") == image_selector.value).sort("original_conf", descending=True)
    if image_rows.is_empty():
        mo.stop(True, mo.callout(mo.md("This image has no saved detections."), kind="warn"))

    detection_options = [str(v) for v in image_rows["detection_id"].to_list()]
    detection_selector = mo.ui.dropdown(
        options=detection_options,
        value=detection_options[0],
        label="Detection ID",
    )
    detection_selector
    return detection_selector, image_rows


@app.cell
def _(
    Image,
    compact_grid_switch,
    detection_detail_visible,
    df_images,
    image_selector,
    mo,
    np,
    plt,
    run_dir,
):
    mo.stop(not detection_detail_visible(), mo.md(""))
    mo.stop(not compact_grid_switch.value, mo.md(""))

    row = df_images.filter(df_images["image_key"] == image_selector.value).to_dicts()[0]
    _panels = [
        (np.array(Image.open(run_dir / row["original_asset"])), "Original"),
        (np.array(Image.open(run_dir / row["annotated_asset"])), "All detections"),
    ]
    if row.get("pred_vs_gt_asset"):
        _panels.append((np.array(Image.open(run_dir / row["pred_vs_gt_asset"])), "Predictions vs GT"))

    _fig, _axes = plt.subplots(1, len(_panels), figsize=(7 * len(_panels), 6))
    if len(_panels) == 1:
        _axes = [_axes]
    for _ax, (_img, _title) in zip(_axes, _panels):
        _ax.imshow(_img)
        _ax.set_title(_title)
        _ax.axis("off")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(
    detection_detail_visible,
    detection_selector,
    df_detections,
    image_selector,
    mo,
    pl,
):
    mo.stop(not detection_detail_visible(), mo.md(""))

    det = (
        df_detections.filter(
            (pl.col("image_key") == image_selector.value)
            & (pl.col("detection_id") == int(detection_selector.value))
        )
        .to_dicts()[0]
    )
    return (det,)


@app.cell
def _(detection_detail_visible, mo):
    mo.stop(not detection_detail_visible(), mo.md(""))

    image_view_width_slider = mo.ui.slider(
        start=600,
        stop=2400,
        step=100,
        value=1400,
        show_value=True,
        include_input=True,
        full_width=True,
        label="Large image width",
    )
    compact_grid_switch = mo.ui.switch(value=False, label="Show compact comparison grids")
    mo.hstack([image_view_width_slider, compact_grid_switch], justify="start", gap=2)
    return compact_grid_switch, image_view_width_slider


@app.cell
def _(
    det,
    detection_detail_visible,
    df_counterfactuals,
    df_images,
    image_selector,
    mo,
    pl,
    run_dir,
):
    mo.stop(not detection_detail_visible(), mo.md(""))

    _image_record = df_images.filter(pl.col("image_key") == image_selector.value).to_dicts()[0]
    visual_asset_rows = []

    def _add_visual(label: str, asset: str | None, group: str, score: float | None = None) -> None:
        if not asset:
            return
        if not (run_dir / asset).exists():
            return
        visual_asset_rows.append(
            {
                "label": label,
                "asset": asset,
                "group": group,
                "score": score,
            }
        )

    _add_visual("Image: original", _image_record.get("original_asset"), "image")
    _add_visual("Image: all detections", _image_record.get("annotated_asset"), "image")
    _add_visual("Image: predictions vs GT", _image_record.get("pred_vs_gt_asset"), "image")
    _add_visual("Detection: target", det.get("target_asset"), "detection")
    _add_visual("Detection: crop", det.get("crop_asset"), "detection", det.get("crop_score_same_class"))
    _add_visual("Detection: object only", det.get("object_only_asset"), "counterfactual", det.get("object_only_score"))
    _add_visual("Detection: object masked", det.get("object_masked_asset"), "counterfactual", det.get("object_masked_score"))
    _add_visual("Heatmap: occlusion", det.get("occlusion_asset"), "heatmap")
    if (det.get("drise_weight_sum") or 0.0) > 0:
        _add_visual("Heatmap: D-RISE", det.get("drise_asset"), "heatmap")
    _gradcam_is_current = det.get("gradcam_letterbox_content_xyxy") is not None
    _gradcam_has_signal = det.get("gradcam_has_signal")
    if _gradcam_is_current and (_gradcam_has_signal is None or bool(_gradcam_has_signal)):
        _add_visual("Heatmap: Grad-CAM", det.get("gradcam_asset"), "heatmap")

    if not df_counterfactuals.is_empty() and "asset" in df_counterfactuals.columns:
        for _cf_row in (
            df_counterfactuals.filter(
                (pl.col("image_key") == det["image_key"])
                & (pl.col("detection_id") == det["detection_id"])
                & pl.col("asset").is_not_null()
            )
            .sort("variant")
            .to_dicts()
        ):
            _add_visual(
                f'Counterfactual: {_cf_row["variant"]}',
                _cf_row.get("asset"),
                "counterfactual",
                _cf_row.get("score"),
            )

    mo.stop(not visual_asset_rows, mo.callout(mo.md("No image assets are available for this selection."), kind="warn"))
    visual_asset_selector = mo.ui.dropdown(
        options=[_row["label"] for _row in visual_asset_rows],
        value=visual_asset_rows[0]["label"],
        label="Large image",
    )
    visual_asset_selector
    return visual_asset_rows, visual_asset_selector


@app.cell
def _(det, detection_detail_visible, mo):
    mo.stop(not detection_detail_visible(), mo.md(""))
    mo.stop((det.get("drise_num_masks") or 0) <= 0 or (det.get("drise_weight_sum") or 0.0) > 0, mo.md(""))

    mo.callout(
        mo.md(
            f"""
            **D-RISE has no positive signal for this detection.**

            The saved blue D-RISE PNG is a legacy zero heatmap rendered through the JET color map.
            It means none of the `{det.get("drise_num_masks")}` random masks preserved this same detection
            with positive `score * IoU`, so there is no meaningful D-RISE region to display.
            """
        ),
        kind="warn",
    )
    return


@app.cell
def _(
    Image,
    image_view_width_slider,
    mo,
    run_dir,
    visual_asset_rows,
    visual_asset_selector,
):
    _selected_visual = next(_row for _row in visual_asset_rows if _row["label"] == visual_asset_selector.value)
    _score = _selected_visual.get("score")
    _caption = f'{_selected_visual["label"]} | {_selected_visual["asset"]}'
    if _score is not None:
        _caption += f" | score={_score:.4f}"
    mo.image(
        Image.open(run_dir / _selected_visual["asset"]),
        width=f"{int(image_view_width_slider.value)}px",
        caption=_caption,
        style={"max-width": "100%", "height": "auto"},
    )
    return


@app.cell
def _(mo, visual_asset_rows):
    mo.md("## Available Images For This Selection")
    mo.ui.table(visual_asset_rows)
    return


@app.cell
def _(det, detection_detail_visible, mo):
    mo.stop(not detection_detail_visible(), mo.md(""))

    gt_lines = ""
    if det.get("gt_available"):
        gt_lines = f"""
        - **GT status**: `{det.get("gt_status")}`
        - **FP type**: `{det.get("fp_type")}`
        - **Matched GT IoU**: `{det.get("matched_gt_iou")}`
        - **Best GT**: `{det.get("best_gt_class_name")}` / IoU `{det.get("best_gt_iou")}`
        """

    mo.md(
        f"""
        ## Detection Summary

        - **Class**: `{det["class_name"]}` (`{det["class_id"]}`)
        - **Original confidence**: `{det["original_conf"]:.4f}`
        - **Matched raw candidate**: `{det.get("matched_candidate_branch")}` / `{det.get("matched_candidate_id")}`
        {gt_lines}
        - **Crop same-class score**: `{det["crop_score_same_class"]:.4f}`
        - **Object-only score**: `{det["object_only_score"]:.4f}`
        - **Object-masked score**: `{det["object_masked_score"]:.4f}`
        - **Mask drop**: `{det["mask_drop"]:.4f}`
        - **Context drop**: `{det["context_drop"]:.4f}`
        - **Occlusion max drop**: `{det["occlusion_max_drop"]:.4f}`
        - **D-RISE masks**: `{det.get("drise_num_masks")}`
        - **Grad-CAM branch/candidate**: `{det.get("gradcam_branch")}` / `{det.get("gradcam_anchor_index")}`
        - **Grad-CAM anchor IoU**: `{det.get("gradcam_anchor_iou")}`
        """
    )
    return


@app.cell
def _(det, detection_detail_visible, mo):
    mo.stop(not detection_detail_visible(), mo.md(""))

    _warnings = []
    if det.get("gradcam_asset") and det.get("gradcam_letterbox_content_xyxy") is None:
        _warnings.append(
            "This run was generated before the corrected letterbox-to-original Grad-CAM mapping. Legacy Grad-CAM overlays are hidden because they can appear as misleading blue frames. Rerun explainability to regenerate corrected Grad-CAM overlays."
        )
    if det.get("gradcam_asset") and det.get("gradcam_has_signal") is False:
        _warnings.append(
            "Grad-CAM has no positive heatmap signal for this detection. The saved overlay is not shown because a zero heatmap can look like a blue mask."
        )
    _gradcam_raw_max = det.get("gradcam_raw_max")
    if _gradcam_raw_max is not None and 0 < _gradcam_raw_max < 1e-5:
        _warnings.append(
            f"Grad-CAM raw signal is very weak ({_gradcam_raw_max:.2e}). The overlay is normalized for visibility, so interpret it cautiously."
        )
    _anchor_iou = det.get("gradcam_anchor_iou")
    if _anchor_iou is not None and _anchor_iou < 0.1:
        _warnings.append(
            f"Grad-CAM target anchor IoU is very low ({_anchor_iou:.4f}), so the heatmap may explain a different raw candidate than the selected detection."
        )
    mo.stop(not _warnings, mo.md(""))
    mo.callout(mo.md("\n\n".join(f"- {_warning}" for _warning in _warnings)), kind="warn")
    return


@app.cell
def _(det, detection_detail_visible, df_robustness, df_sanity, mo, pl):
    mo.stop(not detection_detail_visible(), mo.md(""))

    heatmap_rows = [
        {"metric": "occlusion_max_drop", "value": det.get("occlusion_max_drop")},
        {"metric": "occlusion_mean_drop", "value": det.get("occlusion_mean_drop")},
        {"metric": "occlusion_peak_patch_xyxy", "value": det.get("occlusion_peak_patch_xyxy")},
        {"metric": "drise_num_masks", "value": det.get("drise_num_masks")},
        {"metric": "drise_requested_masks", "value": det.get("drise_requested_masks")},
        {"metric": "drise_min_masks", "value": det.get("drise_min_masks")},
        {"metric": "drise_target_positive_masks", "value": det.get("drise_target_positive_masks")},
        {"metric": "drise_stop_if_no_signal_after", "value": det.get("drise_stop_if_no_signal_after")},
        {"metric": "drise_early_stop_reason", "value": det.get("drise_early_stop_reason")},
        {"metric": "drise_grid_size", "value": det.get("drise_grid_size")},
        {"metric": "drise_mask_probability", "value": det.get("drise_mask_probability")},
        {"metric": "drise_prediction_conf", "value": det.get("drise_prediction_conf")},
        {"metric": "drise_prediction_large_conf", "value": det.get("drise_prediction_large_conf")},
        {"metric": "drise_weight_sum", "value": det.get("drise_weight_sum")},
        {"metric": "drise_best_weight", "value": det.get("drise_best_weight")},
        {"metric": "drise_best_mask_index", "value": det.get("drise_best_mask_index")},
        {"metric": "drise_positive_masks", "value": det.get("drise_positive_masks")},
        {"metric": "drise_has_signal", "value": det.get("drise_has_signal")},
        {"metric": "drise_sahi", "value": det.get("drise_sahi")},
        {"metric": "gradcam_branch", "value": det.get("gradcam_branch")},
        {"metric": "gradcam_anchor_index", "value": det.get("gradcam_anchor_index")},
        {"metric": "gradcam_anchor_iou", "value": det.get("gradcam_anchor_iou")},
        {"metric": "gradcam_requested_layer_index", "value": det.get("gradcam_requested_layer_index")},
        {"metric": "gradcam_feature_level", "value": det.get("gradcam_feature_level")},
        {"metric": "gradcam_stride", "value": det.get("gradcam_stride")},
        {"metric": "gradcam_grid_x", "value": det.get("gradcam_grid_x")},
        {"metric": "gradcam_grid_y", "value": det.get("gradcam_grid_y")},
        {"metric": "gradcam_min_anchor_iou", "value": det.get("gradcam_min_anchor_iou")},
        {"metric": "gradcam_letterbox_content_xyxy", "value": det.get("gradcam_letterbox_content_xyxy")},
        {"metric": "gradcam_raw_max", "value": det.get("gradcam_raw_max")},
        {"metric": "gradcam_max", "value": det.get("gradcam_max")},
        {"metric": "gradcam_has_signal", "value": det.get("gradcam_has_signal")},
        {"metric": "occlusion_gradcam_corr", "value": det.get("occlusion_gradcam_corr")},
        {"metric": "occlusion_drise_corr", "value": det.get("occlusion_drise_corr")},
        {"metric": "gradcam_drise_corr", "value": det.get("gradcam_drise_corr")},
        {"metric": "deletion_check_pass", "value": det.get("deletion_check_pass")},
        {"metric": "deletion_score_margin", "value": det.get("deletion_score_margin")},
    ]
    sanity_selected = (
        df_sanity.filter(
            (pl.col("image_key") == det["image_key"]) & (pl.col("detection_id") == det["detection_id"])
        )
        if not df_sanity.is_empty()
        else pl.DataFrame()
    )
    robustness_selected = (
        df_robustness.filter(
            (pl.col("image_key") == det["image_key"]) & (pl.col("detection_id") == det["detection_id"])
        )
        .select(
            [
                "transform",
                "original_score",
                "score",
                "score_drop",
                "score_retention",
                "matched_iou",
                "detection_present",
            ]
        )
        .sort("score_retention")
        if not df_robustness.is_empty()
        else pl.DataFrame()
    )

    mo.md("## Selected Detection Heatmap, Sanity, And Robustness")
    mo.vstack(
        [
            mo.md("### Heatmap Metadata"),
            mo.ui.table(heatmap_rows),
            mo.md("### Sanity Row"),
            mo.ui.table(sanity_selected) if not sanity_selected.is_empty() else mo.md("No sanity row for this detection."),
            mo.md("### Robustness Rows"),
            mo.ui.table(robustness_selected)
            if not robustness_selected.is_empty()
            else mo.md("No robustness rows for this detection."),
        ]
    )
    return


@app.cell
def _(det, detection_detail_visible, mo, plt):
    mo.stop(not detection_detail_visible(), mo.md(""))

    labels = ["original", "crop", "object_only", "object_masked"]
    values = [
        det["original_conf"],
        det["crop_score_same_class"],
        det["object_only_score"],
        det["object_masked_score"],
    ]
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.bar(labels, values, color=["#264653", "#2a9d8f", "#e9c46a", "#e76f51"])
    _ax.set_ylim(0, 1)
    _ax.set_ylabel("score")
    _ax.set_title("Score retention across counterfactual views")
    _ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(det, detection_detail_visible, df_counterfactuals, mo, pl):
    mo.stop(not detection_detail_visible(), mo.md(""))

    if df_counterfactuals.is_empty():
        mo.stop(True)

    cf = (
        df_counterfactuals.filter(
            (pl.col("image_key") == det["image_key"]) & (pl.col("detection_id") == det["detection_id"])
        )
        .select(["variant", "score", "matched_iou", "score_drop", "detection_present"])
        .sort("variant")
    )
    mo.md("## Counterfactual Score Table")
    mo.ui.table(cf)
    return


@app.cell
def _(
    Image,
    compact_grid_switch,
    det,
    detection_detail_visible,
    df_counterfactuals,
    mo,
    np,
    pl,
    plt,
    run_dir,
):
    mo.stop(not detection_detail_visible(), mo.md(""))
    mo.stop(not compact_grid_switch.value, mo.md(""))

    if df_counterfactuals.is_empty() or "asset" not in df_counterfactuals.columns:
        mo.stop(True)

    cf_assets = (
        df_counterfactuals.filter(
            (pl.col("image_key") == det["image_key"])
            & (pl.col("detection_id") == det["detection_id"])
            & pl.col("asset").is_not_null()
        )
        .select(["variant", "score", "asset"])
        .sort("variant")
        .to_dicts()
    )
    mo.stop(not cf_assets, mo.md(""))

    _cols = 3
    _rows = int(np.ceil(len(cf_assets) / _cols))
    _fig, _axes = plt.subplots(_rows, _cols, figsize=(6 * _cols, 4.5 * _rows))
    _axes = np.array(_axes).reshape(_rows, _cols)
    for index, item in enumerate(cf_assets):
        _ax = _axes.flat[index]
        _ax.imshow(np.array(Image.open(run_dir / item["asset"])))
        score = item.get("score")
        _title = item["variant"] if score is None else f'{item["variant"]}: {score:.3f}'
        _ax.set_title(_title)
        _ax.axis("off")
    for index in range(len(cf_assets), _rows * _cols):
        _axes.flat[index].axis("off")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(compact_grid_switch, detection_detail_visible, mo):
    mo.stop(not detection_detail_visible(), mo.md(""))
    mo.stop(not compact_grid_switch.value, mo.md(""))

    mo.md(
        """
        ## Visual Evidence

        The first row shows the target detection and its counterfactual variants.
        The second row shows which regions most affected the confidence score.
        """
    )
    return


@app.cell
def _(
    Image,
    compact_grid_switch,
    det,
    detection_detail_visible,
    mo,
    np,
    plt,
    run_dir,
):
    mo.stop(not detection_detail_visible(), mo.md(""))
    mo.stop(not compact_grid_switch.value, mo.md(""))

    target = np.array(Image.open(run_dir / det["target_asset"]))
    crop = np.array(Image.open(run_dir / det["crop_asset"]))
    object_only = np.array(Image.open(run_dir / det["object_only_asset"]))
    object_masked = np.array(Image.open(run_dir / det["object_masked_asset"]))

    _fig, _axes = plt.subplots(2, 2, figsize=(14, 10))
    _panels = [
        (target, "Target"),
        (crop, "Crop"),
        (object_only, "Object only"),
        (object_masked, "Object masked"),
    ]
    for _ax, (_img, _title) in zip(_axes.flat, _panels):
        _ax.imshow(_img)
        _ax.set_title(_title)
        _ax.axis("off")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(
    Image,
    compact_grid_switch,
    det,
    detection_detail_visible,
    mo,
    np,
    plt,
    run_dir,
):
    mo.stop(not detection_detail_visible(), mo.md(""))
    mo.stop(not compact_grid_switch.value, mo.md(""))

    panels = [(np.array(Image.open(run_dir / det["occlusion_asset"])), "Occlusion sensitivity")]
    drise_path = det.get("drise_asset")
    if drise_path and (det.get("drise_weight_sum") or 0.0) > 0:
        panels.append((np.array(Image.open(run_dir / drise_path)), "D-RISE"))
    gradcam_path = det.get("gradcam_asset")
    if (
        gradcam_path
        and det.get("gradcam_letterbox_content_xyxy") is not None
        and (det.get("gradcam_has_signal") is None or bool(det.get("gradcam_has_signal")))
    ):
        panels.append((np.array(Image.open(run_dir / gradcam_path)), "Grad-CAM"))

    _cols = len(panels)
    _fig, _axes = plt.subplots(1, _cols, figsize=(7 * _cols, 6))
    if _cols == 1:
        _axes = [_axes]
    for _ax, (_img, _title) in zip(_axes, panels):
        _ax.imshow(_img)
        _ax.set_title(_title)
        _ax.axis("off")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(det, detection_detail_visible, mo):
    mo.stop(not detection_detail_visible(), mo.md(""))

    object_reliance = det["mask_drop"]
    context_reliance = det["context_drop"]
    if object_reliance > context_reliance + 0.1:
        interpretation = "This detection appears more object-driven than context-driven."
    elif context_reliance > object_reliance + 0.1:
        interpretation = "This detection appears to rely heavily on surrounding context."
    else:
        interpretation = "This detection uses both object pixels and context to a similar degree."

    mo.callout(
        mo.md(
            f"""
            **Interpretation**

            {interpretation}

            `mask_drop` measures confidence lost when the detected object area is hidden.
            `context_drop` measures confidence lost when only the object remains and background is removed.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(detection_detail_visible, image_rows, mo):
    mo.stop(not detection_detail_visible(), mo.md(""))

    columns = [
        "detection_id",
        "class_name",
        "original_conf",
        "crop_score_same_class",
        "object_only_score",
        "object_masked_score",
        "mask_drop",
        "context_drop",
        "occlusion_max_drop",
    ]
    for optional in ["gt_status", "fp_type", "matched_gt_iou", "best_gt_class_name", "best_gt_iou"]:
        if optional in image_rows.columns:
            columns.append(optional)

    mo.md("## All Detections In This Image")
    mo.ui.table(image_rows.select(columns))
    return


if __name__ == "__main__":
    app.run()
