# Object Detection SSL

Standalone research framework for reproducible low-label object-detection SSL studies.
It owns the VICReg implementations, detector fine-tuning, shape-reliance benchmark,
study orchestration, exact data manifests, metrics, and statistical comparisons.

## What is included

- Declarative YAML studies and Cartesian experiment matrices.
- Deterministic nested labeled splits and unlabeled scaling manifests.
- Object-centric and shape-aware VICReg implementations for YOLO backbones and trunks.
- Detector fine-tuning with verified SSL checkpoint transfer.
- Geometry-preserving appearance and appearance-preserving shape corruptions.
- Per-detection Grad-CAM, D-RISE, occlusion, and counterfactual explanations.
- Robustness, calibration, dataset-audit, and failure-clustering reports.
- Built-in methods: supervised; global, object, and mixed VICReg; uniform-contour,
  contour-region, and mismatched-contour VICReg controls.
- Custom SSL commands and custom evaluators through tokenized command templates.
- Physical GPU pinning with logical device remapping.
- Sequential, resumable pretraining, fine-tuning, and evaluation stages.
- Immutable configuration snapshots and project Git-commit provenance.
- Independent clean validation, shape/appearance reliance summaries, decision
  criteria, paired image bootstrap, and multi-seed Student-t intervals.
- Supervised object-driven, object-crop, and stylization data interventions.
- SwAV-ResNet50/YOLO and RT-DETR architecture baselines.

## Setup

```bash
cd /home/dmytrob/Projects/experiments
uv sync --dev
uv run sslx validate configs/studies/shape_region_1pct.yaml
```

The supplied study uses only code and manifests in this repository. Raw images remain
external under `/media/datasets/object_detection`; edit only the YAML when changing
data, model, compute, methods, or metrics.

## Code layout

```text
src/ssl_experiments/
  methods/vicreg.py                 SSL datasets, encoders, losses, transfer
  training/pretrain_vicreg.py      resumable VICReg pretraining
  training/finetune_detector.py    supervised and SSL-initialized fine-tuning
  evaluation/shape_reliance.py     corruption definitions and reliance scores
  evaluation/shape_benchmark.py    full detector benchmark CLI
  evaluation/clean.py              independent clean validation
  evaluation/metrics.py            detection and mAP metrics
  explainability/core.py           attribution and diagnostic implementations
  explainability/cli.py            report-generation CLI
  data/splits.py                    deterministic low-label split generation
  data/manifests.py                 unlabeled scaling-manifest generation
  data/interventions/               supervised data-intervention controls
  matrix.py                         multi-seed/fraction orchestration and analysis
  pipeline.py                       resumable study execution
notebooks/vis_explainability.py     interactive report explorer
data/
  manifests/                       fixed 50k and 100k pretraining samples
  splits/                           1/5/10/25% splits for seeds 0, 1, and 2
```

Normal studies should use `sslx`, which supplies consistent arguments and artifact
locations to the low-level training and evaluation entrypoints.

## Reproducible data

Recreate the complete nested split set from a YOLO `data.yaml`:

```bash
uv run ssl-splits \
  /media/datasets/object_detection/cur_train_3_natural_disasters/data.yaml \
  --output data/splits \
  --fractions 0.01 0.05 0.10 0.25 \
  --seeds 0 1 2
```

Generate compute-accounted nested pretraining manifests:

```bash
uv run ssl-manifests data/splits/population.txt \
  --output data/manifests \
  --sizes 50000 100000 \
  --epochs 100 50 \
  --batch 128 --seed 0
```

Existing files are verified byte-for-byte. Use `--regenerate` only when replacing a
frozen split intentionally. The committed metadata records source and manifest hashes,
image counts, label-file coverage, batches per epoch, and optimizer-step budgets.

## Explainability

Generate explanations for one image or a directory:

```bash
uv run ssl-explain path/to/images \
  --weights artifacts/shape_region_1pct/finetuning/region_aligned/weights/best.pt \
  --labels path/to/labels \
  --device 0 \
  --max-images 10
```

The command writes image-level and detection-level Parquet tables, raw candidate and
postprocessing traces, Grad-CAM/D-RISE/occlusion assets, counterfactual results,
robustness and sanity checks, calibration plots, failure clusters, and a Markdown
analysis report under `artifacts/explainability/<run_name>/`.

Launch the interactive report explorer:

```bash
notebooks/vis_explainability.sh
```

## Commands

Inspect every command before using GPU time:

```bash
uv run sslx plan configs/studies/shape_region_1pct.yaml
uv run sslx plan configs/studies/shape_region_1pct.yaml \
  --methods global_vicreg,region_aligned \
  --stages pretrain,finetune
```

Run selected stages or the complete study:

```bash
uv run sslx run configs/studies/shape_region_1pct.yaml \
  --methods region_aligned --stages pretrain,finetune,evaluate

uv run sslx run configs/studies/shape_region_1pct.yaml --analyze
uv run sslx status configs/studies/shape_region_1pct.yaml
uv run sslx analyze configs/studies/shape_region_1pct.yaml
```

Long jobs can run in tmux:

```bash
scripts/run_study_tmux.sh configs/studies/shape_region_1pct.yaml shape_region_gpu0
```

The runner skips completed stages, resumes SSL from `latest.pt`, resumes detector
training from `weights/last.pt`, and refuses ambiguous partial directories.

## Artifact layout

```text
artifacts/<study>/
  study.yaml                  frozen source configuration
  study.json                  configuration hash and Git provenance
  study.lock
  pretraining/<method>/
  finetuning/<method>/
  evaluations/<evaluator>/<method>/
  logs/<method>/<stage>.log
  state/<method>/<stage>.json
  analysis/summary.{json,csv,md}
```

Do not edit a launched study configuration. Copy it and change `name`; the runner
rejects a changed configuration hash for an existing artifact directory.

## Low-label matrix

The supplied matrix is 4 label fractions x 3 seeds x 4 initializations. SSL
pretraining is performed once per method and seed, then reused across the four label
fractions. Render and inspect it before launch:

```bash
uv run sslx matrix-plan configs/matrices/low_label_vicreg.yaml
uv run sslx matrix-status configs/matrices/low_label_vicreg.yaml
scripts/run_matrix_tmux.sh \
  configs/matrices/low_label_vicreg.yaml low_label_gpu0
```

After all cells finish:

```bash
uv run sslx matrix-analyze configs/matrices/low_label_vicreg.yaml
```

The analysis writes raw runs, per-fraction means and 95% Student-t intervals, and
paired method-minus-supervised deltas under
`artifacts/low_label_vicreg/analysis/`.

## Adding SSL methods

For current VICReg variants, add a method entry and override only changed arguments:

```yaml
- name: region_weight_02
  kind: vicreg_shape_region
  pretraining:
    args:
      shape_loss_weight: 0.2
      shape_region_fraction: 0.25
```

For DINO, SimCLR, SwAV, or another implementation, use `kind: custom`. Supported
template fields are `{python}`, `{project_root}`, `{repo}`, `{artifact_root}`, `{method}`,
`{pretrain_dir}`, `{checkpoint}`, `{finetune_dir}`, `{weights}`, and
`{evaluation_dir}`. See `configs/custom_method.example.yaml`.

The custom checkpoint must be compatible with this project's `--ssl-checkpoint`
loader, or the method can override the fine-tuning settings.

`vicreg_global` and the contour methods are label-free. `vicreg_object` and
`vicreg_mixed` read bounding boxes to select crops, so they are label-guided controls,
not strictly unlabeled SSL. Keep that distinction explicit in paper claims.

## Intervention controls

The dataset commands create new YOLO datasets and leave the source untouched:

```bash
uv run ssl-object-augment --help
uv run ssl-object-crops --help
uv run ssl-stylize --help
```

Alternative architecture baselines are available separately:

```bash
uv run ssl-swav-finetune --help
uv run ssl-rtdetr-finetune --help
```

Install `uv sync --extra tracking` only when MLflow dataset logging is needed.

## Adding evaluators

Append a custom evaluator to the study:

```yaml
evaluations:
  custom:
    - name: domain_shift
      command:
        - "{python}"
        - "{project_root}/scripts/evaluate_domain.py"
        - --weights
        - "{weights}"
        - --output
        - "{evaluation_dir}"
      completion: "{evaluation_dir}/report.json"
```

Every evaluator runs on every selected method and receives its own output directory.

## Paired bootstrap

The shape benchmark stores per-image TP/FP/FN rows. Compare any two completed reports:

```bash
uv run sslx bootstrap \
  artifacts/shape_region_1pct/evaluations/shape_reliance/region_aligned \
  artifacts/shape_region_1pct/evaluations/shape_reliance/region_mismatched \
  --left-name region_aligned --right-name region_mismatched \
  --output artifacts/shape_region_1pct/analysis/bootstrap_aligned_vs_mismatched
```

The bootstrap reports confidence intervals for thresholded precision, recall, F1,
and recall-based reliance. Exact mAP intervals require confidence-ranked predictions,
which the current benchmark does not retain.

## Research notes

- `docs/research/experiment_design.md`: hypothesis, controls, and interpretation.
- `docs/research/object_centric_ssl.md`: label-guided object-view experiment.
- `docs/research/swav_baseline.md`: SwAV architecture-control scope.
- `docs/research/dinov2_scope.md`: what would count as an actual DINOv2 baseline.
- `docs/guides/ssod_interactive_tutorial.html`: standalone SSOD method tutorial.
- `docs/migration_scope.md`: retained research scope and intentional exclusions.

This repository intentionally contains no migrated historical checkpoints, run
directories, metric tables, plots, or videos. New artifacts are created under the
ignored `artifacts/` directory.
