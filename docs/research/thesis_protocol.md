# Thesis training and evaluation recipe

`configs/studies/thesis_vicreg.yaml` is a new, configurable implementation of the
documented training and evaluation scheme. It does not recover the code,
hyperparameters, thresholds, manifests, or results of historical runs. Existing
low-label/contour studies remain separate experiments with their own controls.

## Training

All three methods start from `yolo26n.pt` COCO weights. Only training images enter
VICReg. Standard VICReg uses independent RGB views. Improved VICReg uses one of
ten coupled groups in `methods/opposed_views.py`; bbox annotations guide the
object-dependent operations. Thus that pretraining is label-guided. Its criterion
remains the standard global VICReg loss, without an auxiliary contour loss.
Transformation magnitudes in the code are new defaults, not verified historical
parameters. The group selector makes no claim about historical group frequencies.

Both VICReg detectors freeze the entire transferred backbone during fine-tuning.
The freezer disables parameter gradients and holds BatchNorm running statistics
fixed. The neck and detection head remain trainable. Baseline uses normal YOLO
training; standard VICReg uses normal YOLO augmentations for fine-tuning.

The improved method uses a separately prepared custom augmentation dataset and
`augmentation_profile: prepared`, which disables additional random YOLO training
augmentations. Prepare that dataset with the corrected `ssl-object-augment`
generator or supply your verified custom pipeline's output. The bundled generator
supports Copy-Paste, background dropout, object crops and photometric corruption;
it is **not** a claim that the full historical fine-tuning pipeline was recovered.
Consequently, the comparison measures the complete scheme rather than the isolated
effect of opposed pairing. A pair-only ablation must use identical fine-tuning
data and augmentation profiles for both VICReg methods.

Set these environment variables to real absolute paths before planning a run:

| Variable | Meaning |
|---|---|
| `SSL_PYTHON` | Interpreter containing the installed package and dependencies |
| `THESIS_TRAIN_IMAGES`, `THESIS_TRAIN_LABELS` | Prepared training tiles and labels |
| `THESIS_TRAIN_DATA` | Original training YOLO YAML |
| `THESIS_IMPROVED_DATA` | Separately generated custom augmentation YOLO YAML |
| `THESIS_VAL_IMAGES`, `THESIS_VAL_LABELS` | General validation images and labels |
| `THESIS_OBJECTNESS_IMAGES`, `THESIS_OBJECTNESS_LABELS` | Separate fixed objectness sample |

Inspect all settings, including epochs, batch size, augmentation magnitudes,
and SAHI tile/overlap sizes: these are configurable defaults, not recovered values.

```bash
sslx plan configs/studies/thesis_vicreg.yaml --stages pretrain,finetune
sslx run configs/studies/thesis_vicreg.yaml --stages pretrain,finetune
```

## Joint evaluation

```bash
sslx thesis-evaluate configs/studies/thesis_vicreg.yaml --phase calibration
sslx thesis-evaluate configs/studies/thesis_vicreg.yaml --phase objectness
```

Calibration uses the complete supplied general-validation directory with SAHI.
Each checkpoint receives its own confidence threshold maximizing micro F1; equal
confidence scores stay together and F1 ties select the higher threshold. The
report saves operating-point precision/recall/F1/TP/FP/FN and candidate-ranked mAP.
The confidence floor bounds which predictions can be considered. These are
validation-selected operating points, not independently selected test thresholds.

Objectness requires exactly the configured sample size (2000 by default), and
refuses missing annotation files. It runs every clean checkpoint first, intersects
the correctly detected `(relative image path, GT row index)` targets across all
models, and uses that same nonempty intersection for every intervention. A target
lost after intervention receives confidence and IoU zero. Newly detected objects
outside the shared set cannot increase retention. Targets, raw predictions,
image/label hashes, checkpoint hashes and thresholds are saved for audit.

Both stages use the same SAHI and matching settings. The confidence selected on
general validation remains fixed per model across all behavior tests. Changed
checkpoint hashes or inference settings invalidate existing calibration. No
automatic threshold fitting occurs on the behavior sample.

The included behavior modes cover color, texture, patch permutation,
background-swap, object-only, and object-removed. Object-only retains target bbox
pixels on black; object-removed uses inpainting and reports suppression separately
as a diagnostic. Donor images may contain objects. Contour/silhouette-only is not
included. These choices are explicit implementations, not evidence that historical
interventions were byte-identical. No confidence intervals or statistical
significance are claimed by this evaluator.

Outputs go to `artifacts/thesis_vicreg/thesis_evaluation/`. Existing completed
reports require an explicit `--overwrite` to replace them. No training or evaluation
results are supplied by this recipe.
