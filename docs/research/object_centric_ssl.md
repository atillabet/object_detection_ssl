# Object-centric SSL experiment

## Research question

Can self-supervised pretraining on object-centered views improve detection under
texture and context changes without reducing clean detection accuracy?

The implementation uses the same `yolo26n` backbone for every controlled run.
This avoids the architecture confound introduced by comparing the existing YOLO
detector with a ResNet-50 SwAV detector.

Object and mixed sampling use YOLO boxes to select the crop, but never use class
labels as learning targets. This should be described as **label-guided
object-centric SSL**, not completely label-free SSL. A later experiment can
replace ground-truth boxes with unsupervised proposals or pseudo-boxes.

## Pretraining conditions

Run at least these three initializations with identical seeds, image counts,
epochs, optimizer settings, and projection dimensions:

| Initialization | Sampling | Purpose |
|---|---|---|
| Random | none | Lower-bound control |
| Global VICReg | `global` | Standard domain SSL control |
| Object-centric VICReg | `mixed` | Proposed method |

For the proposed run, `75%` of samples use an object-centered view and `25%` use
a global view. The two object views independently change the amount of visible
context. Color jitter, grayscale, blur, and solarization encourage appearance
invariance. Object views are resized without an additional spatial crop so the
selected object's shape remains present.

Images without a valid box use a global-view fallback in both object and mixed
conditions. They are not removed, so the manifest size and optimizer-step budget
stay matched to global VICReg.

## Pilot pretraining

Start with 25 epochs. This is long enough to reject a failed representation
before committing to the final schedule.

```bash
uv run ssl-pretrain \
  --sampling mixed \
  --object-probability 0.75 \
  --model yolo26n.yaml \
  --epochs 25 \
  --batch 128 \
  --imgsz 224 \
  --device 0 \
  --name yolo26n_vicreg_mixed_e25
```

Run the global SSL control with the same settings:

```bash
uv run ssl-pretrain \
  --sampling global \
  --model yolo26n.yaml \
  --epochs 25 \
  --batch 128 \
  --imgsz 224 \
  --device 0 \
  --name yolo26n_vicreg_global_e25
```

## Detection fine-tuning

Fine-tune from the matching model YAML so only the initialization differs:

```bash
uv run ssl-finetune \
  --model yolo26n.yaml \
  --ssl-checkpoint artifacts/pretraining/yolo26n_vicreg_mixed_e25/latest.pt \
  --data /media/datasets/object_detection/cur_train_3_natural_disasters/data.yaml \
  --epochs 100 \
  --imgsz 320 \
  --batch 256 \
  --device 0 \
  --seed 0 \
  --name yolo26n_vicreg_mixed_e25_finetune
```

Use identical fine-tuning arguments for random, global SSL, and object-centric
SSL runs. Repeat the final comparison for at least three seeds.

## Evaluation criteria

Evaluate every resulting detector with the same V2 feature-reliance benchmark.
The proposed method succeeds only if it satisfies both parts:

1. Clean mAP50-95 is maintained or improved.
2. Texture retention and context-swap retention improve relative to both random
   initialization and global SSL.

Shape-corruption sensitivity is not itself a robustness target: destroying
shape should affect a shape-using detector. The useful target is better
retention when appearance or context changes while object shape is preserved.
