# DINOv2 -> detector fine-tuning path

This repository is centered on Ultralytics training loops. That matters because a **true DINOv2 backbone swap is not a small patch** here:

- `DINOv2` is a ViT family model.
- The current `YOLO` models in this repo use CNN backbones.
- Ultralytics `RT-DETR` is more detector-compatible for this workflow than the existing YOLO training path, but its default backbone is still not DINOv2.

So the practical path in this repo is:

1. **Stage 1: learn or obtain DINOv2 features/checkpoint externally**
2. **Stage 2: fine-tune a detector here**

## What "Stage 1" should mean

If your goal is "recognize shapes on objects", DINOv2 is useful because its features tend to preserve object structure better than standard supervised CNN pretraining.

Use one of these two realistic options:

- Use an existing DINOv2 checkpoint as the representation source for a detector experiment outside Ultralytics.
- Use DINOv2 upstream for dataset curation, tile ranking, retrieval, or pseudo-label support, then fine-tune the detector here on the improved labeled set.

## What this project can do today

The `ssl-rtdetr-finetune` entrypoint provides RT-DETR as an alternative detector
baseline. It does not load DINOv2 weights and must not be reported as a DINOv2
experiment.

Run it like this:

```bash
uv run ssl-rtdetr-finetune \
  --data /media/datasets/object_detection/cur_train_3_natural_disasters/data.yaml \
  --model rtdetr-l.pt \
  --device 0 \
  --batch 32 \
  --imgsz 640
```

Recommended defaults for shape-sensitive fine-tuning:

- Keep augmentation moderate.
- Disable heavy `mosaic`/`mixup` at the start.
- Prefer `AdamW`.
- Start from a pretrained detector checkpoint, not from scratch.

## What is *not* implemented here

This repo does **not** yet inject DINOv2 weights directly into the Ultralytics detector backbone. That would require one of these larger changes:

- replacing the detector backbone with a ViT-compatible one,
- adding a feature adapter from DINOv2 tokens to detector pyramid features,
- or moving the experiment to a framework where DINOv2-based detectors are already first-class.

## Recommended next step if you want true DINOv2 usage

If you want the detector itself to consume DINOv2 features, the next engineering step is:

1. add a separate experiment path for a ViT-based detector,
2. load a local DINOv2 checkpoint there,
3. adapt token outputs into multi-scale detector features,
4. then fine-tune on your labeled dataset.

That is feasible, but it is a dedicated model-integration task, not a training-flag change.
