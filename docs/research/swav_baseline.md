# SwAV -> YOLO path

This repository can use `SwAV` more naturally than `DINOv2` because SwAV is commonly used with `ResNet` backbones, and Ultralytics supports `TorchVision` backbones in model YAMLs.

Relevant Ultralytics docs for this pattern:

- TorchVision backbones with multi-scale outputs:
  https://docs.ultralytics.com/guides/model-yaml-config/

## Recommended path in this repo

Use a detector with a `ResNet50` backbone, initialize that backbone from a `SwAV` checkpoint, then fine-tune for detection.

Standalone-project files:

- Model: `src/ssl_experiments/models/swav_resnet50_yolo11.yaml`
- Training entrypoint: `ssl-swav-finetune`

## Why this path

- Official and common `SwAV` checkpoints are typically `ResNet50`.
- Your current `yolo26s` setup does not expose a clean `ResNet` backbone transfer path.
- Ultralytics supports `TorchVision` backbone outputs plus `Index` selection for `P3/P4/P5` detection features.

## Training command

If you already have a local SwAV checkpoint:

```bash
uv run ssl-swav-finetune \
  --data /media/datasets/object_detection/cur_train_3_natural_disasters/data.yaml \
  --swav-ckpt /path/to/swav_resnet50_checkpoint.pth \
  --device 0 \
  --batch 32 \
  --imgsz 640
```

If you do not pass `--swav-ckpt`, the model falls back to torchvision default pretrained weights.

## Checkpoint assumptions

The loader is built for common self-supervised checkpoint layouts such as:

- official `facebookresearch/swav` ResNet50 checkpoints in torchvision-style format,
- `state_dict`
- `model`
- keys prefixed with `module.`
- keys prefixed with `backbone.`
- keys prefixed with `encoder_q.`

It deliberately skips projection/prototype/classification layers and loads only the ResNet backbone weights.

The official SwAV repo says its released `ResNet-50` models are in `torchvision format`, and its `torch.hub` loader removes `module.` before calling `load_state_dict(..., strict=False)`.
Source: https://github.com/facebookresearch/swav and https://github.com/facebookresearch/swav/blob/main/hubconf.py

## Caveats

- This is a practical `SwAV -> detector` workflow, not a guarantee that every SwAV checkpoint format will match.
- The feature pyramid here is taken directly from `ResNet50` stages. It is a reasonable baseline, not a tuned architecture.
- If your unlabeled data is highly domain-specific, pretraining your own `SwAV ResNet50` on that data is better than using a generic checkpoint.
