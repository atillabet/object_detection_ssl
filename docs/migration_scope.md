# Migration scope

This project contains the reusable research code needed to reproduce and extend
the low-label object-detection SSL study. It intentionally does not mirror the
old object-detector repository.

## Included

- VICReg pretraining and verified YOLO backbone/trunk transfer;
- supervised detector fine-tuning and clean validation;
- shape-reliance, detection-metric, SAHI, and bootstrap evaluation;
- Grad-CAM, D-RISE, occlusion, counterfactual, robustness, calibration, and
  failure-analysis explainability;
- deterministic labeled splits and unlabeled scaling manifests;
- multi-seed/fraction matrix orchestration and statistical aggregation;
- object-driven, object-centered, and stylized supervised controls;
- SwAV-ResNet50/YOLO and RT-DETR alternative detector baselines;
- research protocols and the SSOD tutorial.

## Excluded

- historical metrics, checkpoints, logs, plots, run directories, and videos;
- ONNX, OpenVINO, NCNN, RKNN, and other deployment/conversion tools;
- video ROI/telemetry rendering and tracking-specific workflows;
- Label Studio conversion, tiling, hard-negative mining, and one-off dataset
  repair scripts;
- vendored Ultralytics source and model binaries.

Those exclusions keep experiment claims, reusable controls, and evaluation code
separate from deployment work and dataset-specific maintenance.
