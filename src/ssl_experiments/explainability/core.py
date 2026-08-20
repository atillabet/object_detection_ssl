from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops

from ssl_experiments.data.yolo import load_yolo_labels, parse_yolo_yaml
from ssl_experiments.evaluation.sahi import sahi_inference_batched


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_PREDICT_BATCH_SIZE = 32
DEFAULT_SAHI_BATCH_SIZE = 128


@dataclass
class Detection:
    detection_id: int
    cls_id: int
    cls_name: str
    conf: float
    box_xyxy: np.ndarray


@dataclass
class GroundTruth:
    gt_id: int
    cls_id: int
    cls_name: str
    box_xyxy: np.ndarray


@dataclass
class Candidate:
    branch: str
    candidate_id: int
    cls_id: int
    cls_name: str
    score: float
    box_xyxy: np.ndarray
    objectness: float | None = None
    stride: int | None = None
    feature_level: int | None = None
    grid_x: int | None = None
    grid_y: int | None = None
    kept_after_threshold: bool = False
    kept_after_nms: bool = False
    final_detection_id: int | None = None
    matched_final_iou: float = 0.0
    suppressed_by_candidate_id: int | None = None
    suppressed_by_iou: float | None = None


@dataclass
class GradcamTarget:
    anchor_idx: int
    target_score: torch.Tensor
    anchor_iou: float
    branch: str
    feature_level: int | None = None
    stride: int | None = None
    grid_x: int | None = None
    grid_y: int | None = None


@dataclass
class AdapterInference:
    candidates: list[Candidate]
    final_detections: list[Detection]
    final_candidate_matches: dict[int, Candidate]
    raw_output_type: str


@dataclass
class CounterfactualVariant:
    name: str
    image_bgr: np.ndarray
    target_box: np.ndarray
    asset_name: str


@dataclass
class RobustnessVariant:
    transform: str
    image_bgr: np.ndarray
    target_box: np.ndarray


def find_images(path: Path, max_images: int | None = None) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_SUFFIXES else []
    images = [p for p in sorted(path.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]
    if max_images is not None:
        return images[:max_images]
    return images


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def clamp_box(box_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    box = np.array(box_xyxy, dtype=np.float32).copy()
    box[0::2] = np.clip(box[0::2], 0, width - 1)
    box[1::2] = np.clip(box[1::2], 0, height - 1)
    return box


def box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def expand_box(box_xyxy: np.ndarray, width: int, height: int, margin: float) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy.astype(np.float32)
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * margin
    pad_y = bh * margin
    return clamp_box(np.array([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]), width, height)


def draw_box(image_rgb: np.ndarray, box_xyxy: np.ndarray, label: str, color: tuple[int, int, int]) -> np.ndarray:
    out = image_rgb.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.putText(out, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def render_all_detections(image_rgb: np.ndarray, detections: list[Detection]) -> np.ndarray:
    colors = [(230, 57, 70), (29, 53, 87), (69, 123, 157), (42, 157, 143), (233, 196, 106)]
    out = image_rgb.copy()
    for det in detections:
        label = f"{det.detection_id}: {det.cls_name} {det.conf:.3f}"
        out = draw_box(out, det.box_xyxy, label, colors[det.cls_id % len(colors)])
    return out



def class_name(names: dict[int, str] | list[str], cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def model_input_size(model: YOLO, fallback: int = 640) -> int:
    size = model.overrides.get("imgsz", fallback)
    if isinstance(size, (list, tuple)):
        return int(size[0])
    return int(size)


def torch_device_from_option(device: str | int | None) -> str | None:
    if device is None:
        return None
    if isinstance(device, int):
        return f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    value = str(device)
    if value.isdigit():
        return f"cuda:{value}" if torch.cuda.is_available() else "cpu"
    return value


def ensure_model_float32(model: YOLO, device: str | int | None = None) -> None:
    requested_device = torch_device_from_option(device)
    target_device = torch.device(requested_device) if requested_device is not None else next(model.model.parameters()).device
    model.model.to(target_device).float()
    predictor = getattr(model, "predictor", None)
    if predictor is not None and hasattr(predictor, "args"):
        predictor.args.half = False
    backend = getattr(predictor, "model", None) if predictor is not None else None
    if backend is not None and hasattr(backend, "fp16"):
        backend.fp16 = False


def reset_detector_inference_cache(model: YOLO) -> None:
    """Force anchors to be rebuilt outside Ultralytics inference mode."""
    head = model.model.model[-1] if hasattr(model.model, "model") else None
    if head is not None and hasattr(head, "shape"):
        head.shape = None


def capture_detector_head_state(model: YOLO) -> tuple[Any, Any, Any, Any, Any]:
    head = model.model.model[-1] if hasattr(model.model, "model") else None
    return (
        head,
        getattr(head, "cv2", None),
        getattr(head, "cv3", None),
        getattr(head, "one2one_cv2", None),
        getattr(head, "one2one_cv3", None),
    )


def restore_detector_head_state(state: tuple[Any, Any, Any, Any, Any]) -> None:
    head, cv2, cv3, one2one_cv2, one2one_cv3 = state
    if head is None:
        return
    if cv2 is not None:
        head.cv2 = cv2
    if cv3 is not None:
        head.cv3 = cv3
    if one2one_cv2 is not None:
        head.one2one_cv2 = one2one_cv2
    if one2one_cv3 is not None:
        head.one2one_cv3 = one2one_cv3


def infer_labels_dir(image_path: Path, labels_dir: Path | None) -> Path | None:
    if labels_dir is not None:
        return labels_dir
    if image_path.parent.name == "images":
        sibling = image_path.parent.parent / "labels"
        if sibling.is_dir():
            return sibling
    return None


def load_ground_truths(
    image_path: Path,
    image_shape: tuple[int, int],
    names: dict[int, str] | list[str],
    labels_dir: Path | None,
) -> tuple[list[GroundTruth], Path | None]:
    resolved_labels_dir = infer_labels_dir(image_path, labels_dir)
    if resolved_labels_dir is None:
        return [], None

    height, width = image_shape
    label_path = resolved_labels_dir / f"{image_path.stem}.txt"
    boxes, classes = load_yolo_labels(label_path, width, height)
    ground_truths = [
        GroundTruth(
            gt_id=idx,
            cls_id=int(cls_id),
            cls_name=class_name(names, int(cls_id)),
            box_xyxy=np.array(box, dtype=np.float32),
        )
        for idx, (box, cls_id) in enumerate(zip(boxes, classes, strict=False), start=1)
    ]
    return ground_truths, label_path


def render_ground_truth(image_rgb: np.ndarray, ground_truths: list[GroundTruth]) -> np.ndarray:
    out = image_rgb.copy()
    for gt in ground_truths:
        out = draw_box(out, gt.box_xyxy, f"gt{gt.gt_id}: {gt.cls_name}", (42, 157, 143))
    return out


def render_predictions_vs_ground_truth(
    image_rgb: np.ndarray,
    detections: list[Detection],
    ground_truths: list[GroundTruth],
    gt_matches: dict[int, dict[str, Any]],
    false_negative_ids: set[int],
) -> np.ndarray:
    out = image_rgb.copy()
    for gt in ground_truths:
        color = (233, 196, 106) if gt.gt_id in false_negative_ids else (42, 157, 143)
        out = draw_box(out, gt.box_xyxy, f"gt{gt.gt_id}: {gt.cls_name}", color)
    for det in detections:
        match = gt_matches.get(det.detection_id, {})
        status = match.get("gt_status")
        color = (42, 157, 143) if status == "tp" else (230, 57, 70)
        label = f"{status or 'pred'} {det.detection_id}: {det.cls_name} {det.conf:.2f}"
        out = draw_box(out, det.box_xyxy, label, color)
    return out


def match_detections_to_ground_truth(
    detections: list[Detection],
    ground_truths: list[GroundTruth],
    iou_threshold: float,
) -> tuple[dict[int, dict[str, Any]], list[GroundTruth], dict[str, Any]]:
    matched_gt_ids: set[int] = set()
    matches: dict[int, dict[str, Any]] = {}

    for det in detections:
        best_same_iou = 0.0
        best_same_gt: GroundTruth | None = None
        best_any_iou = 0.0
        best_any_gt: GroundTruth | None = None

        for gt in ground_truths:
            iou = box_iou_xyxy(det.box_xyxy, gt.box_xyxy)
            if iou > best_any_iou:
                best_any_iou = iou
                best_any_gt = gt
            if gt.cls_id == det.cls_id and gt.gt_id not in matched_gt_ids and iou > best_same_iou:
                best_same_iou = iou
                best_same_gt = gt

        status = "fp"
        fp_type = "background"
        matched_gt = None
        matched_iou = 0.0
        if best_same_gt is not None and best_same_iou >= iou_threshold:
            status = "tp"
            fp_type = None
            matched_gt = best_same_gt
            matched_iou = best_same_iou
            matched_gt_ids.add(best_same_gt.gt_id)
        elif best_any_gt is not None:
            if best_any_iou >= iou_threshold and best_any_gt.cls_id == det.cls_id:
                fp_type = "duplicate"
            elif best_any_iou >= iou_threshold:
                fp_type = "class_confusion"
            elif best_any_iou >= 0.1:
                fp_type = "localization"

        matches[det.detection_id] = {
            "gt_status": status,
            "fp_type": fp_type,
            "matched_gt_id": matched_gt.gt_id if matched_gt is not None else None,
            "matched_gt_class_id": matched_gt.cls_id if matched_gt is not None else None,
            "matched_gt_class_name": matched_gt.cls_name if matched_gt is not None else None,
            "matched_gt_iou": float(matched_iou),
            "localization_error": float(1.0 - matched_iou) if status == "tp" else None,
            "best_gt_id": best_any_gt.gt_id if best_any_gt is not None else None,
            "best_gt_class_id": best_any_gt.cls_id if best_any_gt is not None else None,
            "best_gt_class_name": best_any_gt.cls_name if best_any_gt is not None else None,
            "best_gt_iou": float(best_any_iou),
        }

    false_negatives = [gt for gt in ground_truths if gt.gt_id not in matched_gt_ids]
    tp = sum(1 for match in matches.values() if match["gt_status"] == "tp")
    fp = sum(1 for match in matches.values() if match["gt_status"] == "fp")
    fn = len(false_negatives)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-6) if precision + recall > 0 else 0.0
    summary = {
        "num_ground_truth": len(ground_truths),
        "num_tp": tp,
        "num_fp": fp,
        "num_fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    return matches, false_negatives, summary


def build_letterboxed_tensor(model: YOLO, image_bgr: np.ndarray, imgsz: int, device: str | int | None) -> tuple[torch.Tensor, tuple[int, int]]:
    ensure_model_float32(model, device)
    stride = int(model.model.stride.max().item()) if hasattr(model.model, "stride") else 32
    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=stride)
    transformed = letterbox(image=image_bgr)
    tensor = torch.from_numpy(np.ascontiguousarray(transformed[..., ::-1].transpose(2, 0, 1)))
    tensor = tensor.unsqueeze(0).float() / 255.0
    requested_device = torch_device_from_option(device)
    target_device = torch.device(requested_device) if requested_device is not None else next(model.model.parameters()).device
    tensor = tensor.to(target_device)
    return tensor, transformed.shape[:2]


def letterbox_content_region(
    original_shape: tuple[int, int],
    input_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    original_h, original_w = original_shape
    input_h, input_w = input_shape
    gain = min(input_h / original_h, input_w / original_w)
    content_w = int(round(original_w * gain))
    content_h = int(round(original_h * gain))
    pad_w = input_w - content_w
    pad_h = input_h - content_h
    left = int(round(pad_w / 2 - 0.1))
    top = int(round(pad_h / 2 - 0.1))
    right = min(input_w, left + content_w)
    bottom = min(input_h, top + content_h)
    return left, top, right, bottom


def resize_letterboxed_heatmap_to_original(
    heatmap: np.ndarray,
    input_shape: tuple[int, int],
    original_shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    input_h, input_w = input_shape
    original_h, original_w = original_shape
    input_heatmap = cv2.resize(heatmap, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    left, top, right, bottom = letterbox_content_region(original_shape, input_shape)
    content_heatmap = input_heatmap[top:bottom, left:right]
    if content_heatmap.size == 0:
        content_heatmap = input_heatmap
    output_heatmap = cv2.resize(content_heatmap, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    if output_heatmap.max() > 0:
        output_heatmap = output_heatmap / output_heatmap.max()
    return output_heatmap, (left, top, right, bottom)


def run_predictions(
    model: YOLO,
    image_bgr: np.ndarray,
    imgsz: int,
    conf: float,
    iou: float,
    device: str | int | None,
    max_det: int,
    use_sahi: bool = False,
    slice_size: int = 320,
    overlap: int = 64,
    sahi_batch_size: int = DEFAULT_SAHI_BATCH_SIZE,
    large_conf: float = 0.25,
    size_threshold: int = 16,
    sahi_model_input_size: int | None = None,
) -> list[Detection]:
    if use_sahi:
        return run_predictions_sahi(
            model=model,
            image_bgr=image_bgr,
            conf=conf,
            iou=iou,
            max_det=max_det,
            slice_size=slice_size,
            overlap=overlap,
            batch_size=sahi_batch_size,
            large_conf=large_conf,
            size_threshold=size_threshold,
            sahi_model_input_size=sahi_model_input_size,
        )
    return run_predictions_batch(model, [image_bgr], imgsz, conf, iou, device, max_det, batch_size=1)[0]


def detections_from_arrays(
    boxes: list[list[float]] | np.ndarray,
    scores: list[float] | np.ndarray,
    classes: list[int] | np.ndarray,
    names: dict[int, str] | list[str],
    image_shape: tuple[int, int],
    max_det: int,
) -> list[Detection]:
    height, width = image_shape
    rows = sorted(
        zip(boxes, scores, classes, strict=False),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    detections = []
    for idx, (box, score, cls_id) in enumerate(rows[:max_det], start=1):
        box_xyxy = clamp_box(np.array(box, dtype=np.float32), width, height)
        if box_xyxy[2] <= box_xyxy[0] or box_xyxy[3] <= box_xyxy[1]:
            continue
        cls_id = int(cls_id)
        detections.append(
            Detection(
                detection_id=idx,
                cls_id=cls_id,
                cls_name=class_name(names, cls_id),
                conf=float(score),
                box_xyxy=box_xyxy,
            )
        )
    return detections


def run_predictions_sahi(
    model: YOLO,
    image_bgr: np.ndarray,
    conf: float,
    iou: float,
    max_det: int,
    slice_size: int,
    overlap: int,
    batch_size: int,
    large_conf: float,
    size_threshold: int,
    sahi_model_input_size: int | None,
) -> list[Detection]:
    ensure_model_float32(model)
    input_size = sahi_model_input_size if sahi_model_input_size is not None else model_input_size(model, fallback=slice_size)
    head_state = capture_detector_head_state(model)
    try:
        boxes, scores, classes = sahi_inference_batched(
            model,
            image_bgr,
            slice_wh=(slice_size, slice_size),
            overlap_wh=(overlap, overlap),
            batch_size=batch_size,
            model_input_size=input_size,
            conf=conf,
            iou_threshold=iou,
            large_conf=large_conf,
            size_threshold=size_threshold,
            half=False,
        )
    finally:
        restore_detector_head_state(head_state)
    return detections_from_arrays(
        boxes,
        scores,
        classes,
        getattr(model, "names", {}),
        image_bgr.shape[:2],
        max_det,
    )


def results_to_detections(result: Any) -> list[Detection]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    names = result.names
    detections = []
    for idx, (box, score, cls_id) in enumerate(
        zip(xyxy, confs, classes, strict=True), start=1
    ):
        detections.append(
            Detection(
                detection_id=idx,
                cls_id=int(cls_id),
                cls_name=str(names[int(cls_id)]),
                conf=float(score),
                box_xyxy=np.array(box, dtype=np.float32),
            )
        )
    return detections


def run_predictions_batch(
    model: YOLO,
    images_bgr: list[np.ndarray],
    imgsz: int,
    conf: float,
    iou: float,
    device: str | int | None,
    max_det: int,
    batch_size: int = DEFAULT_PREDICT_BATCH_SIZE,
) -> list[list[Detection]]:
    if not images_bgr:
        return []
    ensure_model_float32(model, device)
    head_state = capture_detector_head_state(model)
    try:
        results = model.predict(
            source=images_bgr,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            batch=max(1, batch_size),
            half=False,
            verbose=False,
        )
    finally:
        restore_detector_head_state(head_state)
    return [results_to_detections(result) for result in results]


def candidate_key(candidate: Candidate) -> str:
    return f"{candidate.branch}:{candidate.candidate_id}:{candidate.cls_id}"


def box_geometry(box_xyxy: np.ndarray, image_shape: tuple[int, int]) -> dict[str, Any]:
    height, width = image_shape
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    cx = x1 + box_w / 2.0
    cy = y1 + box_h / 2.0
    col = min(2, max(0, int(cx / max(width, 1) * 3)))
    row = min(2, max(0, int(cy / max(height, 1) * 3)))
    row_names = ["top", "middle", "bottom"]
    col_names = ["left", "center", "right"]
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "box_width": box_w,
        "box_height": box_h,
        "box_area": box_w * box_h,
        "aspect_ratio": box_w / box_h if box_h > 0 else None,
        "center_x": cx,
        "center_y": cy,
        "center_x_norm": cx / width if width > 0 else None,
        "center_y_norm": cy / height if height > 0 else None,
        "image_region": f"{row_names[row]}_{col_names[col]}",
    }


def heatmap_correlation(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    x = a.astype(np.float32).reshape(-1)
    y = b.astype(np.float32).reshape(-1)
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def mask_patch(image_bgr: np.ndarray, patch_xyxy: list[int] | np.ndarray | None, fill_value: int = 127) -> np.ndarray:
    out = image_bgr.copy()
    if patch_xyxy is None:
        return out
    h, w = out.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in clamp_box(np.array(patch_xyxy, dtype=np.float32), w, h)]
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = fill_value
    return out


def random_patch_like(
    image_shape: tuple[int, int],
    patch_xyxy: list[int] | np.ndarray | None,
    seed: int,
) -> list[int] | None:
    if patch_xyxy is None:
        return None
    height, width = image_shape
    x1, y1, x2, y2 = [int(round(v)) for v in patch_xyxy]
    patch_w = max(1, x2 - x1)
    patch_h = max(1, y2 - y1)
    rng = np.random.default_rng(seed)
    left = int(rng.integers(0, max(1, width - patch_w + 1)))
    top = int(rng.integers(0, max(1, height - patch_h + 1)))
    return [left, top, min(width, left + patch_w), min(height, top + patch_h)]


def candidate_grid_metadata(
    candidate_id: int,
    branch_raw: dict[str, Any] | None,
    strides: Any,
) -> tuple[int | None, int | None, int | None, int | None]:
    feats = branch_raw.get("feats") if isinstance(branch_raw, dict) else None
    if not feats:
        return None, None, None, None
    stride_values = []
    if strides is not None:
        if isinstance(strides, torch.Tensor):
            stride_values = [int(v) for v in strides.detach().cpu().flatten().tolist()]
        elif isinstance(strides, (list, tuple)):
            stride_values = [int(v) for v in strides]
    offset = 0
    for level, feat in enumerate(feats):
        height, width = feat.shape[-2:]
        level_count = int(height * width)
        if offset <= candidate_id < offset + level_count:
            local = candidate_id - offset
            grid_y = int(local // width)
            grid_x = int(local % width)
            stride = stride_values[level] if level < len(stride_values) else None
            return stride, level, grid_x, grid_y
        offset += level_count
    return None, None, None, None


def decoded_tensor_to_candidates(
    decoded: torch.Tensor,
    *,
    branch: str,
    names: dict[int, str] | list[str],
    image_shape: tuple[int, int],
    input_shape: tuple[int, int],
    conf: float,
    boxes_are_xyxy: bool,
    branch_raw: dict[str, Any] | None = None,
    strides: Any = None,
    max_candidates: int | None = None,
) -> list[Candidate]:
    if decoded.ndim != 3:
        return []
    pred = decoded[0]
    if pred.shape[0] < 5:
        return []

    boxes = pred[:4, :].transpose(0, 1)
    if not boxes_are_xyxy:
        boxes = ops.xywh2xyxy(boxes)
    boxes = ops.scale_boxes(input_shape, boxes.clone(), image_shape)
    scores = pred[4:, :].transpose(0, 1)
    if scores.numel() == 0:
        return []

    best_scores, best_classes = scores.max(dim=1)
    order = torch.arange(best_scores.shape[0], device=best_scores.device)
    if max_candidates is not None and max_candidates > 0 and best_scores.shape[0] > max_candidates:
        order = best_scores.topk(max_candidates).indices

    boxes_np = boxes.detach().cpu().numpy()
    scores_np = best_scores.detach().cpu().numpy()
    classes_np = best_classes.detach().cpu().numpy().astype(int)
    candidates: list[Candidate] = []
    for candidate_id in order.detach().cpu().numpy().astype(int).tolist():
        cls_id = int(classes_np[candidate_id])
        box_xyxy = clamp_box(np.array(boxes_np[candidate_id], dtype=np.float32), image_shape[1], image_shape[0])
        stride, feature_level, grid_x, grid_y = candidate_grid_metadata(candidate_id, branch_raw, strides)
        score = float(scores_np[candidate_id])
        candidates.append(
            Candidate(
                branch=branch,
                candidate_id=int(candidate_id),
                cls_id=cls_id,
                cls_name=class_name(names, cls_id),
                score=score,
                box_xyxy=box_xyxy,
                objectness=None,
                stride=stride,
                feature_level=feature_level,
                grid_x=grid_x,
                grid_y=grid_y,
                kept_after_threshold=score >= conf,
            )
        )
    return candidates


def detections_from_postprocess_tensor(
    postprocessed: torch.Tensor,
    *,
    names: dict[int, str] | list[str],
    image_shape: tuple[int, int],
    input_shape: tuple[int, int],
    conf: float,
    max_det: int,
) -> list[Detection]:
    if postprocessed.ndim == 3:
        postprocessed = postprocessed[0]
    if postprocessed.numel() == 0:
        return []
    rows = postprocessed.detach().clone()
    rows = rows[rows[:, 4] >= conf]
    if rows.numel() == 0:
        return []
    rows[:, :4] = ops.scale_boxes(input_shape, rows[:, :4], image_shape)
    rows = rows[:max_det].detach().cpu().numpy()
    detections: list[Detection] = []
    for idx, row in enumerate(rows, start=1):
        cls_id = int(row[5])
        detections.append(
            Detection(
                detection_id=idx,
                cls_id=cls_id,
                cls_name=class_name(names, cls_id),
                conf=float(row[4]),
                box_xyxy=clamp_box(np.array(row[:4], dtype=np.float32), image_shape[1], image_shape[0]),
            )
        )
    return detections


def match_detections_to_candidates(
    detections: list[Detection],
    candidates: list[Candidate],
    *,
    min_iou: float = 0.05,
    preferred_branch: str | None = "one2one",
) -> dict[int, tuple[Candidate, float]]:
    matches: dict[int, tuple[Candidate, float]] = {}
    used: set[str] = set()
    for det in detections:
        best_candidate = None
        best_iou = 0.0
        best_rank = -1.0
        for candidate in candidates:
            key = candidate_key(candidate)
            if key in used or candidate.cls_id != det.cls_id:
                continue
            iou = box_iou_xyxy(det.box_xyxy, candidate.box_xyxy)
            if iou < min_iou:
                continue
            branch_bonus = 0.25 if preferred_branch is not None and candidate.branch == preferred_branch else 0.0
            rank = iou * 2.0 + candidate.score + branch_bonus
            if rank > best_rank:
                best_candidate = candidate
                best_iou = iou
                best_rank = rank
        if best_candidate is not None:
            used.add(candidate_key(best_candidate))
            matches[det.detection_id] = (best_candidate, float(best_iou))
    return matches


def apply_candidate_postprocess_marks(
    candidates: list[Candidate],
    detections: list[Detection],
    matches: dict[int, tuple[Candidate, float]],
    iou_threshold: float,
) -> None:
    kept_by_key: dict[str, Candidate] = {}
    for det in detections:
        match = matches.get(det.detection_id)
        if match is None:
            continue
        candidate, iou = match
        candidate.kept_after_nms = True
        candidate.final_detection_id = det.detection_id
        candidate.matched_final_iou = float(iou)
        kept_by_key[candidate_key(candidate)] = candidate

    kept_candidates = list(kept_by_key.values())
    for candidate in candidates:
        if candidate.kept_after_nms or not candidate.kept_after_threshold:
            continue
        best_suppressor = None
        best_iou = 0.0
        for kept in kept_candidates:
            if kept.branch != candidate.branch or kept.cls_id != candidate.cls_id:
                continue
            iou = box_iou_xyxy(candidate.box_xyxy, kept.box_xyxy)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_suppressor = kept
        if best_suppressor is not None:
            candidate.suppressed_by_candidate_id = best_suppressor.candidate_id
            candidate.suppressed_by_iou = float(best_iou)


class YoloAnalysisAdapter:
    """Small white-box adapter for raw candidate and postprocess analysis."""

    def __init__(
        self,
        model: YOLO,
        *,
        imgsz: int,
        conf: float,
        iou: float,
        device: str | int | None,
        max_det: int,
        max_candidates: int | None = None,
    ):
        self.model = model
        self.raw_model = model.model
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.max_det = max_det
        self.max_candidates = max_candidates
        self.names = getattr(model, "names", {})
        self.head = self.raw_model.model[-1] if hasattr(self.raw_model, "model") else None
        self.nc = len(self.names) if self.names else int(getattr(self.head, "nc", 0) or 0)

    def preprocess(self, image_bgr: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        return build_letterboxed_tensor(self.model, image_bgr, self.imgsz, self.device)

    def infer_raw(self, tensor: torch.Tensor) -> Any:
        self.raw_model.eval()
        with torch.no_grad():
            return self.raw_model(tensor)

    def decode_raw(
        self,
        raw_outputs: Any,
        *,
        image_shape: tuple[int, int],
        input_shape: tuple[int, int],
    ) -> tuple[list[Candidate], str]:
        candidates: list[Candidate] = []
        if (
            isinstance(raw_outputs, (list, tuple))
            and len(raw_outputs) > 1
            and isinstance(raw_outputs[1], dict)
            and self.head is not None
            and hasattr(self.head, "_inference")
        ):
            raw_dict = raw_outputs[1]
            for branch_name in ("one2one", "one2many"):
                branch_raw = raw_dict.get(branch_name)
                if not isinstance(branch_raw, dict):
                    continue
                try:
                    decoded = self.head._inference(branch_raw)
                except Exception:
                    continue
                candidates.extend(
                    decoded_tensor_to_candidates(
                        decoded,
                        branch=branch_name,
                        names=self.names,
                        image_shape=image_shape,
                        input_shape=input_shape,
                        conf=self.conf,
                        boxes_are_xyxy=True,
                        branch_raw=branch_raw,
                        strides=getattr(self.head, "stride", None),
                        max_candidates=self.max_candidates,
                    )
                )
            return candidates, "end2end_raw_dict"

        pred = raw_outputs[0] if isinstance(raw_outputs, (list, tuple)) else raw_outputs
        if isinstance(pred, torch.Tensor) and pred.ndim == 3 and pred.shape[1] > 6:
            candidates.extend(
                decoded_tensor_to_candidates(
                    pred,
                    branch="raw",
                    names=self.names,
                    image_shape=image_shape,
                    input_shape=input_shape,
                    conf=self.conf,
                    boxes_are_xyxy=False,
                    branch_raw=None,
                    strides=getattr(self.head, "stride", None),
                    max_candidates=self.max_candidates,
                )
            )
            return candidates, "standard_bcn"

        if isinstance(pred, torch.Tensor) and pred.ndim == 3 and pred.shape[-1] == 6:
            rows = pred[0]
            scores = rows[:, 4]
            classes = rows[:, 5].long()
            boxes = ops.scale_boxes(input_shape, rows[:, :4].clone(), image_shape)
            order = torch.arange(rows.shape[0], device=rows.device)
            if self.max_candidates is not None and self.max_candidates > 0 and rows.shape[0] > self.max_candidates:
                order = scores.topk(self.max_candidates).indices
            boxes_np = boxes.detach().cpu().numpy()
            scores_np = scores.detach().cpu().numpy()
            classes_np = classes.detach().cpu().numpy().astype(int)
            for candidate_id in order.detach().cpu().numpy().astype(int).tolist():
                cls_id = int(classes_np[candidate_id])
                score = float(scores_np[candidate_id])
                candidates.append(
                    Candidate(
                        branch="final_like",
                        candidate_id=int(candidate_id),
                        cls_id=cls_id,
                        cls_name=class_name(self.names, cls_id),
                        score=score,
                        box_xyxy=clamp_box(
                            np.array(boxes_np[candidate_id], dtype=np.float32),
                            image_shape[1],
                            image_shape[0],
                        ),
                        kept_after_threshold=score >= self.conf,
                    )
                )
            return candidates, "final_bnc6"

        return candidates, "unsupported"

    def postprocess(
        self,
        raw_outputs: Any,
        *,
        image_shape: tuple[int, int],
        input_shape: tuple[int, int],
    ) -> list[Detection]:
        pred = raw_outputs[0] if isinstance(raw_outputs, (list, tuple)) else raw_outputs
        if not isinstance(pred, torch.Tensor):
            return []

        if pred.ndim == 3 and pred.shape[-1] == 6:
            return detections_from_postprocess_tensor(
                pred,
                names=self.names,
                image_shape=image_shape,
                input_shape=input_shape,
                conf=self.conf,
                max_det=self.max_det,
            )

        nms_rows = ops.non_max_suppression(
            pred,
            conf_thres=self.conf,
            iou_thres=self.iou,
            max_det=self.max_det,
            nc=self.nc,
            in_place=False,
            end2end=getattr(self.raw_model, "end2end", False),
        )
        if not nms_rows:
            return []
        return detections_from_postprocess_tensor(
            nms_rows[0],
            names=self.names,
            image_shape=image_shape,
            input_shape=input_shape,
            conf=self.conf,
            max_det=self.max_det,
        )

    def infer_full(self, image_bgr: np.ndarray) -> AdapterInference:
        tensor, input_shape = self.preprocess(image_bgr)
        raw_outputs = self.infer_raw(tensor)
        image_shape = image_bgr.shape[:2]
        candidates, raw_output_type = self.decode_raw(
            raw_outputs,
            image_shape=image_shape,
            input_shape=input_shape,
        )
        final_detections = self.postprocess(
            raw_outputs,
            image_shape=image_shape,
            input_shape=input_shape,
        )
        preferred_branch = "one2one" if any(c.branch == "one2one" for c in candidates) else None
        matches = match_detections_to_candidates(
            final_detections,
            candidates,
            min_iou=0.05,
            preferred_branch=preferred_branch,
        )
        apply_candidate_postprocess_marks(candidates, final_detections, matches, self.iou)
        return AdapterInference(
            candidates=candidates,
            final_detections=final_detections,
            final_candidate_matches={det_id: match[0] for det_id, match in matches.items()},
            raw_output_type=raw_output_type,
        )


def candidate_rows_for_image(
    *,
    image_key: str,
    image_name: str,
    candidates: list[Candidate],
    image_shape: tuple[int, int],
    source_matches: dict[int, tuple[Candidate, float]],
    source_gt_matches: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    source_by_candidate: dict[str, dict[str, Any]] = {}
    for detection_id, (candidate, iou) in source_matches.items():
        gt_record = source_gt_matches.get(detection_id, {})
        source_by_candidate[candidate_key(candidate)] = {
            "source_detection_id": detection_id,
            "source_match_iou": float(iou),
            "source_gt_status": gt_record.get("gt_status"),
            "source_fp_type": gt_record.get("fp_type"),
            "source_best_gt_iou": gt_record.get("best_gt_iou"),
        }

    rows = []
    for candidate in candidates:
        source_record = source_by_candidate.get(candidate_key(candidate), {})
        rows.append(
            {
                "image_key": image_key,
                "image_name": image_name,
                "branch": candidate.branch,
                "candidate_id": candidate.candidate_id,
                "class_id": candidate.cls_id,
                "class_name": candidate.cls_name,
                "class_score": candidate.score,
                "objectness": candidate.objectness,
                "final_score": candidate.score,
                "stride": candidate.stride,
                "feature_level": candidate.feature_level,
                "grid_x": candidate.grid_x,
                "grid_y": candidate.grid_y,
                "is_pre_nms": True,
                "kept_after_threshold": candidate.kept_after_threshold,
                "kept_after_nms": candidate.kept_after_nms,
                "final_detection_id": candidate.final_detection_id,
                "matched_final_iou": candidate.matched_final_iou,
                "suppressed_by_candidate_id": candidate.suppressed_by_candidate_id,
                "suppressed_by_iou": candidate.suppressed_by_iou,
                "source_detection_id": source_record.get("source_detection_id"),
                "source_match_iou": source_record.get("source_match_iou"),
                "source_gt_status": source_record.get("source_gt_status"),
                "source_fp_type": source_record.get("source_fp_type"),
                "source_best_gt_iou": source_record.get("source_best_gt_iou"),
                **box_geometry(candidate.box_xyxy, image_shape),
            }
        )
    return rows


def postprocess_rows_for_image(
    *,
    image_key: str,
    image_name: str,
    pipeline: str,
    detections: list[Detection],
    candidate_matches: dict[int, tuple[Candidate, float]] | dict[int, Candidate],
    gt_matches: dict[int, dict[str, Any]],
    image_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    rows = []
    for det in detections:
        match = candidate_matches.get(det.detection_id)
        candidate = None
        candidate_iou = None
        if isinstance(match, tuple):
            candidate, candidate_iou = match
        elif isinstance(match, Candidate):
            candidate = match
            candidate_iou = box_iou_xyxy(det.box_xyxy, candidate.box_xyxy)
        gt_record = gt_matches.get(det.detection_id, {})
        rows.append(
            {
                "image_key": image_key,
                "image_name": image_name,
                "pipeline": pipeline,
                "final_detection_id": det.detection_id,
                "class_id": det.cls_id,
                "class_name": det.cls_name,
                "score": det.conf,
                "matched_candidate_branch": candidate.branch if candidate is not None else None,
                "matched_candidate_id": candidate.candidate_id if candidate is not None else None,
                "matched_candidate_score": candidate.score if candidate is not None else None,
                "matched_candidate_iou": candidate_iou,
                "gt_status": gt_record.get("gt_status"),
                "fp_type": gt_record.get("fp_type"),
                "matched_gt_id": gt_record.get("matched_gt_id"),
                "matched_gt_iou": gt_record.get("matched_gt_iou"),
                "best_gt_id": gt_record.get("best_gt_id"),
                "best_gt_iou": gt_record.get("best_gt_iou"),
                **box_geometry(det.box_xyxy, image_shape),
            }
        )
    return rows


def select_detections_for_explanation(
    detections: list[Detection],
    gt_matches: dict[int, dict[str, Any]],
    max_detections: int,
    selection: str,
) -> list[Detection]:
    selection = selection.lower()
    if selection == "false_positive":
        ranked = sorted(
            detections,
            key=lambda det: (gt_matches.get(det.detection_id, {}).get("gt_status") != "fp", -det.conf),
        )
    elif selection == "mixed":
        ranked = sorted(
            detections,
            key=lambda det: (
                0 if gt_matches.get(det.detection_id, {}).get("gt_status") == "fp" else 1,
                -det.conf,
            ),
        )
    elif selection == "low_conf":
        ranked = sorted(detections, key=lambda det: det.conf)
    else:
        ranked = sorted(detections, key=lambda det: det.conf, reverse=True)
    return ranked[:max_detections]


def match_detection(
    detections: list[Detection],
    target_box: np.ndarray,
    target_cls: int,
    min_iou: float = 0.05,
) -> tuple[float, float, np.ndarray | None]:
    best_score = 0.0
    best_iou = 0.0
    best_box = None
    for det in detections:
        if det.cls_id != target_cls:
            continue
        iou = box_iou_xyxy(target_box, det.box_xyxy)
        if iou < min_iou:
            continue
        rank = det.conf + iou
        current = best_score + best_iou
        if best_box is None or rank > current:
            best_score = det.conf
            best_iou = iou
            best_box = det.box_xyxy.copy()
    return best_score, best_iou, best_box


def mask_box(image_bgr: np.ndarray, box_xyxy: np.ndarray, fill_value: int, invert: bool = False) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in clamp_box(box_xyxy, w, h)]
    out = image_bgr.copy()
    if invert:
        out[:] = fill_value
        out[y1:y2, x1:x2] = image_bgr[y1:y2, x1:x2]
        return out
    out[y1:y2, x1:x2] = fill_value
    return out


def crop_box(image_bgr: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in clamp_box(box_xyxy, w, h)]
    return image_bgr[y1:y2, x1:x2].copy()


def raw_anchor_match_score(
    pred: torch.Tensor,
    target_box_orig: np.ndarray,
    target_cls: int,
    orig_shape: tuple[int, int],
    input_shape: tuple[int, int],
    min_anchor_iou: float = 0.05,
) -> GradcamTarget | None:
    boxes_xywh = pred[0, :4, :].transpose(0, 1)
    class_scores = pred[0, 4 + target_cls, :]
    boxes_xyxy = ops.xywh2xyxy(boxes_xywh)
    boxes_xyxy = ops.scale_boxes(input_shape, boxes_xyxy, orig_shape)
    boxes_xyxy_np = boxes_xyxy.detach().cpu().numpy()
    class_scores_np = class_scores.detach().cpu().numpy()

    best_idx = None
    best_score = -1.0
    best_iou = 0.0
    for idx, (box, cls_score) in enumerate(zip(boxes_xyxy_np, class_scores_np, strict=False)):
        iou = box_iou_xyxy(target_box_orig, box)
        if iou < min_anchor_iou:
            continue
        score = float(cls_score)
        if iou > best_iou or (math.isclose(iou, best_iou) and score > best_score):
            best_idx = idx
            best_iou = iou
            best_score = score
    if best_idx is None:
        return None
    return GradcamTarget(
        anchor_idx=int(best_idx),
        target_score=class_scores[best_idx],
        anchor_iou=float(best_iou),
        branch="raw",
    )


def gradcam_target_score(
    raw_model: torch.nn.Module,
    outputs: Any,
    target_box_orig: np.ndarray,
    target_cls: int,
    orig_shape: tuple[int, int],
    input_shape: tuple[int, int],
    min_anchor_iou: float = 0.05,
) -> GradcamTarget | None:
    if isinstance(outputs, (list, tuple)) and len(outputs) > 1 and isinstance(outputs[1], dict):
        raw_dict = outputs[1]
        head = raw_model.model[-1] if hasattr(raw_model, "model") else None
        if head is not None and hasattr(head, "_inference"):
            fallback: GradcamTarget | None = None
            best_match: GradcamTarget | None = None
            best_score_value = -1.0
            for branch_name in ("one2one", "one2many"):
                branch_raw = raw_dict.get(branch_name)
                if not isinstance(branch_raw, dict):
                    continue
                if "feats" not in branch_raw:
                    continue
                decoded = head._inference(branch_raw)
                boxes = decoded[0, :4, :].transpose(0, 1)
                boxes_scaled = ops.scale_boxes(input_shape, boxes.clone(), orig_shape)
                class_scores = decoded[0, 4 + target_cls, :]
                boxes_np = boxes_scaled.detach().cpu().numpy()
                scores_np = class_scores.detach().cpu().numpy()
                best_idx = None
                best_score = -1.0
                best_iou = 0.0
                for idx, (box, score) in enumerate(zip(boxes_np, scores_np, strict=False)):
                    iou = box_iou_xyxy(target_box_orig, box)
                    if iou < min_anchor_iou:
                        continue
                    score_value = float(score)
                    if iou > best_iou or (math.isclose(iou, best_iou) and score_value > best_score):
                        best_idx = idx
                        best_iou = iou
                        best_score = score_value
                if best_idx is not None:
                    target_score = class_scores[best_idx]
                    stride, feature_level, grid_x, grid_y = candidate_grid_metadata(
                        int(best_idx),
                        branch_raw,
                        getattr(head, "stride", None),
                    )
                    match = GradcamTarget(
                        anchor_idx=int(best_idx),
                        target_score=target_score,
                        anchor_iou=float(best_iou),
                        branch=branch_name,
                        feature_level=feature_level,
                        stride=stride,
                        grid_x=grid_x,
                        grid_y=grid_y,
                    )
                    if target_score.requires_grad:
                        if best_match is None or best_iou > best_match.anchor_iou or (
                            math.isclose(best_iou, best_match.anchor_iou) and best_score > best_score_value
                        ):
                            best_match = match
                            best_score_value = best_score
                    if fallback is None:
                        fallback = match
            if best_match is not None:
                return best_match
            if fallback is not None:
                return fallback

    pred = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    if isinstance(pred, torch.Tensor) and pred.ndim == 3 and pred.shape[1] > 6:
        return raw_anchor_match_score(
            pred=pred,
            target_box_orig=target_box_orig,
            target_cls=target_cls,
            orig_shape=orig_shape,
            input_shape=input_shape,
            min_anchor_iou=min_anchor_iou,
        )
    return None


def resolve_model_layer_index(layer_index: int, num_layers: int) -> int:
    return layer_index if layer_index >= 0 else num_layers + layer_index


def gradcam_feature_layer_indices(raw_model: torch.nn.Module, fallback_layer_index: int) -> tuple[list[int], list[int], int]:
    num_layers = len(raw_model.model) if hasattr(raw_model, "model") else 0
    head = raw_model.model[-1] if num_layers else None
    feature_layer_indices: list[int] = []
    head_inputs = getattr(head, "f", None)
    if isinstance(head_inputs, int):
        head_inputs = [head_inputs]
    if isinstance(head_inputs, (list, tuple)):
        for index in head_inputs:
            resolved = resolve_model_layer_index(int(index), num_layers)
            if 0 <= resolved < num_layers and resolved not in feature_layer_indices:
                feature_layer_indices.append(resolved)

    fallback = resolve_model_layer_index(fallback_layer_index, num_layers)
    hook_layer_indices = feature_layer_indices.copy()
    if 0 <= fallback < num_layers and fallback not in hook_layer_indices:
        hook_layer_indices.append(fallback)
    return feature_layer_indices, hook_layer_indices, fallback


def compute_gradcam(
    model: YOLO,
    image_bgr: np.ndarray,
    target_box: np.ndarray,
    target_cls: int,
    imgsz: int,
    device: str | int | None,
    layer_index: int = -2,
    min_anchor_iou: float = 0.05,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    raw_model = model.model
    raw_model.eval()
    feature_layer_indices, target_layer_indices, fallback_layer_index = gradcam_feature_layer_indices(raw_model, layer_index)
    activations: dict[int, torch.Tensor] = {}
    original_grad_state = torch.is_grad_enabled()

    def make_forward_hook(layer_idx: int):
        def forward_hook(_module, _inputs, output):
            tensor = output
            while isinstance(tensor, (list, tuple)):
                tensor = tensor[-1]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("Grad-CAM target layer did not produce a tensor output.")
            activations[layer_idx] = tensor
            if tensor.requires_grad:
                tensor.retain_grad()

        return forward_hook

    hooks = [raw_model.model[index].register_forward_hook(make_forward_hook(index)) for index in target_layer_indices]
    try:
        with torch.enable_grad():
            raw_model.zero_grad(set_to_none=True)
            tensor, input_shape = build_letterboxed_tensor(model, image_bgr, imgsz, device)
            tensor.requires_grad_(True)
            reset_detector_inference_cache(model)
            outputs = raw_model(tensor)
            target = gradcam_target_score(
                raw_model=raw_model,
                outputs=outputs,
                target_box_orig=target_box,
                target_cls=target_cls,
                orig_shape=image_bgr.shape[:2],
                input_shape=input_shape,
                min_anchor_iou=min_anchor_iou,
            )
            if target is None:
                return None, {
                    "gradcam_error": f"No raw anchor matched the selected detection at IoU >= {min_anchor_iou}.",
                    "gradcam_min_anchor_iou": float(min_anchor_iou),
                }
            if not target.target_score.requires_grad:
                return None, {"gradcam_error": "Selected detection score did not require gradients."}

            selected_layer_index = (
                feature_layer_indices[target.feature_level]
                if target.feature_level is not None and 0 <= target.feature_level < len(feature_layer_indices)
                else fallback_layer_index
            )
            act = activations.get(selected_layer_index)
            if act is None or not act.requires_grad:
                return None, {"gradcam_error": "Target layer activations did not require gradients."}

            target.target_score.backward(retain_graph=False)
            grads = act.grad
            if grads is None:
                return None, {"gradcam_error": "No gradients captured for the target layer."}
            act = act[0]
            grads = grads[0]
            weights = grads.mean(dim=(1, 2))
            cam = torch.relu((weights[:, None, None] * act).sum(dim=0))
            cam = cam.detach().cpu().numpy()
            raw_cam_max = float(np.nanmax(cam)) if cam.size else 0.0
            cam, content_region = resize_letterboxed_heatmap_to_original(
                cam,
                input_shape=input_shape,
                original_shape=image_bgr.shape[:2],
            )
            output_cam_max = float(np.nanmax(cam)) if cam.size else 0.0
            return cam, {
                "gradcam_anchor_iou": float(target.anchor_iou),
                "gradcam_anchor_index": int(target.anchor_idx),
                "gradcam_branch": target.branch,
                "gradcam_layer_index": int(selected_layer_index),
                "gradcam_requested_layer_index": int(layer_index),
                "gradcam_feature_level": target.feature_level,
                "gradcam_stride": target.stride,
                "gradcam_grid_x": target.grid_x,
                "gradcam_grid_y": target.grid_y,
                "gradcam_min_anchor_iou": float(min_anchor_iou),
                "gradcam_letterbox_content_xyxy": [int(value) for value in content_region],
                "gradcam_raw_max": raw_cam_max,
                "gradcam_max": output_cam_max,
                "gradcam_has_signal": bool(raw_cam_max > 0),
            }
    finally:
        torch.set_grad_enabled(original_grad_state)
        for hook in hooks:
            hook.remove()


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if heatmap.size == 0 or not np.isfinite(heatmap).any() or float(np.nanmax(heatmap)) <= 0:
        return image_rgb.copy()
    heatmap_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return np.clip((1.0 - alpha) * image_rgb + alpha * colored, 0, 255).astype(np.uint8)


def patch_axis_positions(region_start: float, region_end: float, image_size: int, patch_size: int, stride: int) -> list[int]:
    max_start = max(0, image_size - patch_size)
    start = int(np.clip(region_start, 0, max_start))
    stop = int(np.clip(region_end - patch_size, 0, max_start))
    if stop < start:
        centered = int(round((region_start + region_end - patch_size) / 2))
        return [int(np.clip(centered, 0, max_start))]

    positions = list(range(start, max(start + 1, stop + 1), stride))
    if positions[-1] != stop:
        positions.append(stop)
    return positions


def compute_occlusion_sensitivity(
    model: YOLO,
    image_bgr: np.ndarray,
    target_box: np.ndarray,
    target_cls: int,
    original_score: float,
    imgsz: int,
    conf: float,
    iou: float,
    device: str | int | None,
    max_det: int,
    patch_size: int,
    stride: int,
    batch_size: int = DEFAULT_PREDICT_BATCH_SIZE,
    scope: str = "target",
    roi_margin: float = 0.5,
    use_sahi: bool = False,
    slice_size: int = 320,
    overlap: int = 64,
    sahi_batch_size: int = DEFAULT_SAHI_BATCH_SIZE,
    large_conf: float = 0.25,
    size_threshold: int = 16,
    sahi_model_input_size: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = image_bgr.shape[:2]
    patch_size = max(8, min(patch_size, min(height, width)))
    stride = max(4, stride)
    score_sum = np.zeros((height, width), dtype=np.float32)
    count_sum = np.zeros((height, width), dtype=np.float32)

    if scope == "target":
        roi = expand_box(target_box, width, height, roi_margin)
        region_x1, region_y1, region_x2, region_y2 = roi
    elif scope == "full":
        region_x1, region_y1, region_x2, region_y2 = 0.0, 0.0, float(width), float(height)
    else:
        raise ValueError(f"Unsupported occlusion scope: {scope}. Expected 'target' or 'full'.")

    y_positions = patch_axis_positions(region_y1, region_y2, height, patch_size, stride)
    x_positions = patch_axis_positions(region_x1, region_x2, width, patch_size, stride)

    positions = [(top, left) for top in y_positions for left in x_positions]
    predict_batch_size = max(1, batch_size)

    max_drop = 0.0
    max_drop_box = None
    for start in range(0, len(positions), predict_batch_size):
        batch_positions = positions[start : start + predict_batch_size]
        masked_images = []
        for top, left in batch_positions:
            masked = image_bgr.copy()
            masked[top : top + patch_size, left : left + patch_size] = 127
            masked_images.append(masked)

        if use_sahi:
            batch_detections = [
                run_predictions(
                    model,
                    masked,
                    imgsz,
                    conf,
                    iou,
                    device,
                    max_det,
                    use_sahi=True,
                    slice_size=slice_size,
                    overlap=overlap,
                    sahi_batch_size=sahi_batch_size,
                    large_conf=large_conf,
                    size_threshold=size_threshold,
                    sahi_model_input_size=sahi_model_input_size,
                )
                for masked in masked_images
            ]
        else:
            batch_detections = run_predictions_batch(
                model,
                masked_images,
                imgsz,
                conf,
                iou,
                device,
                max_det,
                batch_size=predict_batch_size,
            )
        for (top, left), detections in zip(batch_positions, batch_detections, strict=False):
            matched_score, _, _ = match_detection(detections, target_box, target_cls)
            drop = max(0.0, original_score - matched_score)
            score_sum[top : top + patch_size, left : left + patch_size] += drop
            count_sum[top : top + patch_size, left : left + patch_size] += 1
            if drop >= max_drop:
                max_drop = drop
                max_drop_box = [left, top, left + patch_size, top + patch_size]

    heatmap = score_sum / np.maximum(count_sum, 1e-6)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap, {
        "occlusion_patch_size": int(patch_size),
        "occlusion_stride": int(stride),
        "occlusion_scope": scope,
        "occlusion_roi_margin": float(roi_margin),
        "occlusion_roi_xyxy": [float(region_x1), float(region_y1), float(region_x2), float(region_y2)],
        "occlusion_num_patches": int(len(positions)),
        "occlusion_batch_size": int(predict_batch_size),
        "occlusion_sahi": bool(use_sahi),
        "occlusion_max_drop": float(max_drop),
        "occlusion_mean_drop": float(score_sum.sum() / np.maximum(count_sum.sum(), 1.0)),
        "occlusion_peak_patch_xyxy": max_drop_box,
    }


def build_counterfactual_variants(
    image_bgr: np.ndarray,
    target_box: np.ndarray,
    expanded_box: np.ndarray,
) -> list[CounterfactualVariant]:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in clamp_box(target_box, w, h)]
    mean_color = [int(v) for v in image_bgr.reshape(-1, 3).mean(axis=0)]

    crop_bgr = crop_box(image_bgr, expanded_box)
    crop_target_box = np.array(
        [
            target_box[0] - expanded_box[0],
            target_box[1] - expanded_box[1],
            target_box[2] - expanded_box[0],
            target_box[3] - expanded_box[1],
        ],
        dtype=np.float32,
    )

    object_only_bgr = mask_box(image_bgr, target_box, fill_value=0, invert=True)
    patch_on_mean_bgr = np.full_like(image_bgr, mean_color, dtype=np.uint8)
    patch_on_mean_bgr[y1:y2, x1:x2] = image_bgr[y1:y2, x1:x2]
    object_masked_bgr = mask_box(image_bgr, target_box, fill_value=127, invert=False)

    target_blurred_bgr = image_bgr.copy()
    if y2 > y1 and x2 > x1:
        k = max(9, int(max(x2 - x1, y2 - y1) // 2) * 2 + 1)
        target_blurred_bgr[y1:y2, x1:x2] = cv2.GaussianBlur(image_bgr[y1:y2, x1:x2], (k, k), 0)

    background_blurred_bgr = cv2.GaussianBlur(image_bgr, (31, 31), 0)
    background_blurred_bgr[y1:y2, x1:x2] = image_bgr[y1:y2, x1:x2]

    brightness_up_bgr = cv2.convertScaleAbs(image_bgr, alpha=1.15, beta=20)
    brightness_down_bgr = cv2.convertScaleAbs(image_bgr, alpha=0.85, beta=-20)

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    shift_x = int(round(box_w * 0.25))
    shifted_box = expanded_box.copy()
    shifted_box[[0, 2]] = shifted_box[[0, 2]] + shift_x
    shifted_box = clamp_box(shifted_box, w, h)
    shifted_crop_bgr = crop_box(image_bgr, shifted_box)
    shifted_target_box = np.array(
        [
            target_box[0] - shifted_box[0],
            target_box[1] - shifted_box[1],
            target_box[2] - shifted_box[0],
            target_box[3] - shifted_box[1],
        ],
        dtype=np.float32,
    )

    resized_object_bgr = image_bgr.copy()
    if y2 > y1 and x2 > x1:
        resized_object_bgr[y1:y2, x1:x2] = mean_color
        scale = 0.7
        new_w = max(1, int(round(box_w * scale)))
        new_h = max(1, int(round(box_h * scale)))
        patch = cv2.resize(image_bgr[y1:y2, x1:x2], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        cx = int(round((x1 + x2) / 2))
        cy = int(round((y1 + y2) / 2))
        nx1 = int(np.clip(cx - new_w // 2, 0, w - new_w))
        ny1 = int(np.clip(cy - new_h // 2, 0, h - new_h))
        resized_object_bgr[ny1 : ny1 + new_h, nx1 : nx1 + new_w] = patch
        resized_target_box = np.array([nx1, ny1, nx1 + new_w, ny1 + new_h], dtype=np.float32)
    else:
        resized_target_box = target_box.copy()

    return [
        CounterfactualVariant("original", image_bgr.copy(), target_box.copy(), "counterfactual_original.png"),
        CounterfactualVariant("crop", crop_bgr, crop_target_box, "counterfactual_crop.png"),
        CounterfactualVariant("object_only", object_only_bgr, target_box.copy(), "counterfactual_object_only.png"),
        CounterfactualVariant("patch_on_mean", patch_on_mean_bgr, target_box.copy(), "counterfactual_patch_on_mean.png"),
        CounterfactualVariant("object_masked", object_masked_bgr, target_box.copy(), "counterfactual_object_masked.png"),
        CounterfactualVariant("target_blurred", target_blurred_bgr, target_box.copy(), "counterfactual_target_blurred.png"),
        CounterfactualVariant(
            "background_blurred",
            background_blurred_bgr,
            target_box.copy(),
            "counterfactual_background_blurred.png",
        ),
        CounterfactualVariant("brightness_up", brightness_up_bgr, target_box.copy(), "counterfactual_brightness_up.png"),
        CounterfactualVariant(
            "brightness_down",
            brightness_down_bgr,
            target_box.copy(),
            "counterfactual_brightness_down.png",
        ),
        CounterfactualVariant("shifted_crop", shifted_crop_bgr, shifted_target_box, "counterfactual_shifted_crop.png"),
        CounterfactualVariant(
            "object_resized_70",
            resized_object_bgr,
            resized_target_box,
            "counterfactual_object_resized_70.png",
        ),
    ]


def run_counterfactual_variants(
    model: YOLO,
    variants: list[CounterfactualVariant],
    target_cls: int,
    imgsz: int,
    conf: float,
    iou: float,
    device: str | int | None,
    max_det: int,
    predict_batch_size: int,
    *,
    use_sahi: bool,
    slice_size: int,
    overlap: int,
    sahi_batch_size: int,
    large_conf: float,
    size_threshold: int,
    sahi_model_input_size: int | None,
) -> list[tuple[CounterfactualVariant, float, float, np.ndarray | None]]:
    if use_sahi:
        detections_by_variant = [
            run_predictions(
                model,
                variant.image_bgr,
                imgsz,
                conf,
                iou,
                device,
                max_det,
                use_sahi=True,
                slice_size=slice_size,
                overlap=overlap,
                sahi_batch_size=sahi_batch_size,
                large_conf=large_conf,
                size_threshold=size_threshold,
                sahi_model_input_size=sahi_model_input_size,
            )
            for variant in variants
        ]
    else:
        detections_by_variant = run_predictions_batch(
            model,
            [variant.image_bgr for variant in variants],
            imgsz,
            conf,
            iou,
            device,
            max_det,
            batch_size=max(1, min(predict_batch_size, len(variants))),
        )

    rows = []
    for variant, detections in zip(variants, detections_by_variant, strict=False):
        score, matched_iou, matched_box = match_detection(
            detections,
            variant.target_box,
            target_cls,
            min_iou=0.05,
        )
        rows.append((variant, score, matched_iou, matched_box))
    return rows


def compute_drise(
    model: YOLO,
    image_bgr: np.ndarray,
    target_box: np.ndarray,
    target_cls: int,
    imgsz: int,
    conf: float,
    iou: float,
    device: str | int | None,
    max_det: int,
    *,
    prediction_conf: float,
    num_masks: int,
    min_masks: int,
    target_positive_masks: int,
    stop_if_no_signal_after: int,
    grid_size: int,
    mask_prob: float,
    batch_size: int,
    seed: int,
    use_sahi: bool,
    slice_size: int,
    overlap: int,
    sahi_batch_size: int,
    large_conf: float,
    size_threshold: int,
    sahi_model_input_size: int | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if num_masks <= 0:
        return None, {"drise_num_masks": 0}

    requested_masks = int(num_masks)
    min_masks = int(np.clip(min_masks, 0, requested_masks))
    target_positive_masks = max(0, int(target_positive_masks))
    stop_if_no_signal_after = int(np.clip(stop_if_no_signal_after, 0, requested_masks))
    prediction_conf = float(np.clip(prediction_conf, 0.0, 1.0))
    prediction_large_conf = min(float(large_conf), prediction_conf)
    height, width = image_bgr.shape[:2]
    rng = np.random.default_rng(seed)
    heatmap_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = 0.0
    best_weight = 0.0
    best_mask_index = None
    positive_masks = 0
    processed = 0
    early_stop_reason = None
    batch_size = max(1, batch_size)

    for start in range(0, requested_masks, batch_size):
        count = min(batch_size, requested_masks - start)
        masks = []
        masked_images = []
        for _ in range(count):
            low_res = (rng.random((grid_size, grid_size)) < mask_prob).astype(np.float32)
            mask = cv2.resize(low_res, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = np.clip(mask, 0.0, 1.0)
            masked = (image_bgr.astype(np.float32) * mask[..., None] + 127.0 * (1.0 - mask[..., None])).astype(
                np.uint8
            )
            masks.append(mask)
            masked_images.append(masked)

        if use_sahi:
            detections_batch = [
                run_predictions(
                    model,
                    masked,
                    imgsz,
                    prediction_conf,
                    iou,
                    device,
                    max_det,
                    use_sahi=True,
                    slice_size=slice_size,
                    overlap=overlap,
                    sahi_batch_size=sahi_batch_size,
                    large_conf=prediction_large_conf,
                    size_threshold=size_threshold,
                    sahi_model_input_size=sahi_model_input_size,
                )
                for masked in masked_images
            ]
        else:
            detections_batch = run_predictions_batch(
                model,
                masked_images,
                imgsz,
                prediction_conf,
                iou,
                device,
                max_det,
                batch_size=count,
            )

        for offset, (mask, detections) in enumerate(zip(masks, detections_batch, strict=False)):
            score, matched_iou, _ = match_detection(detections, target_box, target_cls, min_iou=0.05)
            weight = float(score * matched_iou)
            heatmap_sum += mask * weight
            weight_sum += weight
            if weight > 0:
                positive_masks += 1
            if weight > best_weight:
                best_weight = weight
                best_mask_index = start + offset
            processed += 1

        if processed >= min_masks:
            if target_positive_masks > 0 and positive_masks >= target_positive_masks:
                early_stop_reason = "target_positive_masks"
                break
            if stop_if_no_signal_after > 0 and processed >= stop_if_no_signal_after and positive_masks == 0:
                early_stop_reason = "no_signal_after_threshold"
                break

    if weight_sum > 0:
        heatmap = heatmap_sum / weight_sum
    else:
        heatmap = heatmap_sum
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap, {
        "drise_requested_masks": int(requested_masks),
        "drise_num_masks": int(processed),
        "drise_min_masks": int(min_masks),
        "drise_target_positive_masks": int(target_positive_masks),
        "drise_stop_if_no_signal_after": int(stop_if_no_signal_after),
        "drise_early_stop_reason": early_stop_reason,
        "drise_grid_size": int(grid_size),
        "drise_mask_probability": float(mask_prob),
        "drise_prediction_conf": float(prediction_conf),
        "drise_prediction_large_conf": float(prediction_large_conf),
        "drise_weight_sum": float(weight_sum),
        "drise_best_weight": float(best_weight),
        "drise_best_mask_index": best_mask_index,
        "drise_positive_masks": int(positive_masks),
        "drise_has_signal": bool(weight_sum > 0),
        "drise_sahi": bool(use_sahi),
    }


def build_robustness_variants(image_bgr: np.ndarray, target_box: np.ndarray) -> list[RobustnessVariant]:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in clamp_box(target_box, w, h)]

    blur = cv2.GaussianBlur(image_bgr, (9, 9), 0)

    rng = np.random.default_rng(123)
    noise = np.clip(image_bgr.astype(np.float32) + rng.normal(0, 12, image_bgr.shape), 0, 255).astype(np.uint8)

    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 35])
    compression = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else image_bgr.copy()

    brightness_contrast = cv2.convertScaleAbs(image_bgr, alpha=1.25, beta=18)

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    shift_x = int(round(box_w * 0.20))
    shift_y = int(round(box_h * 0.10))
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    crop_shift = cv2.warpAffine(image_bgr, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    shifted_target = clamp_box(
        np.array([target_box[0] + shift_x, target_box[1] + shift_y, target_box[2] + shift_x, target_box[3] + shift_y]),
        w,
        h,
    )

    small = cv2.resize(image_bgr, (max(1, int(w * 0.75)), max(1, int(h * 0.75))), interpolation=cv2.INTER_AREA)
    scale_change = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    kernel_size = 13
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    motion_blur = cv2.filter2D(image_bgr, -1, kernel)

    partial_occlusion = image_bgr.copy()
    occ_w = max(1, int(box_w * 0.45))
    occ_h = max(1, int(box_h * 0.45))
    occ_x1 = int(np.clip((x1 + x2) / 2 - occ_w / 2, 0, w - occ_w))
    occ_y1 = int(np.clip((y1 + y2) / 2 - occ_h / 2, 0, h - occ_h))
    partial_occlusion[occ_y1 : occ_y1 + occ_h, occ_x1 : occ_x1 + occ_w] = 127

    return [
        RobustnessVariant("blur", blur, target_box.copy()),
        RobustnessVariant("noise", noise, target_box.copy()),
        RobustnessVariant("compression", compression, target_box.copy()),
        RobustnessVariant("brightness_contrast", brightness_contrast, target_box.copy()),
        RobustnessVariant("crop_shift", crop_shift, shifted_target.astype(np.float32)),
        RobustnessVariant("scale_change", scale_change, target_box.copy()),
        RobustnessVariant("motion_blur", motion_blur, target_box.copy()),
        RobustnessVariant("partial_occlusion", partial_occlusion, target_box.copy()),
    ]


def run_robustness_variants(
    model: YOLO,
    variants: list[RobustnessVariant],
    target_cls: int,
    imgsz: int,
    conf: float,
    iou: float,
    device: str | int | None,
    max_det: int,
    predict_batch_size: int,
) -> list[tuple[RobustnessVariant, float, float, np.ndarray | None]]:
    detections_by_variant = run_predictions_batch(
        model,
        [variant.image_bgr for variant in variants],
        imgsz,
        conf,
        iou,
        device,
        max_det,
        batch_size=max(1, min(predict_batch_size, len(variants))),
    )
    rows = []
    for variant, detections in zip(variants, detections_by_variant, strict=False):
        score, matched_iou, matched_box = match_detection(
            detections,
            variant.target_box,
            target_cls,
            min_iou=0.05,
        )
        rows.append((variant, score, matched_iou, matched_box))
    return rows


def run_deletion_sanity_check(
    model: YOLO,
    image_bgr: np.ndarray,
    target_box: np.ndarray,
    target_cls: int,
    original_score: float,
    important_patch: list[int] | None,
    imgsz: int,
    conf: float,
    iou: float,
    device: str | int | None,
    max_det: int,
    seed: int,
) -> dict[str, Any]:
    random_patch = random_patch_like(image_bgr.shape[:2], important_patch, seed)
    important_image = mask_patch(image_bgr, important_patch)
    random_image = mask_patch(image_bgr, random_patch)
    important_dets, random_dets = run_predictions_batch(
        model,
        [important_image, random_image],
        imgsz,
        conf,
        iou,
        device,
        max_det,
        batch_size=2,
    )
    important_score, important_iou, _ = match_detection(important_dets, target_box, target_cls, min_iou=0.05)
    random_score, random_iou, _ = match_detection(random_dets, target_box, target_cls, min_iou=0.05)
    important_drop = max(0.0, original_score - important_score)
    random_drop = max(0.0, original_score - random_score)
    return {
        "deletion_important_patch_xyxy": important_patch,
        "deletion_random_patch_xyxy": random_patch,
        "deletion_important_score": float(important_score),
        "deletion_random_score": float(random_score),
        "deletion_important_iou": float(important_iou),
        "deletion_random_iou": float(random_iou),
        "deletion_important_drop": float(important_drop),
        "deletion_random_drop": float(random_drop),
        "deletion_score_margin": float(important_drop - random_drop),
        "deletion_check_pass": bool(important_drop > random_drop),
    }


def compute_randomized_head_gradcam(
    model: YOLO,
    image_bgr: np.ndarray,
    target_box: np.ndarray,
    target_cls: int,
    imgsz: int,
    device: str | int | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    raw_model = model.model
    if not hasattr(raw_model, "model") or not raw_model.model:
        return None, {"randomized_weights_check": "unsupported_model"}
    head = raw_model.model[-1]
    state = {key: value.detach().clone() for key, value in head.state_dict().items()}
    try:
        for param in head.parameters():
            if param.ndim >= 2:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            else:
                torch.nn.init.zeros_(param)
        heatmap, meta = compute_gradcam(
            model=model,
            image_bgr=image_bgr,
            target_box=target_box,
            target_cls=target_cls,
            imgsz=imgsz,
            device=device,
        )
        meta["randomized_weights_check"] = "run"
        return heatmap, meta
    finally:
        head.load_state_dict(state)


def diagnose_detection(record: dict[str, Any]) -> tuple[str, str]:
    if record.get("gt_status") == "fp":
        if record.get("original_conf", 0.0) < 0.15:
            return (
                "Low-confidence false positive.",
                "Try confidence calibration, class-specific thresholds, or hard-negative mining around this score range.",
            )
        if record.get("context_drop", 0.0) > record.get("mask_drop", 0.0) + 0.1:
            return (
                "False positive appears context-heavy.",
                "Add hard negatives with similar background and use background-randomizing augmentation.",
            )
        return (
            "False positive remains after object/context tests.",
            "Inspect raw candidates and nearby labels; this may be missing annotation coverage or visual class confusion.",
        )
    if record.get("context_drop", 0.0) > record.get("mask_drop", 0.0) + 0.1:
        return (
            "Detection relies more on context than object pixels.",
            "Increase object-centric examples, tighter crops, and augmentation that breaks background shortcuts.",
        )
    if record.get("mask_drop", 0.0) > record.get("context_drop", 0.0) + 0.1:
        return (
            "Detection appears object-driven.",
            "Keep this as a positive reference pattern when comparing false positives.",
        )
    return (
        "Detection uses object pixels and context at similar strength.",
        "Compare D-RISE/occlusion peaks with the box edges and nearby background before changing data.",
    )


def write_detection_markdown_report(
    path: Path,
    record: dict[str, Any],
    counterfactual_records: list[dict[str, Any]],
) -> None:
    diagnosis, recommendation = diagnose_detection(record)
    lines = [
        f"# Detection {record['detection_id']} - {record['class_name']}",
        "",
        "## Target",
        "",
        f"- image: `{record['image_name']}`",
        f"- confidence: `{record['original_conf']:.4f}`",
        f"- box: `[{record['x1']:.1f}, {record['y1']:.1f}, {record['x2']:.1f}, {record['y2']:.1f}]`",
        f"- GT status: `{record.get('gt_status')}`",
        f"- FP type: `{record.get('fp_type')}`",
        f"- matched candidate: `{record.get('matched_candidate_branch')}:{record.get('matched_candidate_id')}`",
        "",
        "## Evidence",
        "",
        f"- crop score: `{record['crop_score_same_class']:.4f}`",
        f"- object-only score: `{record['object_only_score']:.4f}`",
        f"- object-masked score: `{record['object_masked_score']:.4f}`",
        f"- mask drop: `{record['mask_drop']:.4f}`",
        f"- context drop: `{record['context_drop']:.4f}`",
        f"- occlusion max drop: `{record.get('occlusion_max_drop')}`",
        f"- D-RISE masks: `{record.get('drise_num_masks')}`",
        f"- Grad-CAM anchor: `{record.get('gradcam_branch')}:{record.get('gradcam_anchor_index')}`",
        "",
        "## Counterfactuals",
        "",
        "| variant | score | matched IoU | drop |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in counterfactual_records:
        lines.append(
            f"| {row['variant']} | {row['score']:.4f} | {row['matched_iou']:.4f} | {row['score_drop']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            diagnosis,
            "",
            "## Recommended Fix",
            "",
            recommendation,
            "",
        ]
    )
    path.write_text("\n".join(lines))


def explain_image(
    model: YOLO,
    image_path: Path,
    run_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    device: str | int | None,
    max_det: int,
    max_detections: int,
    crop_margin: float,
    occlusion_patch: int,
    occlusion_stride: int,
    labels_dir: Path | None = None,
    gt_iou_threshold: float = 0.5,
    predict_batch_size: int = DEFAULT_PREDICT_BATCH_SIZE,
    occlusion_scope: str = "target",
    occlusion_roi_margin: float = 0.5,
    use_sahi: bool = True,
    counterfactual_sahi: bool = False,
    slice_size: int = 320,
    overlap: int = 64,
    sahi_batch_size: int = DEFAULT_SAHI_BATCH_SIZE,
    large_conf: float = 0.25,
    size_threshold: int = 16,
    selection: str = "top_conf",
    max_candidates: int | None = None,
    drise_masks: int = 0,
    drise_min_masks: int = 64,
    drise_target_positive_masks: int = 24,
    drise_stop_if_no_signal_after: int = 64,
    drise_grid_size: int = 12,
    drise_mask_prob: float = 0.5,
    drise_conf: float | None = None,
    drise_seed: int = 0,
    robustness: bool = True,
    sanity_checks: bool = True,
    randomized_weights_sanity: bool = False,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    image_bgr = load_image(image_path)
    image_rgb = bgr_to_rgb(image_bgr)
    image_key = image_path.stem
    image_dir = run_dir / "assets" / image_key
    image_dir.mkdir(parents=True, exist_ok=True)
    predict_batch_size = max(1, predict_batch_size)
    drise_prediction_conf = min(conf, 0.01) if drise_conf is None else float(drise_conf)
    occlusion_scope = occlusion_scope.lower()
    sahi_model_input_size = model_input_size(model, fallback=imgsz)
    adapter = YoloAnalysisAdapter(
        model,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        max_det=max_det,
        max_candidates=max_candidates,
    )
    adapter_result = adapter.infer_full(image_bgr)

    names = getattr(model, "names", {})
    ground_truths, label_path = load_ground_truths(image_path, image_bgr.shape[:2], names, labels_dir)
    gt_available = label_path is not None

    if use_sahi:
        all_detections = run_predictions(
            model,
            image_bgr,
            imgsz,
            conf,
            iou,
            device,
            max_det,
            use_sahi=True,
            slice_size=slice_size,
            overlap=overlap,
            sahi_batch_size=sahi_batch_size,
            large_conf=large_conf,
            size_threshold=size_threshold,
            sahi_model_input_size=sahi_model_input_size,
        )
    else:
        all_detections = adapter_result.final_detections
    preferred_branch = "one2one" if any(c.branch == "one2one" for c in adapter_result.candidates) else None
    source_candidate_matches = match_detections_to_candidates(
        all_detections,
        adapter_result.candidates,
        min_iou=0.05,
        preferred_branch=preferred_branch,
    )

    gt_matches: dict[int, dict[str, Any]] = {}
    adapter_gt_matches: dict[int, dict[str, Any]] = {}
    false_negatives: list[GroundTruth] = []
    gt_summary: dict[str, Any] = {
        "num_ground_truth": None,
        "num_tp": None,
        "num_fp": None,
        "num_fn": None,
        "precision": None,
        "recall": None,
        "f1": None,
    }
    ground_truth_asset = None
    pred_vs_gt_asset = None
    false_negatives_file = None
    if gt_available:
        gt_matches, false_negatives, gt_summary = match_detections_to_ground_truth(
            all_detections, ground_truths, gt_iou_threshold
        )
        adapter_gt_matches, _, _ = match_detections_to_ground_truth(
            adapter_result.final_detections, ground_truths, gt_iou_threshold
        )
        false_negative_ids = {gt.gt_id for gt in false_negatives}
        save_rgb(image_dir / "ground_truth.png", render_ground_truth(image_rgb, ground_truths))
        save_rgb(
            image_dir / "pred_vs_gt.png",
            render_predictions_vs_ground_truth(image_rgb, all_detections, ground_truths, gt_matches, false_negative_ids),
        )
        false_negative_rows = [
            {
                "gt_id": gt.gt_id,
                "class_id": gt.cls_id,
                "class_name": gt.cls_name,
                "x1": float(gt.box_xyxy[0]),
                "y1": float(gt.box_xyxy[1]),
                "x2": float(gt.box_xyxy[2]),
                "y2": float(gt.box_xyxy[3]),
            }
            for gt in false_negatives
        ]
        (image_dir / "false_negatives.json").write_text(json.dumps(false_negative_rows, indent=2))
        ground_truth_asset = str(Path("assets") / image_key / "ground_truth.png")
        pred_vs_gt_asset = str(Path("assets") / image_key / "pred_vs_gt.png")
        false_negatives_file = str(Path("assets") / image_key / "false_negatives.json")

    detections = select_detections_for_explanation(all_detections, gt_matches, max_detections, selection)
    candidate_rows = candidate_rows_for_image(
        image_key=image_key,
        image_name=image_path.name,
        candidates=adapter_result.candidates,
        image_shape=image_bgr.shape[:2],
        source_matches=source_candidate_matches,
        source_gt_matches=gt_matches,
    )
    postprocess_rows = postprocess_rows_for_image(
        image_key=image_key,
        image_name=image_path.name,
        pipeline="full_image_adapter",
        detections=adapter_result.final_detections,
        candidate_matches=adapter_result.final_candidate_matches,
        gt_matches=adapter_gt_matches,
        image_shape=image_bgr.shape[:2],
    )
    postprocess_rows.extend(
        postprocess_rows_for_image(
            image_key=image_key,
            image_name=image_path.name,
            pipeline="source_sahi" if use_sahi else "source_full_image",
            detections=all_detections,
            candidate_matches=source_candidate_matches,
            gt_matches=gt_matches,
            image_shape=image_bgr.shape[:2],
        )
    )

    save_rgb(image_dir / "original.png", image_rgb)
    save_rgb(image_dir / "annotated.png", render_all_detections(image_rgb, all_detections))

    image_summary = {
        "image_key": image_key,
        "image_name": image_path.name,
        "image_path": str(image_path.resolve()),
        "label_path": str(label_path.resolve()) if label_path is not None else None,
        "original_asset": str(Path("assets") / image_key / "original.png"),
        "annotated_asset": str(Path("assets") / image_key / "annotated.png"),
        "ground_truth_asset": ground_truth_asset,
        "pred_vs_gt_asset": pred_vs_gt_asset,
        "false_negatives_file": false_negatives_file,
        "num_detections": len(detections),
        "num_predictions": len(all_detections),
        "num_raw_candidates": len(adapter_result.candidates),
        "num_adapter_final_detections": len(adapter_result.final_detections),
        "image_width": int(image_bgr.shape[1]),
        "image_height": int(image_bgr.shape[0]),
        "raw_output_type": adapter_result.raw_output_type,
        "selection": selection,
        "sahi": use_sahi,
        "slice_size": slice_size if use_sahi else None,
        "overlap": overlap if use_sahi else None,
        "large_conf": large_conf if use_sahi else None,
        "size_threshold": size_threshold if use_sahi else None,
        **gt_summary,
    }

    detection_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    sanity_rows: list[dict[str, Any]] = []
    for det in detections:
        det_dir = image_dir / f"detection_{det.detection_id:03d}"
        det_dir.mkdir(parents=True, exist_ok=True)

        expanded_box = expand_box(det.box_xyxy, image_bgr.shape[1], image_bgr.shape[0], crop_margin)
        variants = build_counterfactual_variants(image_bgr, det.box_xyxy, expanded_box)
        variant_results = run_counterfactual_variants(
            model,
            variants,
            det.cls_id,
            imgsz,
            conf,
            iou,
            device,
            max_det,
            predict_batch_size,
            use_sahi=counterfactual_sahi,
            slice_size=slice_size,
            overlap=overlap,
            sahi_batch_size=sahi_batch_size,
            large_conf=large_conf,
            size_threshold=size_threshold,
            sahi_model_input_size=sahi_model_input_size,
        )
        variant_scores = {variant.name: score for variant, score, _, _ in variant_results}
        variant_ious = {variant.name: matched_iou for variant, _, matched_iou, _ in variant_results}
        variant_boxes = {variant.name: matched_box for variant, _, _, matched_box in variant_results}
        variant_assets: dict[str, str] = {}

        for variant, score, matched_iou, matched_box in variant_results:
            vh, vw = variant.image_bgr.shape[:2]
            variant_box = clamp_box(variant.target_box, vw, vh)
            variant_rgb = draw_box(
                bgr_to_rgb(variant.image_bgr),
                variant_box,
                f"{variant.name} {score:.3f}",
                (42, 157, 143),
            )
            save_rgb(det_dir / variant.asset_name, variant_rgb)
            rel_asset = str(Path("assets") / image_key / det_dir.name / variant.asset_name)
            variant_assets[variant.name] = rel_asset
            cf_record = {
                "image_key": image_key,
                "image_name": image_path.name,
                "detection_id": det.detection_id,
                "class_id": det.cls_id,
                "class_name": det.cls_name,
                "variant": variant.name,
                "score": float(score),
                "matched_iou": float(matched_iou),
                "score_drop": float(det.conf - score),
                "detection_present": bool(score > 0),
                "asset": rel_asset,
                "target_x1": float(variant.target_box[0]),
                "target_y1": float(variant.target_box[1]),
                "target_x2": float(variant.target_box[2]),
                "target_y2": float(variant.target_box[3]),
                "match_x1": float(matched_box[0]) if matched_box is not None else None,
                "match_y1": float(matched_box[1]) if matched_box is not None else None,
                "match_x2": float(matched_box[2]) if matched_box is not None else None,
                "match_y2": float(matched_box[3]) if matched_box is not None else None,
            }
            counterfactual_rows.append(cf_record)

        (det_dir / "counterfactuals.json").write_text(
            json.dumps([row for row in counterfactual_rows if row["detection_id"] == det.detection_id], indent=2)
        )
        for variant_name, legacy_name in {
            "crop": "crop.png",
            "object_only": "object_only.png",
            "object_masked": "object_masked.png",
        }.items():
            variant = next(v for v in variants if v.name == variant_name)
            save_rgb(det_dir / legacy_name, bgr_to_rgb(variant.image_bgr))

        crop_score = variant_scores.get("crop", 0.0)
        crop_iou = variant_ious.get("crop", 0.0)
        crop_box_xyxy = variant_boxes.get("crop")
        object_only_score = variant_scores.get("object_only", 0.0)
        object_only_iou = variant_ious.get("object_only", 0.0)
        object_only_box = variant_boxes.get("object_only")
        object_masked_score = variant_scores.get("object_masked", 0.0)
        object_masked_iou = variant_ious.get("object_masked", 0.0)
        object_masked_box = variant_boxes.get("object_masked")

        occlusion_heatmap, occlusion_meta = compute_occlusion_sensitivity(
            model=model,
            image_bgr=image_bgr,
            target_box=det.box_xyxy,
            target_cls=det.cls_id,
            original_score=det.conf,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            max_det=max_det,
            patch_size=occlusion_patch,
            stride=occlusion_stride,
            batch_size=predict_batch_size,
            scope=occlusion_scope,
            roi_margin=occlusion_roi_margin,
            use_sahi=counterfactual_sahi,
            slice_size=slice_size,
            overlap=overlap,
            sahi_batch_size=sahi_batch_size,
            large_conf=large_conf,
            size_threshold=size_threshold,
            sahi_model_input_size=sahi_model_input_size,
        )
        drise_heatmap, drise_meta = compute_drise(
            model=model,
            image_bgr=image_bgr,
            target_box=det.box_xyxy,
            target_cls=det.cls_id,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            max_det=max_det,
            prediction_conf=drise_prediction_conf,
            num_masks=drise_masks,
            min_masks=drise_min_masks,
            target_positive_masks=drise_target_positive_masks,
            stop_if_no_signal_after=drise_stop_if_no_signal_after,
            grid_size=drise_grid_size,
            mask_prob=drise_mask_prob,
            batch_size=predict_batch_size,
            seed=drise_seed + det.detection_id,
            use_sahi=counterfactual_sahi,
            slice_size=slice_size,
            overlap=overlap,
            sahi_batch_size=sahi_batch_size,
            large_conf=large_conf,
            size_threshold=size_threshold,
            sahi_model_input_size=sahi_model_input_size,
        )
        if robustness:
            robustness_results = run_robustness_variants(
                model,
                build_robustness_variants(image_bgr, det.box_xyxy),
                det.cls_id,
                imgsz,
                conf,
                iou,
                device,
                max_det,
                predict_batch_size,
            )
            for variant, score, matched_iou, matched_box in robustness_results:
                robustness_rows.append(
                    {
                        "image_key": image_key,
                        "image_name": image_path.name,
                        "detection_id": det.detection_id,
                        "class_id": det.cls_id,
                        "class_name": det.cls_name,
                        "transform": variant.transform,
                        "original_score": float(det.conf),
                        "score": float(score),
                        "score_drop": float(det.conf - score),
                        "score_retention": float(score / det.conf) if det.conf > 0 else 0.0,
                        "matched_iou": float(matched_iou),
                        "detection_present": bool(score > 0),
                        "match_x1": float(matched_box[0]) if matched_box is not None else None,
                        "match_y1": float(matched_box[1]) if matched_box is not None else None,
                        "match_x2": float(matched_box[2]) if matched_box is not None else None,
                        "match_y2": float(matched_box[3]) if matched_box is not None else None,
                    }
                )
        gradcam_heatmap, gradcam_meta = compute_gradcam(
            model=model,
            image_bgr=image_bgr,
            target_box=det.box_xyxy,
            target_cls=det.cls_id,
            imgsz=imgsz,
            device=device,
        )
        sanity_meta: dict[str, Any] = {}
        if sanity_checks:
            randomized_heatmap = None
            randomized_status = "not_run_by_default"
            randomized_corr = None
            if randomized_weights_sanity:
                randomized_heatmap, randomized_meta = compute_randomized_head_gradcam(
                    model=model,
                    image_bgr=image_bgr,
                    target_box=det.box_xyxy,
                    target_cls=det.cls_id,
                    imgsz=imgsz,
                    device=device,
                )
                randomized_status = randomized_meta.get("randomized_weights_check", "run")
                randomized_corr = heatmap_correlation(gradcam_heatmap, randomized_heatmap)
            deletion_meta = run_deletion_sanity_check(
                model=model,
                image_bgr=image_bgr,
                target_box=det.box_xyxy,
                target_cls=det.cls_id,
                original_score=det.conf,
                important_patch=occlusion_meta.get("occlusion_peak_patch_xyxy"),
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                max_det=max_det,
                seed=drise_seed + det.detection_id + 1000,
            )
            sanity_meta = {
                "image_key": image_key,
                "image_name": image_path.name,
                "detection_id": det.detection_id,
                "class_id": det.cls_id,
                "class_name": det.cls_name,
                "occlusion_gradcam_corr": heatmap_correlation(occlusion_heatmap, gradcam_heatmap),
                "occlusion_drise_corr": heatmap_correlation(occlusion_heatmap, drise_heatmap),
                "gradcam_drise_corr": heatmap_correlation(gradcam_heatmap, drise_heatmap),
                "target_change_check": "compare_across_detection_rows" if len(detections) > 1 else "needs_multiple_detections",
                "randomized_weights_check": randomized_status,
                "gradcam_randomized_corr": randomized_corr,
                **deletion_meta,
            }
            sanity_rows.append(sanity_meta)

        target_rgb = draw_box(image_rgb, det.box_xyxy, f"{det.cls_name} {det.conf:.3f}", (230, 57, 70))
        occlusion_overlay = draw_box(
            overlay_heatmap(image_rgb, occlusion_heatmap),
            det.box_xyxy,
            "occlusion",
            (255, 255, 255),
        )

        save_rgb(det_dir / "target.png", target_rgb)
        save_rgb(det_dir / "occlusion_heatmap.png", occlusion_overlay)

        drise_asset = None
        if drise_heatmap is not None:
            drise_label = "d-rise" if drise_meta.get("drise_weight_sum", 0.0) > 0 else "d-rise no signal"
            drise_overlay = draw_box(
                overlay_heatmap(image_rgb, drise_heatmap),
                det.box_xyxy,
                drise_label,
                (255, 255, 255),
            )
            save_rgb(det_dir / "drise_heatmap.png", drise_overlay)
            drise_asset = str(Path("assets") / image_key / det_dir.name / "drise_heatmap.png")

        gradcam_asset = None
        if gradcam_heatmap is not None:
            gradcam_label = "gradcam" if gradcam_meta.get("gradcam_has_signal", True) else "gradcam no signal"
            gradcam_overlay = draw_box(
                overlay_heatmap(image_rgb, gradcam_heatmap),
                det.box_xyxy,
                gradcam_label,
                (255, 255, 255),
            )
            save_rgb(det_dir / "gradcam.png", gradcam_overlay)
            gradcam_asset = str(Path("assets") / image_key / det_dir.name / "gradcam.png")

        gt_record = gt_matches.get(
            det.detection_id,
            {
                "gt_status": None,
                "fp_type": None,
                "matched_gt_id": None,
                "matched_gt_class_id": None,
                "matched_gt_class_name": None,
                "matched_gt_iou": None,
                "localization_error": None,
                "best_gt_id": None,
                "best_gt_class_id": None,
                "best_gt_class_name": None,
                "best_gt_iou": None,
            },
        )
        candidate_match = source_candidate_matches.get(det.detection_id)
        matched_candidate = candidate_match[0] if candidate_match is not None else None
        matched_candidate_iou = candidate_match[1] if candidate_match is not None else None

        record = {
            "image_key": image_key,
            "image_name": image_path.name,
            "detection_id": det.detection_id,
            "class_id": det.cls_id,
            "class_name": det.cls_name,
            "original_conf": det.conf,
            "gt_available": gt_available,
            "gt_iou_threshold": gt_iou_threshold if gt_available else None,
            "sahi": use_sahi,
            "counterfactual_sahi": counterfactual_sahi,
            "matched_candidate_branch": matched_candidate.branch if matched_candidate is not None else None,
            "matched_candidate_id": matched_candidate.candidate_id if matched_candidate is not None else None,
            "matched_candidate_score": matched_candidate.score if matched_candidate is not None else None,
            "matched_candidate_iou": matched_candidate_iou,
            **gt_record,
            "x1": float(det.box_xyxy[0]),
            "y1": float(det.box_xyxy[1]),
            "x2": float(det.box_xyxy[2]),
            "y2": float(det.box_xyxy[3]),
            "crop_score_same_class": float(crop_score),
            "crop_iou": float(crop_iou),
            "object_only_score": float(object_only_score),
            "object_only_iou": float(object_only_iou),
            "object_masked_score": float(object_masked_score),
            "object_masked_iou": float(object_masked_iou),
            "patch_on_mean_score": float(variant_scores.get("patch_on_mean", 0.0)),
            "target_blurred_score": float(variant_scores.get("target_blurred", 0.0)),
            "background_blurred_score": float(variant_scores.get("background_blurred", 0.0)),
            "brightness_up_score": float(variant_scores.get("brightness_up", 0.0)),
            "brightness_down_score": float(variant_scores.get("brightness_down", 0.0)),
            "shifted_crop_score": float(variant_scores.get("shifted_crop", 0.0)),
            "object_resized_70_score": float(variant_scores.get("object_resized_70", 0.0)),
            "mask_drop": float(det.conf - object_masked_score),
            "context_drop": float(det.conf - object_only_score),
            "crop_retention": float(crop_score / det.conf) if det.conf > 0 else 0.0,
            "target_asset": str(Path("assets") / image_key / det_dir.name / "target.png"),
            "crop_asset": str(Path("assets") / image_key / det_dir.name / "crop.png"),
            "object_only_asset": str(Path("assets") / image_key / det_dir.name / "object_only.png"),
            "object_masked_asset": str(Path("assets") / image_key / det_dir.name / "object_masked.png"),
            "occlusion_asset": str(Path("assets") / image_key / det_dir.name / "occlusion_heatmap.png"),
            "drise_asset": drise_asset,
            "gradcam_asset": gradcam_asset,
            "counterfactuals_file": str(Path("assets") / image_key / det_dir.name / "counterfactuals.json"),
            "counterfactual_original_asset": variant_assets.get("original"),
            "counterfactual_background_blurred_asset": variant_assets.get("background_blurred"),
            "counterfactual_target_blurred_asset": variant_assets.get("target_blurred"),
            "object_only_match_x1": float(object_only_box[0]) if object_only_box is not None else None,
            "object_only_match_y1": float(object_only_box[1]) if object_only_box is not None else None,
            "object_only_match_x2": float(object_only_box[2]) if object_only_box is not None else None,
            "object_only_match_y2": float(object_only_box[3]) if object_only_box is not None else None,
            "object_masked_match_x1": float(object_masked_box[0]) if object_masked_box is not None else None,
            "object_masked_match_y1": float(object_masked_box[1]) if object_masked_box is not None else None,
            "object_masked_match_x2": float(object_masked_box[2]) if object_masked_box is not None else None,
            "object_masked_match_y2": float(object_masked_box[3]) if object_masked_box is not None else None,
            "crop_match_x1": float(crop_box_xyxy[0]) if crop_box_xyxy is not None else None,
            "crop_match_y1": float(crop_box_xyxy[1]) if crop_box_xyxy is not None else None,
            "crop_match_x2": float(crop_box_xyxy[2]) if crop_box_xyxy is not None else None,
            "crop_match_y2": float(crop_box_xyxy[3]) if crop_box_xyxy is not None else None,
            **occlusion_meta,
            **drise_meta,
            **gradcam_meta,
            **{
                key: value
                for key, value in sanity_meta.items()
                if key
                not in {
                    "image_key",
                    "image_name",
                    "detection_id",
                    "class_id",
                    "class_name",
                }
            },
        }
        (det_dir / "report.json").write_text(json.dumps(record, indent=2))
        write_detection_markdown_report(
            det_dir / "report.md",
            record,
            [row for row in counterfactual_rows if row["detection_id"] == det.detection_id],
        )
        detection_rows.append(record)

    return (
        image_summary,
        detection_rows,
        candidate_rows,
        postprocess_rows,
        counterfactual_rows,
        robustness_rows,
        sanity_rows,
    )


def write_run_outputs(
    run_dir: Path,
    run_manifest: dict[str, Any],
    images_rows: list[dict[str, Any]],
    detection_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]] | None = None,
    postprocess_rows: list[dict[str, Any]] | None = None,
    counterfactual_rows: list[dict[str, Any]] | None = None,
    robustness_rows: list[dict[str, Any]] | None = None,
    sanity_rows: list[dict[str, Any]] | None = None,
) -> None:
    candidate_rows = candidate_rows or []
    postprocess_rows = postprocess_rows or []
    counterfactual_rows = counterfactual_rows or []
    robustness_rows = robustness_rows or []
    sanity_rows = sanity_rows or []
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(run_manifest, indent=2))
    pl.DataFrame(images_rows, infer_schema_length=None).write_parquet(run_dir / "images.parquet")
    if detection_rows:
        pl.DataFrame(detection_rows, infer_schema_length=None).write_parquet(run_dir / "detections.parquet")
    else:
        pl.DataFrame(
            {
                "image_key": [],
                "image_name": [],
                "detection_id": [],
                "class_id": [],
                "class_name": [],
                "original_conf": [],
            }
        ).write_parquet(run_dir / "detections.parquet")
    write_rows_parquet(
        run_dir / "raw_candidates.parquet",
        candidate_rows,
        [
            "image_key",
            "image_name",
            "branch",
            "candidate_id",
            "class_id",
            "class_name",
            "class_score",
            "x1",
            "y1",
            "x2",
            "y2",
            "kept_after_threshold",
            "kept_after_nms",
        ],
    )
    write_rows_parquet(
        run_dir / "postprocess.parquet",
        postprocess_rows,
        [
            "image_key",
            "image_name",
            "pipeline",
            "final_detection_id",
            "class_id",
            "class_name",
            "score",
            "matched_candidate_id",
            "gt_status",
            "fp_type",
        ],
    )
    write_rows_parquet(
        run_dir / "counterfactuals.parquet",
        counterfactual_rows,
        [
            "image_key",
            "image_name",
            "detection_id",
            "class_id",
            "class_name",
            "variant",
            "score",
            "matched_iou",
            "score_drop",
            "detection_present",
        ],
    )
    write_rows_parquet(
        run_dir / "robustness.parquet",
        robustness_rows,
        [
            "image_key",
            "image_name",
            "detection_id",
            "class_id",
            "class_name",
            "transform",
            "original_score",
            "score",
            "score_drop",
            "score_retention",
            "matched_iou",
            "detection_present",
        ],
    )
    write_rows_parquet(
        run_dir / "sanity_checks.parquet",
        sanity_rows,
        [
            "image_key",
            "image_name",
            "detection_id",
            "class_id",
            "class_name",
            "occlusion_gradcam_corr",
            "occlusion_drise_corr",
            "gradcam_drise_corr",
            "gradcam_randomized_corr",
            "deletion_important_drop",
            "deletion_random_drop",
            "deletion_check_pass",
        ],
    )
    dataset_rows, dataset_box_rows, leakage_rows, dataset_summary = run_dataset_audit_from_manifest(
        run_manifest,
        images_rows,
    )
    write_rows_parquet(
        run_dir / "dataset_audit.parquet",
        dataset_rows,
        [
            "dataset_role",
            "image_path",
            "image_name",
            "image_width",
            "image_height",
            "num_labels",
            "missing_label_file",
            "empty_label_file",
            "blur_laplacian",
            "brightness",
            "contrast",
            "scene_signature",
        ],
    )
    write_rows_parquet(
        run_dir / "dataset_boxes.parquet",
        dataset_box_rows,
        [
            "dataset_role",
            "image_path",
            "image_name",
            "class_id",
            "class_name",
            "box_width",
            "box_height",
            "box_area",
            "min_side",
            "area_ratio",
            "tiny_box",
            "small_box",
            "loose_box_proxy",
            "extreme_aspect",
        ],
    )
    write_rows_parquet(
        run_dir / "train_val_leakage.parquet",
        leakage_rows,
        ["train_image_path", "source_image_path", "match_type", "scene_signature"],
    )
    (run_dir / "dataset_audit_summary.json").write_text(json.dumps(dataset_summary, indent=2))

    calibration_rows, threshold_rows, calibration_summary = compute_calibration_outputs(images_rows, postprocess_rows)
    write_rows_parquet(
        run_dir / "calibration.parquet",
        calibration_rows,
        ["class_name", "bin_start", "bin_end", "count", "accuracy", "mean_confidence", "ece_contribution"],
    )
    write_rows_parquet(
        run_dir / "threshold_sweep.parquet",
        threshold_rows,
        ["class_name", "threshold", "tp", "fp", "fn", "precision", "recall", "f1"],
    )
    (run_dir / "calibration_summary.json").write_text(json.dumps(calibration_summary, indent=2))
    write_calibration_plots(run_dir / "reports", calibration_rows, threshold_rows)

    failure_rows, failure_summary_rows = compute_failure_clusters(postprocess_rows)
    write_rows_parquet(
        run_dir / "failure_clusters.parquet",
        failure_rows,
        ["image_key", "image_name", "final_detection_id", "class_name", "score", "cluster", "cluster_label"],
    )
    write_rows_parquet(
        run_dir / "failure_cluster_summary.parquet",
        failure_summary_rows,
        ["cluster", "cluster_label", "count", "mean_score", "mean_box_area", "top_class", "top_region"],
    )

    recommendations = build_fix_recommendations(
        images_rows=images_rows,
        detection_rows=detection_rows,
        postprocess_rows=postprocess_rows,
        dataset_summary=dataset_summary,
        calibration_summary=calibration_summary,
        robustness_rows=robustness_rows,
        sanity_rows=sanity_rows,
        failure_summary_rows=failure_summary_rows,
    )
    (run_dir / "fix_recommendations.json").write_text(json.dumps(recommendations, indent=2))
    write_run_analysis_report(
        run_dir=run_dir,
        images_rows=images_rows,
        detection_rows=detection_rows,
        candidate_rows=candidate_rows,
        postprocess_rows=postprocess_rows,
        counterfactual_rows=counterfactual_rows,
        robustness_rows=robustness_rows,
        sanity_rows=sanity_rows,
        dataset_summary=dataset_summary,
        calibration_summary=calibration_summary,
        failure_summary_rows=failure_summary_rows,
        recommendations=recommendations,
    )


def write_rows_parquet(path: Path, rows: list[dict[str, Any]], empty_columns: list[str]) -> None:
    if rows:
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(path)
    else:
        pl.DataFrame({column: [] for column in empty_columns}).write_parquet(path)


def counter_table(counter: Counter, limit: int = 12) -> list[str]:
    if not counter:
        return ["_none_"]
    return [f"- `{key}`: {value}" for key, value in counter.most_common(limit)]


def score_bin(score: float) -> str:
    if score < 0.05:
        return "<0.05"
    if score < 0.15:
        return "0.05-0.15"
    if score < 0.30:
        return "0.15-0.30"
    if score < 0.50:
        return "0.30-0.50"
    return ">=0.50"


def area_bin(area: float) -> str:
    if area < 64:
        return "<64px2"
    if area < 256:
        return "64-256px2"
    if area < 1024:
        return "256-1024px2"
    if area < 4096:
        return "1024-4096px2"
    return ">=4096px2"


def write_bar_plot(path: Path, title: str, counter: Counter) -> None:
    if not counter:
        return
    try:
        import matplotlib.pyplot as plt

        labels, values = zip(*counter.most_common(), strict=False)
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.8), 4))
        ax.bar([str(label) for label in labels], values, color="#2a9d8f")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    except Exception:
        return


def write_score_histogram(path: Path, candidate_rows: list[dict[str, Any]]) -> None:
    if not candidate_rows:
        return
    try:
        import matplotlib.pyplot as plt

        all_scores = [float(row["class_score"]) for row in candidate_rows if row.get("class_score") is not None]
        kept_scores = [
            float(row["class_score"])
            for row in candidate_rows
            if row.get("class_score") is not None and row.get("kept_after_threshold")
        ]
        source_fp_scores = [
            float(row["class_score"])
            for row in candidate_rows
            if row.get("class_score") is not None and row.get("source_gt_status") == "fp"
        ]
        source_tp_scores = [
            float(row["class_score"])
            for row in candidate_rows
            if row.get("class_score") is not None and row.get("source_gt_status") == "tp"
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(0, 1, 41)
        if all_scores:
            ax.hist(all_scores, bins=bins, alpha=0.25, label="all candidates")
        if kept_scores:
            ax.hist(kept_scores, bins=bins, alpha=0.5, label="above threshold")
        if source_fp_scores:
            ax.hist(source_fp_scores, bins=bins, alpha=0.65, label="source FP")
        if source_tp_scores:
            ax.hist(source_tp_scores, bins=bins, alpha=0.65, label="source TP")
        ax.set_title("Candidate Score Histogram")
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    except Exception:
        return


def image_average_hash(image_bgr: np.ndarray, hash_size: int = 8) -> str:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    bits = resized > resized.mean()
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    if image_path.parent.name == "images":
        return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    return image_path.with_suffix(".txt")


def collect_dataset_pairs(path: Path, max_images: int | None) -> list[tuple[Path, Path | None]]:
    path = Path(path)
    image_paths: list[Path] = []
    if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
        data = parse_yolo_yaml(path)
        root = Path(data.get("path", path.parent))
        if not root.is_absolute():
            root = (path.parent / root).resolve()
        for split in ("train", "val", "valid", "test"):
            split_value = data.get(split)
            if split_value is None:
                continue
            values = split_value if isinstance(split_value, list) else [split_value]
            for value in values:
                split_path = Path(value)
                if not split_path.is_absolute():
                    split_path = root / split_path
                if split_path.is_dir():
                    image_paths.extend(
                        sorted(p for p in split_path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
                    )
                elif split_path.is_file() and split_path.suffix.lower() in IMAGE_SUFFIXES:
                    image_paths.append(split_path)
    elif path.is_dir():
        image_paths = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        image_paths = [path]

    if max_images is not None:
        image_paths = image_paths[:max_images]
    return [(image_path, label_path_for_image(image_path)) for image_path in image_paths]


def parse_label_file_for_audit(
    label_path: Path | None,
    image_shape: tuple[int, int],
    names: list[str],
) -> tuple[list[dict[str, Any]], int]:
    if label_path is None or not label_path.exists():
        return [], 0
    height, width = image_shape
    rows = []
    invalid_lines = 0
    for line_no, line in enumerate(label_path.read_text().splitlines(), start=1):
        parts = line.split()
        if len(parts) < 5:
            invalid_lines += 1
            continue
        try:
            cls_float, x_c, y_c, bw, bh = map(float, parts[:5])
        except ValueError:
            invalid_lines += 1
            continue
        cls_id = int(cls_float)
        invalid = (
            cls_float != cls_id
            or cls_id < 0
            or x_c < 0
            or x_c > 1
            or y_c < 0
            or y_c > 1
            or bw <= 0
            or bh <= 0
            or bw > 1.25
            or bh > 1.25
        )
        x1 = (x_c - bw / 2.0) * width
        y1 = (y_c - bh / 2.0) * height
        x2 = (x_c + bw / 2.0) * width
        y2 = (y_c + bh / 2.0) * height
        rows.append(
            {
                "line_no": line_no,
                "class_id": cls_id,
                "class_name": names[cls_id] if 0 <= cls_id < len(names) else str(cls_id),
                "invalid_label": invalid,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
        )
    return rows, invalid_lines


def audit_dataset_pairs(
    pairs: list[tuple[Path, Path | None]],
    *,
    role: str,
    names: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    image_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    for image_path, label_path in pairs:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            image_rows.append(
                {
                    "dataset_role": role,
                    "image_path": str(image_path),
                    "image_name": image_path.name,
                    "read_error": True,
                    "missing_label_file": label_path is None or not label_path.exists(),
                    "empty_label_file": None,
                    "num_labels": None,
                }
            )
            continue
        height, width = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        contrast = float(gray.std())
        label_rows, invalid_lines = parse_label_file_for_audit(label_path, (height, width), names)
        scene_signature = image_average_hash(image_bgr)
        class_names = sorted({row["class_name"] for row in label_rows})
        missing_label = label_path is None or not label_path.exists()
        empty_label = (not missing_label) and len(label_rows) == 0
        image_rows.append(
            {
                "dataset_role": role,
                "image_path": str(image_path),
                "image_name": image_path.name,
                "label_path": str(label_path) if label_path is not None else None,
                "read_error": False,
                "image_width": width,
                "image_height": height,
                "num_labels": len(label_rows),
                "class_names": ",".join(class_names),
                "missing_label_file": missing_label,
                "empty_label_file": empty_label,
                "invalid_label_lines": invalid_lines,
                "blur_laplacian": blur,
                "brightness": brightness,
                "contrast": contrast,
                "blurred_image_proxy": blur < 60.0,
                "low_contrast_proxy": contrast < 25.0,
                "scene_signature": scene_signature,
                "brightness_bin": "dark" if brightness < 75 else "bright" if brightness > 180 else "normal",
            }
        )
        boxes = []
        for row in label_rows:
            box = clamp_box(np.array([row["x1"], row["y1"], row["x2"], row["y2"]], dtype=np.float32), width, height)
            boxes.append((row, box))

        duplicate_flags = [False] * len(boxes)
        for i, (row_i, box_i) in enumerate(boxes):
            for j in range(i + 1, len(boxes)):
                row_j, box_j = boxes[j]
                if row_i["class_id"] == row_j["class_id"] and box_iou_xyxy(box_i, box_j) > 0.95:
                    duplicate_flags[i] = True
                    duplicate_flags[j] = True

        for idx, (row, box) in enumerate(boxes):
            geom = box_geometry(box, (height, width))
            min_side = min(geom["box_width"], geom["box_height"])
            area_ratio = geom["box_area"] / max(1.0, float(width * height))
            aspect = geom["aspect_ratio"] or 0.0
            x1, y1, x2, y2 = box
            crop = gray[int(y1) : int(y2), int(x1) : int(x2)]
            edge_density = None
            if crop.size:
                edges = cv2.Canny(crop, 80, 160)
                edge_density = float((edges > 0).mean())
            box_rows.append(
                {
                    "dataset_role": role,
                    "image_path": str(image_path),
                    "image_name": image_path.name,
                    "label_path": str(label_path) if label_path is not None else None,
                    "line_no": row["line_no"],
                    "class_id": row["class_id"],
                    "class_name": row["class_name"],
                    "invalid_label": row["invalid_label"],
                    "min_side": float(min_side),
                    "area_ratio": float(area_ratio),
                    "tiny_box": bool(min_side < 8),
                    "small_box": bool(min_side < 16),
                    "extreme_aspect": bool(aspect > 4.0 or (aspect > 0 and aspect < 0.25)),
                    "border_touching": bool(x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1),
                    "duplicate_label": duplicate_flags[idx],
                    "edge_density": edge_density,
                    "loose_box_proxy": bool(area_ratio > 0.25 or (edge_density is not None and edge_density < 0.015)),
                    "image_blur_laplacian": blur,
                    "image_brightness": brightness,
                    "image_contrast": contrast,
                    "scene_signature": scene_signature,
                    **geom,
                }
            )
    return image_rows, box_rows


def summarize_dataset_audit(
    dataset_rows: list[dict[str, Any]],
    dataset_box_rows: list[dict[str, Any]],
    leakage_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    roles = sorted({row.get("dataset_role") for row in dataset_rows if row.get("dataset_role") is not None})
    summary: dict[str, Any] = {"roles": {}, "leakage_count": len(leakage_rows)}
    for role in roles:
        role_images = [row for row in dataset_rows if row.get("dataset_role") == role]
        role_boxes = [row for row in dataset_box_rows if row.get("dataset_role") == role]
        class_counts = Counter(row.get("class_name") for row in role_boxes)
        scene_by_class: dict[str, list[dict[str, Any]]] = {}
        for class_name in class_counts:
            scenes = Counter(
                row.get("scene_signature")
                for row in role_boxes
                if row.get("class_name") == class_name and row.get("scene_signature")
            )
            scene_by_class[str(class_name)] = [
                {"scene_signature": str(scene), "count": int(count)} for scene, count in scenes.most_common(5)
            ]
        summary["roles"][str(role)] = {
            "images": len(role_images),
            "boxes": len(role_boxes),
            "class_counts": dict(class_counts),
            "missing_label_files": sum(bool(row.get("missing_label_file")) for row in role_images),
            "empty_label_files": sum(bool(row.get("empty_label_file")) for row in role_images),
            "invalid_label_lines": sum(int(row.get("invalid_label_lines") or 0) for row in role_images),
            "hard_negative_images": sum(bool(row.get("empty_label_file")) for row in role_images),
            "blurred_images_proxy": sum(bool(row.get("blurred_image_proxy")) for row in role_images),
            "low_contrast_images_proxy": sum(bool(row.get("low_contrast_proxy")) for row in role_images),
            "tiny_boxes_lt8": sum(bool(row.get("tiny_box")) for row in role_boxes),
            "small_boxes_lt16": sum(bool(row.get("small_box")) for row in role_boxes),
            "duplicate_labels": sum(bool(row.get("duplicate_label")) for row in role_boxes),
            "loose_box_proxy": sum(bool(row.get("loose_box_proxy")) for row in role_boxes),
            "extreme_aspect_boxes": sum(bool(row.get("extreme_aspect")) for row in role_boxes),
            "background_class_correlation_proxy": scene_by_class,
        }
    return summary


def run_dataset_audit_from_manifest(
    manifest: dict[str, Any],
    images_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    names = [str(name) for name in manifest.get("class_names", [])]
    max_images = manifest.get("audit_max_images")
    source_pairs = [
        (Path(row["image_path"]), Path(row["label_path"]) if row.get("label_path") else label_path_for_image(Path(row["image_path"])))
        for row in images_rows
        if row.get("image_path")
    ]
    if max_images is not None:
        source_pairs = source_pairs[: int(max_images)]
    source_rows, source_box_rows = audit_dataset_pairs(source_pairs, role="source", names=names)

    audit_rows: list[dict[str, Any]] = []
    audit_box_rows: list[dict[str, Any]] = []
    audit_data = manifest.get("audit_data")
    if audit_data:
        audit_pairs = collect_dataset_pairs(Path(audit_data), int(max_images) if max_images is not None else None)
        audit_rows, audit_box_rows = audit_dataset_pairs(audit_pairs, role="audit", names=names)

    source_by_signature = {
        row.get("scene_signature"): row for row in source_rows if row.get("scene_signature") and not row.get("read_error")
    }
    leakage_rows = []
    for row in audit_rows:
        signature = row.get("scene_signature")
        source_row = source_by_signature.get(signature)
        if source_row is not None:
            leakage_rows.append(
                {
                    "train_image_path": row.get("image_path"),
                    "source_image_path": source_row.get("image_path"),
                    "match_type": "average_hash_exact",
                    "scene_signature": signature,
                }
            )

    dataset_rows = source_rows + audit_rows
    dataset_box_rows = source_box_rows + audit_box_rows
    summary = summarize_dataset_audit(dataset_rows, dataset_box_rows, leakage_rows)
    return dataset_rows, dataset_box_rows, leakage_rows, summary


def source_postprocess_rows(postprocess_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in postprocess_rows if str(row.get("pipeline", "")).startswith("source")]


def compute_calibration_outputs(
    images_rows: list[dict[str, Any]],
    postprocess_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = [row for row in source_postprocess_rows(postprocess_rows) if row.get("gt_status") in {"tp", "fp"}]
    if not rows:
        return [], [], {"ece": None, "num_predictions": 0}

    total_gt = int(sum(int(row.get("num_ground_truth") or 0) for row in images_rows))
    calibration_rows = []
    ece = 0.0
    for class_name in ["all", *sorted({str(row.get("class_name")) for row in rows})]:
        class_rows = rows if class_name == "all" else [row for row in rows if str(row.get("class_name")) == class_name]
        if not class_rows:
            continue
        n = len(class_rows)
        for bin_idx in range(10):
            bin_start = bin_idx / 10.0
            bin_end = (bin_idx + 1) / 10.0
            in_bin = [
                row
                for row in class_rows
                if float(row.get("score") or 0.0) >= bin_start
                and (float(row.get("score") or 0.0) < bin_end or bin_idx == 9)
            ]
            if not in_bin:
                continue
            accuracy = sum(row.get("gt_status") == "tp" for row in in_bin) / len(in_bin)
            mean_conf = sum(float(row.get("score") or 0.0) for row in in_bin) / len(in_bin)
            contribution = len(in_bin) / n * abs(accuracy - mean_conf)
            if class_name == "all":
                ece += contribution
            calibration_rows.append(
                {
                    "class_name": class_name,
                    "bin_start": bin_start,
                    "bin_end": bin_end,
                    "count": len(in_bin),
                    "accuracy": float(accuracy),
                    "mean_confidence": float(mean_conf),
                    "ece_contribution": float(contribution),
                }
            )

    threshold_rows = []
    thresholds = np.linspace(0.0, 1.0, 101)
    for class_name in ["all", *sorted({str(row.get("class_name")) for row in rows})]:
        class_rows = rows if class_name == "all" else [row for row in rows if str(row.get("class_name")) == class_name]
        if not class_rows:
            continue
        for threshold in thresholds:
            kept = [row for row in class_rows if float(row.get("score") or 0.0) >= threshold]
            tp = sum(row.get("gt_status") == "tp" for row in kept)
            fp = sum(row.get("gt_status") == "fp" for row in kept)
            if class_name == "all":
                fn = max(0, total_gt - tp)
                recall = tp / total_gt if total_gt > 0 else None
            else:
                fn = None
                recall = None
            precision = tp / (tp + fp) if tp + fp > 0 else None
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and precision + recall > 0
                else None
            )
            threshold_rows.append(
                {
                    "class_name": class_name,
                    "threshold": float(threshold),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    best_all = max(
        (row for row in threshold_rows if row["class_name"] == "all" and row["f1"] is not None),
        key=lambda row: row["f1"],
        default=None,
    )
    summary = {
        "ece": float(ece),
        "num_predictions": len(rows),
        "num_ground_truth": total_gt,
        "best_all_threshold": best_all,
    }
    return calibration_rows, threshold_rows, summary


def write_calibration_plots(reports_dir: Path, calibration_rows: list[dict[str, Any]], threshold_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt

        all_cal = [row for row in calibration_rows if row.get("class_name") == "all"]
        if all_cal:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot([0, 1], [0, 1], "--", color="#777777", label="perfect")
            ax.scatter(
                [row["mean_confidence"] for row in all_cal],
                [row["accuracy"] for row in all_cal],
                s=[max(20, row["count"] * 3) for row in all_cal],
                color="#2a9d8f",
                alpha=0.8,
                label="observed",
            )
            ax.set_xlabel("mean confidence")
            ax.set_ylabel("accuracy")
            ax.set_title("Reliability Diagram")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            reports_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(reports_dir / "reliability_diagram.png")
            plt.close(fig)

        pr_rows = [row for row in threshold_rows if row.get("class_name") == "all" and row.get("precision") is not None and row.get("recall") is not None]
        if pr_rows:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot([row["recall"] for row in pr_rows], [row["precision"] for row in pr_rows], color="#264653")
            ax.set_xlabel("recall")
            ax.set_ylabel("precision")
            ax.set_title("Precision-Recall Curve")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            reports_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(reports_dir / "precision_recall_curve.png")
            plt.close(fig)
    except Exception:
        return


def compute_failure_clusters(
    postprocess_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fps = [row for row in source_postprocess_rows(postprocess_rows) if row.get("gt_status") == "fp"]
    if len(fps) < 3:
        return [], []
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        features = np.array(
            [
                [
                    float(row.get("score") or 0.0),
                    math.log1p(float(row.get("box_area") or 0.0)),
                    min(10.0, float(row.get("aspect_ratio") or 0.0)),
                    float(row.get("center_x_norm") or 0.0),
                    float(row.get("center_y_norm") or 0.0),
                    float(row.get("class_id") or 0.0),
                ]
                for row in fps
            ],
            dtype=np.float32,
        )
        features = StandardScaler().fit_transform(features)
        n_clusters = min(5, max(2, int(round(math.sqrt(len(fps) / 2)))))
        labels = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit_predict(features)
    except Exception:
        labels = np.zeros(len(fps), dtype=int)

    cluster_rows = []
    for row, cluster in zip(fps, labels, strict=False):
        cluster_rows.append(
            {
                "image_key": row.get("image_key"),
                "image_name": row.get("image_name"),
                "final_detection_id": row.get("final_detection_id"),
                "class_id": row.get("class_id"),
                "class_name": row.get("class_name"),
                "score": row.get("score"),
                "box_area": row.get("box_area"),
                "aspect_ratio": row.get("aspect_ratio"),
                "image_region": row.get("image_region"),
                "cluster": int(cluster),
                "cluster_label": f"failure_cluster_{int(cluster)}",
            }
        )

    summary_rows = []
    for cluster in sorted(set(int(v) for v in labels)):
        members = [row for row in cluster_rows if row["cluster"] == cluster]
        top_class = Counter(row.get("class_name") for row in members).most_common(1)[0][0]
        top_region = Counter(row.get("image_region") for row in members).most_common(1)[0][0]
        summary_rows.append(
            {
                "cluster": cluster,
                "cluster_label": f"failure_cluster_{cluster}",
                "count": len(members),
                "mean_score": float(np.mean([float(row.get("score") or 0.0) for row in members])),
                "mean_box_area": float(np.mean([float(row.get("box_area") or 0.0) for row in members])),
                "top_class": top_class,
                "top_region": top_region,
            }
        )
    return cluster_rows, summary_rows


def build_fix_recommendations(
    *,
    images_rows: list[dict[str, Any]],
    detection_rows: list[dict[str, Any]],
    postprocess_rows: list[dict[str, Any]],
    dataset_summary: dict[str, Any],
    calibration_summary: dict[str, Any],
    robustness_rows: list[dict[str, Any]],
    sanity_rows: list[dict[str, Any]],
    failure_summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    source_rows = source_postprocess_rows(postprocess_rows)
    fps = [row for row in source_rows if row.get("gt_status") == "fp"]
    low_fp = [row for row in fps if float(row.get("score") or 0.0) < 0.15]
    high_bg_fp = [
        row
        for row in fps
        if float(row.get("score") or 0.0) >= 0.5 and row.get("fp_type") in {None, "background"}
    ]
    if fps and len(low_fp) / len(fps) > 0.5:
        recommendations.append(
            {
                "priority": "high",
                "diagnosis": "Low-score false positives dominate.",
                "evidence": f"{len(low_fp)}/{len(fps)} source false positives are below confidence 0.15.",
                "action": "Use a higher or class-specific threshold, then mine hard negatives near the deployment threshold.",
            }
        )
    if high_bg_fp:
        recommendations.append(
            {
                "priority": "high",
                "diagnosis": "High-score background false positives exist.",
                "evidence": f"{len(high_bg_fp)} background false positives have score >= 0.50.",
                "action": "Add hard-negative images with similar backgrounds and inspect labels for missing objects.",
            }
        )
    ece = calibration_summary.get("ece")
    if ece is not None and ece > 0.10:
        recommendations.append(
            {
                "priority": "medium",
                "diagnosis": "Confidence is poorly calibrated.",
                "evidence": f"Expected calibration error is {ece:.3f}.",
                "action": "Choose thresholds from validation PR curves and consider temperature/isotonic calibration for scores.",
            }
        )
    audit_role = dataset_summary.get("roles", {}).get("audit") or dataset_summary.get("roles", {}).get("source", {})
    hard_negatives = int(audit_role.get("hard_negative_images") or 0)
    images = int(audit_role.get("images") or 0)
    if images and hard_negatives / images < 0.02:
        recommendations.append(
            {
                "priority": "high",
                "diagnosis": "Dataset has few explicit hard negatives.",
                "evidence": f"{hard_negatives}/{images} audited images have empty labels.",
                "action": "Add background-only frames from the same cameras/scenes where the model currently fires.",
            }
        )
    if int(audit_role.get("loose_box_proxy") or 0) > 0:
        recommendations.append(
            {
                "priority": "medium",
                "diagnosis": "Some boxes may include excessive background.",
                "evidence": f"{audit_role.get('loose_box_proxy')} audited boxes triggered the loose-box proxy.",
                "action": "Review large/low-edge-density labels and tighten boxes where possible.",
            }
        )
    if dataset_summary.get("leakage_count", 0):
        recommendations.append(
            {
                "priority": "high",
                "diagnosis": "Potential train/validation scene leakage.",
                "evidence": f"{dataset_summary['leakage_count']} exact average-hash matches found.",
                "action": "Split by video/scene before retraining and re-run validation.",
            }
        )
    if robustness_rows:
        by_transform: dict[str, list[float]] = {}
        for row in robustness_rows:
            by_transform.setdefault(str(row.get("transform")), []).append(float(row.get("score_retention") or 0.0))
        weak = {k: float(np.mean(v)) for k, v in by_transform.items() if v and float(np.mean(v)) < 0.5}
        if weak:
            recommendations.append(
                {
                    "priority": "medium",
                    "diagnosis": "Detector is fragile under some controlled transforms.",
                    "evidence": ", ".join(f"{k}: {v:.2f}" for k, v in sorted(weak.items())),
                    "action": "Add matching augmentations and real examples for the weakest camera/preprocessing conditions.",
                }
            )
    failed_deletion = [row for row in sanity_rows if row.get("deletion_check_pass") is False]
    if failed_deletion:
        recommendations.append(
            {
                "priority": "medium",
                "diagnosis": "Explanation sanity check failed for some detections.",
                "evidence": f"{len(failed_deletion)} detections did not drop more when important pixels were deleted than random pixels.",
                "action": "Treat Grad-CAM/saliency cautiously; rely more on occlusion/counterfactual tests and inspect preprocessing.",
            }
        )
    if failure_summary_rows:
        recommendations.append(
            {
                "priority": "info",
                "diagnosis": "False positives split into failure clusters.",
                "evidence": f"{len(failure_summary_rows)} failure clusters were found.",
                "action": "Review each cluster separately; avoid one global fix if clusters have different class/region/size patterns.",
            }
        )
    if not recommendations:
        recommendations.append(
            {
                "priority": "info",
                "diagnosis": "No dominant automatic diagnosis was triggered.",
                "evidence": "Available audit/calibration/robustness checks did not cross built-in warning thresholds.",
                "action": "Inspect the candidate tables, heatmaps, and failure clusters manually before changing training data.",
            }
        )
    return recommendations


def write_run_analysis_report(
    *,
    run_dir: Path,
    images_rows: list[dict[str, Any]],
    detection_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    postprocess_rows: list[dict[str, Any]],
    counterfactual_rows: list[dict[str, Any]],
    robustness_rows: list[dict[str, Any]],
    sanity_rows: list[dict[str, Any]],
    dataset_summary: dict[str, Any],
    calibration_summary: dict[str, Any],
    failure_summary_rows: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
) -> None:
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    source_rows = [row for row in postprocess_rows if str(row.get("pipeline", "")).startswith("source")]
    source_fp = [row for row in source_rows if row.get("gt_status") == "fp"]
    source_tp = [row for row in source_rows if row.get("gt_status") == "tp"]
    candidates_above_threshold = [row for row in candidate_rows if row.get("kept_after_threshold")]
    candidates_kept = [row for row in candidate_rows if row.get("kept_after_nms")]
    source_matched_candidates = [row for row in candidate_rows if row.get("source_detection_id") is not None]

    fp_by_class = Counter(row.get("class_name") for row in source_fp)
    fp_by_area = Counter(area_bin(float(row.get("box_area") or 0.0)) for row in source_fp)
    fp_by_region = Counter(row.get("image_region") for row in source_fp)
    fp_by_score = Counter(score_bin(float(row.get("score") or 0.0)) for row in source_fp)
    candidate_by_branch = Counter(row.get("branch") for row in candidate_rows)

    write_score_histogram(reports_dir / "candidate_score_histogram.png", candidate_rows)
    write_bar_plot(reports_dir / "false_positives_by_class.png", "False Positives by Class", fp_by_class)
    write_bar_plot(reports_dir / "false_positives_by_area.png", "False Positives by Box Area", fp_by_area)
    write_bar_plot(reports_dir / "false_positives_by_region.png", "False Positives by Image Region", fp_by_region)

    failure_lines = [
        f"- cluster `{row.get('cluster')}`: {row.get('count')} FP, top class `{row.get('top_class')}`, top region `{row.get('top_region')}`"
        for row in failure_summary_rows
    ] or ["_none_"]

    lines = [
        "# Explainability Analysis Report",
        "",
        "## Coverage",
        "",
        "- model adapter: implemented via raw preprocess/infer/decode/postprocess path",
        "- raw candidate logging: `raw_candidates.parquet`",
        "- postprocess logging: `postprocess.parquet`",
        "- target resolver: final detections are matched back to raw candidates by class and IoU",
        "- occlusion: per-detection occlusion heatmaps",
        "- D-RISE: per-detection heatmaps when `drise_num_masks > 0`",
        "- Grad-CAM: per-detection overlays when a differentiable raw target is available",
        "- counterfactuals: `counterfactuals.parquet` plus per-detection variant images",
        "- dataset audit: `dataset_audit.parquet`, `dataset_boxes.parquet`, `dataset_audit_summary.json`",
        "- confidence calibration: `calibration.parquet`, `threshold_sweep.parquet`, `calibration_summary.json`",
        "- robustness tests: `robustness.parquet`",
        "- explanation sanity checks: `sanity_checks.parquet`",
        "- failure clustering: `failure_clusters.parquet`, `failure_cluster_summary.parquet`",
        "- fix recommendation matrix: `fix_recommendations.json`",
        "",
        "## Run Counts",
        "",
        f"- images: {len(images_rows)}",
        f"- explained detections: {len(detection_rows)}",
        f"- raw candidates logged: {len(candidate_rows)}",
        f"- candidates above threshold: {len(candidates_above_threshold)}",
        f"- candidates kept by adapter postprocess: {len(candidates_kept)}",
        f"- candidates matched to source detections: {len(source_matched_candidates)}",
        f"- source detections logged: {len(source_rows)}",
        f"- source TP: {len(source_tp)}",
        f"- source FP: {len(source_fp)}",
        f"- counterfactual rows: {len(counterfactual_rows)}",
        f"- robustness rows: {len(robustness_rows)}",
        f"- sanity-check rows: {len(sanity_rows)}",
        "",
        "## Candidate Branches",
        "",
        *counter_table(candidate_by_branch),
        "",
        "## False Positives By Class",
        "",
        *counter_table(fp_by_class),
        "",
        "## False Positives By Score",
        "",
        *counter_table(fp_by_score),
        "",
        "## False Positives By Box Area",
        "",
        *counter_table(fp_by_area),
        "",
        "## False Positives By Image Region",
        "",
        *counter_table(fp_by_region),
        "",
        "## Generated Plots",
        "",
        "- `reports/candidate_score_histogram.png`",
        "- `reports/false_positives_by_class.png`",
        "- `reports/false_positives_by_area.png`",
        "- `reports/false_positives_by_region.png`",
        "- `reports/reliability_diagram.png`",
        "- `reports/precision_recall_curve.png`",
        "",
        "## Dataset Audit",
        "",
        f"- roles: `{', '.join(dataset_summary.get('roles', {}).keys())}`",
        f"- potential train/source leakage matches: `{dataset_summary.get('leakage_count', 0)}`",
        "",
        "## Calibration",
        "",
        f"- expected calibration error: `{calibration_summary.get('ece')}`",
        f"- best all-class threshold row: `{calibration_summary.get('best_all_threshold')}`",
        "",
        "## Failure Clusters",
        "",
        *failure_lines,
        "",
        "## Fix Recommendation Matrix",
        "",
        *[
            f"- **{row.get('priority')}**: {row.get('diagnosis')} Evidence: {row.get('evidence')} Action: {row.get('action')}"
            for row in recommendations
        ],
        "",
    ]
    (run_dir / "analysis_report.md").write_text("\n".join(lines))
