import bisect
from collections import defaultdict

import numpy as np
from rich.console import Console
from rich.table import Table

from ssl_experiments.evaluation.iou import iou_one_vs_many as _iou_one_vs_many

console = Console()


# --- Size category definitions ---
SIZE_CATEGORIES = [
    ("tiny", 0, 8),
    ("small", 8, 16),
    ("medium", 16, 32),
    ("large", 32, 64),
    ("xlarge", 64, float("inf")),
]


# Lookup structures for fast size categorisation
_SIZE_BOUNDARIES = [low for _, low, _ in SIZE_CATEGORIES]  # [0, 8, 16, 32, 64]
_SIZE_NAMES = [name for name, _, _ in SIZE_CATEGORIES]  # ["tiny", "small", ...]


# --- Utility: Get object size category ---
def get_size_category(box):
    """Categorize object by min dimension (O(log n) via bisect)."""
    min_dim = min(box[2] - box[0], box[3] - box[1])
    idx = bisect.bisect_right(_SIZE_BOUNDARIES, min_dim) - 1
    idx = max(0, min(idx, len(_SIZE_NAMES) - 1))
    return _SIZE_NAMES[idx]


# --- Utility: Evaluate predictions ---
def evaluate_image(gt_boxes, gt_classes, pred_boxes, pred_classes, iou_thr):
    matched_gt = set()
    tp = fp = 0

    if len(gt_boxes) > 0:
        gt_arr = np.array(gt_boxes, dtype=np.float32)
        gt_cls_arr = np.array(gt_classes)

    for pb, pc in zip(pred_boxes, pred_classes, strict=True):
        best_iou, best_idx = 0, None
        if len(gt_boxes) > 0:
            # Only compare against same-class GTs
            cls_mask = gt_cls_arr == pc
            if cls_mask.any():
                ious = _iou_one_vs_many(pb, gt_arr)
                ious[~cls_mask] = 0
                ious[[i for i in matched_gt]] = 0  # mask already-matched
                best_idx_arr = int(np.argmax(ious))
                best_iou = float(ious[best_idx_arr])
                best_idx = best_idx_arr

        if best_iou >= iou_thr and best_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec + 1e-6) if (prec + rec) > 0 else 0.0
    return tp, fp, fn, prec, rec, f1


# --- Utility: Evaluate predictions by size and class ---
def evaluate_image_by_size_and_class(
    gt_boxes, gt_classes, pred_boxes, pred_classes, iou_thr, num_classes
):
    """Evaluate and return metrics grouped by object size AND class."""
    matched_gt = set()

    # Initialize stats: {class_id: {size_name: {tp, fp, fn, total_gt}}}
    stats = {}
    for cls_id in range(num_classes):
        stats[cls_id] = {
            name: {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0}
            for name, _, _ in SIZE_CATEGORIES
        }

    # Count GT by size and class
    gt_sizes = [get_size_category(box) for box in gt_boxes]
    for size, cls_id in zip(gt_sizes, gt_classes, strict=True):
        if cls_id in stats:
            stats[cls_id][size]["total_gt"] += 1

    if len(gt_boxes) > 0:
        gt_arr = np.array(gt_boxes, dtype=np.float32)
        gt_cls_arr = np.array(gt_classes)

    # Match predictions to GT
    for pb, pc in zip(pred_boxes, pred_classes, strict=True):
        best_iou, best_idx = 0, None
        if len(gt_boxes) > 0:
            cls_mask = gt_cls_arr == pc
            if cls_mask.any():
                ious = _iou_one_vs_many(pb, gt_arr)
                ious[~cls_mask] = 0
                ious[[i for i in matched_gt]] = 0
                best_idx_arr = int(np.argmax(ious))
                best_iou = float(ious[best_idx_arr])
                best_idx = best_idx_arr

        if best_iou >= iou_thr and best_idx not in matched_gt:
            matched_gt.add(best_idx)
            size = gt_sizes[best_idx]
            cls_id = gt_classes[best_idx]
            if cls_id in stats:
                stats[cls_id][size]["tp"] += 1
        else:
            pred_size = get_size_category(pb)
            if pc in stats:
                stats[pc][pred_size]["fp"] += 1

    # Count FN by size and class
    for idx, (size, cls_id) in enumerate(zip(gt_sizes, gt_classes, strict=True)):
        if idx not in matched_gt:
            if cls_id in stats:
                stats[cls_id][size]["fn"] += 1

    return stats


# --- Utility: Compute AP for a single class ---
def compute_ap(recalls, precisions):
    """Compute Average Precision using 101-point interpolation (COCO style)."""
    recalls = np.array(recalls, dtype=np.float32)
    precisions = np.array(precisions, dtype=np.float32)

    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]

    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])

    # Monotonically decreasing precision (vectorised)
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    # 101-point interpolation (vectorised)
    thresholds = np.linspace(0, 1, 101, dtype=np.float32)
    # For each threshold, max precision where recall >= threshold
    # Equivalent to: for each t, precisions[recalls >= t].max() if any
    indices = np.searchsorted(recalls, thresholds, side="left")
    indices = np.clip(indices, 0, len(precisions) - 1)
    ap = float(np.mean(precisions[indices]))
    return ap


# --- Utility: Compute mAP from all predictions ---
def compute_map_per_class(all_predictions, all_ground_truths, iou_thresholds=[0.5]):
    """
    Compute AP per class at given IoU thresholds.
    Returns: {iou_thr: {class_id: ap}}
    """
    # Group ground truths by image and class
    gt_by_img_cls = defaultdict(list)
    for img_id, cls_id, box in all_ground_truths:
        gt_by_img_cls[(img_id, cls_id)].append(box)

    # Count total GT per class
    n_gt_per_class = defaultdict(int)
    for _img_id, cls_id, _box in all_ground_truths:
        n_gt_per_class[cls_id] += 1

    # Sort predictions by confidence (descending)
    all_predictions = sorted(all_predictions, key=lambda x: x[2], reverse=True)

    # Get unique classes
    all_classes = set(cls_id for _, cls_id, _, _ in all_predictions)
    all_classes.update(cls_id for _, cls_id, _ in all_ground_truths)

    aps_per_iou = {}

    for iou_thr in iou_thresholds:
        aps = {}

        for cls_id in sorted(all_classes):
            if n_gt_per_class[cls_id] == 0:
                aps[cls_id] = 0
                continue

            # Get predictions for this class
            cls_preds = [
                (img_id, conf, box)
                for img_id, c, conf, box in all_predictions
                if c == cls_id
            ]

            if not cls_preds:
                aps[cls_id] = 0
                continue

            # Track which GT boxes have been matched
            gt_matched = defaultdict(set)

            tps = []
            fps = []

            # Pre-convert GT lists to numpy arrays for vectorised IoU
            gt_arrays = {
                k: np.array(v, dtype=np.float32)
                for k, v in gt_by_img_cls.items()
                if k[1] == cls_id
            }

            for img_id, _conf, pred_box in cls_preds:
                gt_arr = gt_arrays.get((img_id, cls_id))
                best_iou = 0
                best_gt_idx = -1

                if gt_arr is not None and len(gt_arr) > 0:
                    matched_set = gt_matched[(img_id, cls_id)]
                    ious = _iou_one_vs_many(pred_box, gt_arr)
                    for i in matched_set:
                        ious[i] = 0
                    best_gt_idx = int(np.argmax(ious))
                    best_iou = float(ious[best_gt_idx])

                if best_iou >= iou_thr:
                    tps.append(1)
                    fps.append(0)
                    gt_matched[(img_id, cls_id)].add(best_gt_idx)
                else:
                    tps.append(0)
                    fps.append(1)

            # Cumulative sum
            tps_cumsum = np.cumsum(tps)
            fps_cumsum = np.cumsum(fps)

            recalls = tps_cumsum / n_gt_per_class[cls_id]
            precisions = tps_cumsum / (tps_cumsum + fps_cumsum)

            ap = compute_ap(recalls, precisions)
            aps[cls_id] = ap

        aps_per_iou[iou_thr] = aps

    return aps_per_iou


# --- Utility: Compute mAP50-95 ---
def compute_map_50_95_per_class(all_predictions, all_ground_truths):
    """Compute mAP at IoU thresholds from 0.5 to 0.95 with step 0.05, per class."""
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps_per_iou = compute_map_per_class(
        all_predictions, all_ground_truths, iou_thresholds
    )

    # Get all classes
    all_classes = set()
    for _iou_thr, aps in aps_per_iou.items():
        all_classes.update(aps.keys())

    # Compute per-class mAP50 and mAP50-95
    per_class_metrics = {}
    for cls_id in all_classes:
        map50 = aps_per_iou.get(0.5, {}).get(cls_id, 0)
        map50_95 = np.mean([aps.get(cls_id, 0) for aps in aps_per_iou.values()])
        per_class_metrics[cls_id] = {"mAP50": map50, "mAP50-95": map50_95}

    # Overall mAP
    overall_map50 = (
        np.mean([m["mAP50"] for m in per_class_metrics.values()])
        if per_class_metrics
        else 0
    )
    overall_map50_95 = (
        np.mean([m["mAP50-95"] for m in per_class_metrics.values()])
        if per_class_metrics
        else 0
    )

    return overall_map50, overall_map50_95, per_class_metrics, aps_per_iou


# --- Utility: Optimal confidence calibration per class and size ---
def find_optimal_conf_per_class_size(
    all_predictions,
    all_ground_truths,
    num_classes,
    class_names,
    iou_thr=0.5,
):
    """
    For each (class, size_category) combination independently, find the confidence
    threshold that maximises F1.

    Each bucket runs its own isolated IoU matching against only GT boxes in that
    bucket, so results are independent and correct per bucket.

    Fast approach per bucket:
      1. Match predictions against only this bucket's GT boxes (conf-descending),
         labelling each prediction as TP or FP for this bucket.
      2. Sweep thresholds incrementally (remove low-conf preds one by one),
         updating TP/FP/FN counts in O(1) per step.

    Returns:
        dict: {(cls_id, size_name): {"conf", "precision", "recall", "f1", "gt_count"}}
    """
    # Build GT index per (img_id, cls_id, size_name) -> numpy array of boxes
    # so each bucket has its own independent GT set
    gt_by_bucket = defaultdict(list)  # (img_id, cls_id, size_name) -> [box, ...]
    for img_id, cls_id, box in all_ground_truths:
        size_name = get_size_category(box)
        gt_by_bucket[(img_id, cls_id, size_name)].append(box)

    # Pre-convert to numpy arrays
    gt_arrays_bucket = {
        k: np.array(v, dtype=np.float32) for k, v in gt_by_bucket.items()
    }

    # Total GT count per (cls_id, size_name)
    gt_count_per_cls_size = defaultdict(int)
    for _img_id, cls_id, box in all_ground_truths:
        gt_count_per_cls_size[(cls_id, get_size_category(box))] += 1

    # Group predictions by class, sorted by conf descending
    preds_by_cls = defaultdict(list)
    for img_id, cls_id, conf, box in all_predictions:
        preds_by_cls[cls_id].append((conf, img_id, box))
    for cls_id in preds_by_cls:
        preds_by_cls[cls_id].sort(key=lambda x: x[0], reverse=True)

    def _score(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec + 1e-6) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    results = {}

    for cls_id in range(num_classes):
        preds = preds_by_cls.get(cls_id, [])

        for size_name, _, _ in SIZE_CATEGORIES:
            total_gt = gt_count_per_cls_size[(cls_id, size_name)]

            # --- Phase 1: IoU matching restricted to this bucket's GT boxes ---
            # Each prediction is labelled: is_tp (matched bucket GT) or fp (pred in bucket or not)
            # Only predictions whose pred_size == size_name or that match a GT in this bucket matter.
            gt_matched = defaultdict(
                set
            )  # img_id -> set of matched gt indices within this bucket
            annotated = []  # (conf_val, is_tp_for_bucket, pred_in_bucket)
            # is_tp_for_bucket: matched a GT in this size bucket
            # pred_in_bucket: prediction box itself is in this size bucket (for FP counting)

            for conf_val, img_id, box in preds:
                pred_size = get_size_category(box)
                gt_arr = gt_arrays_bucket.get((img_id, cls_id, size_name))
                best_iou = 0.0
                best_gt_idx = -1

                if gt_arr is not None and len(gt_arr) > 0:
                    matched_set = gt_matched[img_id]
                    ious = _iou_one_vs_many(box, gt_arr)
                    for i in matched_set:
                        ious[i] = 0
                    best_gt_idx = int(np.argmax(ious))
                    best_iou = float(ious[best_gt_idx])

                if best_iou >= iou_thr:
                    gt_matched[img_id].add(best_gt_idx)
                    annotated.append((conf_val, True, pred_size == size_name))
                else:
                    annotated.append((conf_val, False, pred_size == size_name))

            # --- Phase 2: incremental threshold sweep ---
            # Start fully active (threshold = 0), remove predictions as threshold rises
            tp = sum(1 for _, is_tp, _ in annotated if is_tp)
            fp = sum(1 for _, is_tp, in_bucket in annotated if not is_tp and in_bucket)
            fn = total_gt - tp

            # Evaluate at threshold=0 as baseline
            prec, rec, f1 = _score(tp, fp, fn)
            best_f1 = f1
            best_entry = {
                "conf": 0.0,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "gt_count": total_gt,
            }

            if not annotated:
                results[(cls_id, size_name)] = best_entry
                continue

            # Group by conf for fast removal
            by_conf = defaultdict(list)
            for conf_val, is_tp, in_bucket in annotated:
                by_conf[conf_val].append((is_tp, in_bucket))

            # Sweep thresholds from lowest to highest unique conf value.
            # At each step we raise the threshold to the next conf level,
            # dropping all predictions below it, then score.
            conf_vals_asc = sorted(by_conf.keys())  # ascending

            for thr in conf_vals_asc:
                # Drop predictions at exactly this conf level (they fall below thr+epsilon)
                for is_tp, in_bucket in by_conf[thr]:
                    if is_tp:
                        tp -= 1
                        fn += 1
                    elif in_bucket:
                        fp -= 1

                # The new threshold is just above `thr`, so active = conf > thr.
                # Use the next conf value as the reported threshold (or thr+eps).
                prec, rec, f1 = _score(tp, fp, fn)
                if f1 > best_f1:
                    best_f1 = f1
                    best_entry = {
                        "conf": thr,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "gt_count": total_gt,
                    }

            results[(cls_id, size_name)] = best_entry

    # Print table
    console.print()
    console.rule("[bold blue]Optimal Confidence Calibration by Class and Size[/bold blue]")
    console.print(f"[cyan](maximising F1 at IoU={iou_thr})[/cyan]")

    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("Class", style="cyan", min_width=12)
    table.add_column("Size", min_width=16)
    table.add_column("GT", justify="right", min_width=6)
    table.add_column("Opt.Conf", justify="right", min_width=10)
    table.add_column("Precision", justify="right", min_width=10)
    table.add_column("Recall", justify="right", min_width=10)
    table.add_column("F1", justify="right", min_width=10)

    for cls_id, name in enumerate(class_names):
        first = True
        for size_name, low, high in SIZE_CATEGORIES:
            r = results[(cls_id, size_name)]
            if r["gt_count"] == 0:
                continue
            high_str = str(int(high)) if high != float("inf") else "inf"
            size_label = f"{size_name}({int(low)}-{high_str}px)"
            cls_label = name if first else ""
            first = False
            table.add_row(
                cls_label,
                size_label,
                str(r["gt_count"]),
                f"{r['conf']:.2f}",
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1']:.4f}",
            )

    console.print(table)

    return results


def eval_frame(gt_boxes, pred_boxes, iou_thr: float = 0.5) -> tuple[int, int, int]:
    """Return (tp, fp, fn) for one frame."""
    if not gt_boxes:
        return 0, len(pred_boxes), 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)

    gt_arr = np.array(gt_boxes, dtype=float)
    matched_gt = set()
    tp = fp = 0

    for pb in pred_boxes:
        ious = _iou_one_vs_many(pb, gt_arr)
        best = int(np.argmax(ious))
        if ious[best] >= iou_thr and best not in matched_gt:
            tp += 1
            matched_gt.add(best)
        else:
            fp += 1

    fn = len(gt_boxes) - tp
    return tp, fp, fn
