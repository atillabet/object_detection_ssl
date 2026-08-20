import cv2
import numpy as np


# --- Utility: Slice image into patches ---
def slice_image(image, slice_wh, overlap_wh):
    """Slice image into overlapping patches."""
    h, w = image.shape[:2]
    slice_w, slice_h = slice_wh
    overlap_w, overlap_h = overlap_wh
    step_w = slice_w - overlap_w
    step_h = slice_h - overlap_h

    slices = []
    for y in range(0, h, step_h):
        for x in range(0, w, step_w):
            x2 = min(x + slice_w, w)
            y2 = min(y + slice_h, h)
            patch = image[y:y2, x:x2]

            # Pad if needed
            if patch.shape[0] < slice_h or patch.shape[1] < slice_w:
                padded = np.zeros((slice_h, slice_w, 3), dtype=np.uint8)
                padded[: patch.shape[0], : patch.shape[1]] = patch
                patch = padded

            slices.append((patch, x, y))

    return slices


# --- Utility: Batched SAHI inference ---
def sahi_inference_batched(
    model,
    image,
    slice_wh,
    overlap_wh,
    batch_size,
    model_input_size=None,
    conf=0.05,
    iou_threshold=0.5,
    large_conf=0.25,
    size_threshold=16,
    half=True,
    device=None,
):
    """Run SAHI inference with batched YOLO predictions.

    Uses adaptive confidence: detections with max(w, h) >= size_threshold require
    large_conf, while smaller objects only need conf.
    """
    slices = slice_image(image, slice_wh, overlap_wh)

    all_boxes = []
    all_scores = []
    all_classes = []

    # Calculate scale factor if resizing tiles
    if model_input_size and model_input_size != slice_wh[0]:
        scale_factor = slice_wh[0] / model_input_size
        inference_size = model_input_size
    else:
        scale_factor = 1.0
        inference_size = slice_wh[0]

    # Process in batches
    for i in range(0, len(slices), batch_size):
        batch_slices = slices[i : i + batch_size]
        batch_offsets = [(s[1], s[2]) for s in batch_slices]

        # Resize tiles to model input size if needed
        if model_input_size and model_input_size != slice_wh[0]:
            batch_images = [
                cv2.resize(s[0], (model_input_size, model_input_size))
                for s in batch_slices
            ]
        else:
            batch_images = [s[0] for s in batch_slices]

        # Batched inference at low conf to catch small objects; filter large ones below
        results = model(
            batch_images,
            conf=conf,
            imgsz=inference_size,
            verbose=False,
            half=half,
            device=device,
        )

        # Collect detections with offsets (vectorised per tile)
        for result, (x_offset, y_offset) in zip(
            results, batch_offsets, strict=True
        ):
            if result.boxes is None or len(result.boxes) == 0:
                continue

            xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
            scores = result.boxes.conf.cpu().numpy()  # (N,)
            classes = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

            # Scale + offset in one numpy op
            xyxy = xyxy * scale_factor
            xyxy[:, 0] += x_offset
            xyxy[:, 2] += x_offset
            xyxy[:, 1] += y_offset
            xyxy[:, 3] += y_offset

            bw = xyxy[:, 2] - xyxy[:, 0]
            bh = xyxy[:, 3] - xyxy[:, 1]
            max_dim = np.maximum(bw, bh)

            # Adaptive confidence filter (vectorised)
            min_confs = np.where(max_dim >= size_threshold, large_conf, conf)
            keep = scores >= min_confs

            all_boxes.extend(xyxy[keep].tolist())
            all_scores.extend(scores[keep].tolist())
            all_classes.extend(classes[keep].tolist())

    if len(all_boxes) == 0:
        return [], [], []

    # NMS to merge overlapping detections
    boxes_for_nms = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in all_boxes]
    indices = cv2.dnn.NMSBoxes(
        boxes_for_nms, all_scores, score_threshold=0.0, nms_threshold=iou_threshold
    )

    if len(indices) == 0:
        return [], [], []

    indices = indices.flatten()
    final_boxes = [all_boxes[i] for i in indices]
    final_scores = [all_scores[i] for i in indices]
    final_classes = [all_classes[i] for i in indices]

    return final_boxes, final_scores, final_classes
