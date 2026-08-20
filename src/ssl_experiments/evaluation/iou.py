import numpy as np


def iou_one_vs_many(box, gt_array):
    """Vectorised IoU of one [x1,y1,x2,y2] box against an (N,4) array. Returns (N,) array."""
    xA = np.maximum(box[0], gt_array[:, 0])
    yA = np.maximum(box[1], gt_array[:, 1])
    xB = np.minimum(box[2], gt_array[:, 2])
    yB = np.minimum(box[3], gt_array[:, 3])
    inter = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
    aA = (box[2] - box[0]) * (box[3] - box[1])
    aB = (gt_array[:, 2] - gt_array[:, 0]) * (gt_array[:, 3] - gt_array[:, 1])
    return inter / (aA + aB - inter + 1e-6)
