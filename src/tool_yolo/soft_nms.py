from __future__ import annotations

import numpy as np


def bbox_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    intersection = inter_w * inter_h

    box_area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection + 1e-6
    return intersection / union


def soft_nms_single_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    sigma: float = 0.5,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.001,
    method: str = "gaussian",
) -> tuple[np.ndarray, np.ndarray]:
    boxes = boxes.astype(np.float32).copy()
    scores = scores.astype(np.float32).copy()
    indices = np.arange(len(boxes))

    kept_indices: list[int] = []
    kept_scores: list[float] = []

    while len(indices) > 0:
        best_pos = int(np.argmax(scores))
        best_index = int(indices[best_pos])
        best_box = boxes[best_pos].copy()
        best_score = float(scores[best_pos])

        kept_indices.append(best_index)
        kept_scores.append(best_score)

        boxes = np.delete(boxes, best_pos, axis=0)
        scores = np.delete(scores, best_pos, axis=0)
        indices = np.delete(indices, best_pos, axis=0)

        if len(indices) == 0:
            break

        ious = bbox_iou(best_box, boxes)
        if method == "linear":
            decay = np.where(ious > iou_threshold, 1.0 - ious, 1.0)
        else:
            decay = np.exp(-(ious * ious) / max(sigma, 1e-6))

        scores = scores * decay
        keep_mask = scores >= score_threshold
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        indices = indices[keep_mask]

    return np.asarray(kept_indices, dtype=np.int32), np.asarray(kept_scores, dtype=np.float32)


def class_wise_soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    sigma: float = 0.5,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.001,
    method: str = "gaussian",
) -> tuple[np.ndarray, np.ndarray]:
    selected_indices: list[int] = []
    selected_scores: list[float] = []

    for class_id in np.unique(labels):
        cls_mask = labels == class_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_original_indices = np.where(cls_mask)[0]

        keep_local, keep_scores = soft_nms_single_class(
            boxes=cls_boxes,
            scores=cls_scores,
            sigma=sigma,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            method=method,
        )
        selected_indices.extend(cls_original_indices[keep_local].tolist())
        selected_scores.extend(keep_scores.tolist())

    if not selected_indices:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32)

    order = np.argsort(-np.asarray(selected_scores))
    return np.asarray(selected_indices, dtype=np.int32)[order], np.asarray(selected_scores, dtype=np.float32)[order]

