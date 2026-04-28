from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .constants import PROJECT_ROOT, TOOL_CLASSES_ZH
from .retinex import msrcr
from .soft_nms import class_wise_soft_nms


def default_weights_path() -> Path:
    candidates = [
        PROJECT_ROOT / "runs" / "detect" / "tool15_user_practical_v1" / "weights" / "best.pt",
        PROJECT_ROOT / "runs" / "detect" / "tool15_user_practical_v1" / "weights" / "last.pt",
        PROJECT_ROOT / "runs" / "detect" / "tool15_user_practical_smoke" / "weights" / "best.pt",
        PROJECT_ROOT / "runs" / "detect" / "tool15_real_bootstrap_v1" / "weights" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("没有找到可用的训练权重，请先完成训练。")


def default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


@lru_cache(maxsize=4)
def load_model(weights: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("缺少 ultralytics，请先安装项目依赖。") from exc
    return YOLO(weights)


def color_for_class(class_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(class_id + 2026)
    bgr = rng.integers(60, 255, size=3)
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    names: dict[int, str],
) -> np.ndarray:
    canvas = image.copy()
    for box, score, class_id in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int).tolist()
        color = color_for_class(int(class_id))
        class_name = names.get(int(class_id), str(class_id))
        text = f"{class_name} {score:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, text, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return canvas


def run_inference(
    image_bgr: np.ndarray,
    weights: Path,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.5,
    device: str | None = None,
    soft_nms: bool = False,
    retinex: bool = False,
) -> dict[str, object]:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("没有收到有效图片。")

    weights = weights.expanduser().resolve()
    if not weights.exists():
        raise FileNotFoundError(f"模型权重不存在: {weights}")

    model_input = msrcr(image_bgr) if retinex else image_bgr.copy()
    model = load_model(str(weights))
    result = model.predict(
        source=model_input,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device or default_device(),
        save=False,
        verbose=False,
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return {
            "image": image_bgr,
            "detections": [],
            "summary": "未检测到目标。",
        }

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    labels = result.boxes.cls.cpu().numpy().astype(np.int32)
    names = result.names

    if soft_nms:
        keep_indices, updated_scores = class_wise_soft_nms(
            boxes=boxes,
            scores=scores,
            labels=labels,
            sigma=0.5,
            iou_threshold=iou,
            score_threshold=conf,
            method="gaussian",
        )
        boxes = boxes[keep_indices]
        labels = labels[keep_indices]
        scores = updated_scores

    drawn = draw_detections(image=image_bgr, boxes=boxes, scores=scores, labels=labels, names=names)
    detections: list[dict[str, object]] = []
    for box, score, class_id in zip(boxes, scores, labels):
        class_name = names.get(int(class_id), str(class_id))
        detections.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "class_name_zh": TOOL_CLASSES_ZH.get(class_name, class_name),
                "confidence": float(score),
                "bbox_xyxy": [round(float(value), 1) for value in box.tolist()],
            }
        )

    top_detection = max(detections, key=lambda item: item["confidence"])
    summary = (
        f"检测到 {len(detections)} 个目标，"
        f"最高置信度类别为 {top_detection['class_name_zh']} "
        f"({top_detection['confidence']:.3f})。"
    )
    return {
        "image": drawn,
        "detections": detections,
        "summary": summary,
    }
