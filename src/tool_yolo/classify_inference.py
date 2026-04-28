from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .constants import PROJECT_ROOT, TOOL_CLASSES_ZH
from .inference import default_device


MODEL_CANDIDATES = [
    PROJECT_ROOT / 'models' / 'tool15_cls_user_v1_best.pt',
    PROJECT_ROOT / 'runs' / 'classify' / 'tool15_cls_user_v1' / 'weights' / 'best.pt',
    PROJECT_ROOT / 'runs' / 'classify' / 'tool15_cls_user_v1' / 'weights' / 'last.pt',
]


def default_classify_weights_path() -> Path:
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError('没有找到可用的分类模型权重，请先完成分类训练。')


@lru_cache(maxsize=4)
def load_classify_model(weights: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('缺少 ultralytics，请先安装项目依赖。') from exc
    return YOLO(weights)


def annotate_classification(image: np.ndarray, summary: str) -> np.ndarray:
    canvas = image.copy()
    h, w = canvas.shape[:2]
    bar_h = max(64, h // 10)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (24, 28, 40), -1)
    canvas = cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0)
    cv2.putText(canvas, summary, (18, min(bar_h - 18, 42)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def run_classification(
    image_bgr: np.ndarray,
    weights: Path,
    imgsz: int = 224,
    device: str | None = None,
) -> dict[str, object]:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError('没有收到有效图片。')

    weights = weights.expanduser().resolve()
    if not weights.exists():
        raise FileNotFoundError(f'模型权重不存在: {weights}')

    model = load_classify_model(str(weights))
    result = model.predict(
        source=image_bgr,
        imgsz=imgsz,
        device=device or default_device(),
        save=False,
        verbose=False,
    )[0]

    probs = getattr(result, 'probs', None)
    if probs is None:
        raise RuntimeError('分类结果为空，无法解析预测概率。')

    names = result.names
    top1_id = int(probs.top1)
    top1_name = names[top1_id]
    top1_conf = float(probs.top1conf.item())

    topk: list[dict[str, object]] = []
    for class_id, conf in zip(probs.top5, probs.top5conf):
        class_id = int(class_id)
        class_name = names[class_id]
        topk.append(
            {
                'class_id': class_id,
                'class_name': class_name,
                'class_name_zh': TOOL_CLASSES_ZH.get(class_name, class_name),
                'confidence': float(conf.item()),
            }
        )

    summary = f'预测结果: {TOOL_CLASSES_ZH.get(top1_name, top1_name)} ({top1_conf:.3f})'
    image_summary = f'Prediction: {top1_name} ({top1_conf:.3f})'
    annotated = annotate_classification(image_bgr, image_summary)
    return {
        'image': annotated,
        'summary': summary,
        'top1': {
            'class_id': top1_id,
            'class_name': top1_name,
            'class_name_zh': TOOL_CLASSES_ZH.get(top1_name, top1_name),
            'confidence': top1_conf,
        },
        'topk': topk,
    }
