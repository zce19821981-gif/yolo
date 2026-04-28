from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_yolo.constants import TOOL_CLASSES_ZH
from tool_yolo.retinex import msrcr
from tool_yolo.soft_nms import class_wise_soft_nms

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for tool recognition.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to best.pt")
    parser.add_argument("--source", type=Path, required=True, help="Image file or image directory.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "predict")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--soft-nms", action="store_true", help="Apply Soft-NMS after YOLO output.")
    parser.add_argument("--retinex", action="store_true", help="Apply Retinex before inference.")
    parser.add_argument("--save-txt", action="store_true", help="Save predicted labels in YOLO format.")
    return parser.parse_args()


def list_source_images(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(f"无法识别推理源: {source}")


def color_for_class(class_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(class_id + 2026)
    bgr = rng.integers(60, 255, size=3)
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def xyxy_to_yolo(box: np.ndarray, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box.tolist()
    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return x_center, y_center, width, height


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
        display_name = TOOL_CLASSES_ZH.get(class_name, class_name)
        text = f"{display_name} {score:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, text, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("缺少 ultralytics，请先执行 `pip install -r requirements.txt`。") from exc

    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image_paths = list_source_images(args.source)
    if not image_paths:
        raise SystemExit("没有找到可用于推理的图片。")

    temp_source_dir = args.output / "temp_source"
    predict_paths: list[Path] = []
    original_map: dict[Path, Path] = {}

    if args.retinex:
        temp_source_dir.mkdir(parents=True, exist_ok=True)
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            enhanced = msrcr(image)
            processed_path = temp_source_dir / image_path.name
            cv2.imwrite(str(processed_path), enhanced)
            predict_paths.append(processed_path)
            original_map[processed_path] = image_path
    else:
        predict_paths = image_paths
        original_map = {path: path for path in image_paths}

    model = YOLO(str(args.weights))
    results = model.predict(
        source=[str(path) for path in predict_paths],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=False,
        verbose=False,
    )

    label_dir = args.output / "labels"
    if args.save_txt:
        label_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        predicted_path = Path(result.path)
        original_path = original_map.get(predicted_path, predicted_path)
        original_image = cv2.imread(str(original_path))
        if original_image is None:
            continue

        if result.boxes is None or len(result.boxes) == 0:
            output_image_path = args.output / original_path.name
            cv2.imwrite(str(output_image_path), original_image)
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy().astype(np.int32)

        if args.soft_nms:
            keep_indices, updated_scores = class_wise_soft_nms(
                boxes=boxes,
                scores=scores,
                labels=labels,
                sigma=0.5,
                iou_threshold=args.iou,
                score_threshold=args.conf,
                method="gaussian",
            )
            boxes = boxes[keep_indices]
            labels = labels[keep_indices]
            scores = updated_scores

        drawn = draw_detections(original_image, boxes, scores, labels, result.names)
        output_image_path = args.output / original_path.name
        cv2.imwrite(str(output_image_path), drawn)

        if args.save_txt:
            h, w = original_image.shape[:2]
            txt_path = label_dir / f"{original_path.stem}.txt"
            with txt_path.open("w", encoding="utf-8") as file:
                for box, score, class_id in zip(boxes, scores, labels):
                    x_center, y_center, width, height = xyxy_to_yolo(box, image_width=w, image_height=h)
                    file.write(
                        f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}\n"
                    )

    print(f"预测结果已保存到: {args.output}")


if __name__ == "__main__":
    main()

