from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_yolo.constants import TOOL_CLASSES


MANUAL_BOXES = {
    "turning_tool": (0.48, 0.53, 0.90, 0.82),
    "milling_cutter": (0.60, 0.38, 0.72, 0.56),
    "drill_bit": (0.50, 0.52, 0.90, 0.62),
    "reamer": (0.50, 0.50, 0.86, 0.20),
    "boring_tool": (0.34, 0.48, 0.58, 0.42),
    "tap": (0.53, 0.54, 0.84, 0.28),
    "broach": (0.50, 0.48, 0.70, 0.62),
    "hob": (0.52, 0.50, 0.58, 0.90),
    "slotting_tool": (0.50, 0.50, 0.94, 0.42),
    "planer_tool": (0.50, 0.56, 0.92, 0.42),
    "twist_drill": (0.50, 0.51, 0.92, 0.64),
    "end_mill": (0.52, 0.45, 0.84, 0.52),
    "alloy_saw_blade": (0.49, 0.46, 0.94, 0.86),
    "countersink": (0.50, 0.51, 0.58, 0.54),
    "counterbore_drill": (0.52, 0.50, 0.86, 0.44),
}


def xywhn_to_xyxy(box: tuple[float, float, float, float], w: int, h: int) -> tuple[int, int, int, int]:
    x_center, y_center, bw, bh = box
    x1 = int((x_center - bw / 2.0) * w)
    y1 = int((y_center - bh / 2.0) * h)
    x2 = int((x_center + bw / 2.0) * w)
    y2 = int((y_center + bh / 2.0) * h)
    return max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)


def main() -> None:
    source_dir = PROJECT_ROOT / "photos" / "class_samples"
    output_root = PROJECT_ROOT / "datasets" / "raw_tool15_real"
    images_dir = output_root / "images"
    labels_dir = output_root / "labels"
    preview_dir = output_root / "preview"

    if output_root.exists():
        shutil.rmtree(output_root)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    for class_id, class_name in enumerate(TOOL_CLASSES):
        src = source_dir / f"{class_name}.jpg"
        dst = images_dir / src.name
        shutil.copy2(src, dst)

        box = MANUAL_BOXES[class_name]
        label_path = labels_dir / f"{class_name}.txt"
        label_path.write_text(
            f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n",
            encoding="utf-8",
        )

        image = cv2.imread(str(dst))
        if image is None:
            continue
        h, w = image.shape[:2]
        x1, y1, x2, y2 = xywhn_to_xyxy(box, w, h)
        preview = image.copy()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 80), 3)
        cv2.putText(preview, class_name, (max(10, x1), max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 80), 2, cv2.LINE_AA)
        cv2.imwrite(str(preview_dir / f"{class_name}.jpg"), preview)

    print(f"real-only dataset scaffolded at {output_root}")


if __name__ == "__main__":
    main()
