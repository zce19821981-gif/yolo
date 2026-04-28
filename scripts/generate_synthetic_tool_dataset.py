from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_yolo.constants import TOOL_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic starter dataset for 15 tool classes.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "raw_tool15_synth",
        help="Output directory for generated images and labels.",
    )
    parser.add_argument("--per-class", type=int, default=12, help="Synthetic image count per class.")
    parser.add_argument("--image-size", type=int, default=640, help="Square image size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def make_background(size: int, rng: random.Random) -> np.ndarray:
    bg = np.zeros((size, size, 3), dtype=np.uint8)
    base = rng.randint(170, 235)
    bg[:] = (base, base, base)

    for _ in range(rng.randint(8, 16)):
        color = tuple(int(np.clip(base + rng.randint(-35, 20), 0, 255)) for _ in range(3))
        x1 = rng.randint(0, size - 1)
        y1 = rng.randint(0, size - 1)
        x2 = rng.randint(0, size - 1)
        y2 = rng.randint(0, size - 1)
        thickness = rng.randint(1, 3)
        cv2.line(bg, (x1, y1), (x2, y2), color, thickness)

    noise = np.random.normal(0, 8, bg.shape).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return bg


def rotate_points(points: np.ndarray, center: tuple[float, float], angle_deg: float) -> np.ndarray:
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    shifted = points - np.array(center, dtype=np.float32)
    rotated = shifted @ rotation.T
    return rotated + np.array(center, dtype=np.float32)


def draw_polygon_tool(
    canvas: np.ndarray,
    center: tuple[int, int],
    size: tuple[int, int],
    angle: float,
    kind: str,
    color: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    cx, cy = center
    w, h = size

    if kind == "turning_tool":
        points = np.array(
            [
                [cx - w * 0.45, cy + h * 0.10],
                [cx + w * 0.20, cy + h * 0.10],
                [cx + w * 0.45, cy - h * 0.20],
                [cx + w * 0.10, cy - h * 0.40],
                [cx - w * 0.45, cy - h * 0.40],
            ],
            dtype=np.float32,
        )
    elif kind in {"milling_cutter", "end_mill", "reamer", "tap", "twist_drill"}:
        step = 7 if kind in {"tap", "reamer"} else 6
        points = []
        for i in range(step):
            px = cx - w * 0.45 + (w * i / max(step - 1, 1))
            offset = h * 0.38 if i % 2 == 0 else h * 0.15
            points.append([px, cy - offset])
        for i in reversed(range(step)):
            px = cx - w * 0.45 + (w * i / max(step - 1, 1))
            offset = h * 0.15 if i % 2 == 0 else h * 0.38
            points.append([px, cy + offset])
        points = np.array(points, dtype=np.float32)
    elif kind in {"broach", "slotting_tool", "planer_tool", "boring_tool"}:
        points = np.array(
            [
                [cx - w * 0.48, cy - h * 0.18],
                [cx + w * 0.28, cy - h * 0.18],
                [cx + w * 0.48, cy - h * 0.34],
                [cx + w * 0.48, cy + h * 0.34],
                [cx + w * 0.28, cy + h * 0.18],
                [cx - w * 0.48, cy + h * 0.18],
            ],
            dtype=np.float32,
        )
    elif kind == "hob":
        points = []
        teeth = 10
        for i in range(teeth * 2):
            angle_i = (2 * math.pi * i) / (teeth * 2)
            radius = min(w, h) * (0.42 if i % 2 == 0 else 0.30)
            points.append([cx + radius * math.cos(angle_i), cy + radius * math.sin(angle_i)])
        points = np.array(points, dtype=np.float32)
    elif kind == "alloy_saw_blade":
        points = []
        teeth = 12
        for i in range(teeth * 2):
            angle_i = (2 * math.pi * i) / (teeth * 2)
            radius = min(w, h) * (0.46 if i % 2 == 0 else 0.34)
            points.append([cx + radius * math.cos(angle_i), cy + radius * math.sin(angle_i)])
        points = np.array(points, dtype=np.float32)
    elif kind in {"countersink", "counterbore_drill"}:
        points = np.array(
            [
                [cx - w * 0.15, cy - h * 0.45],
                [cx + w * 0.15, cy - h * 0.45],
                [cx + w * 0.35, cy + h * 0.15],
                [cx + w * 0.12, cy + h * 0.45],
                [cx - w * 0.12, cy + h * 0.45],
                [cx - w * 0.35, cy + h * 0.15],
            ],
            dtype=np.float32,
        )
    elif kind == "drill_bit":
        points = np.array(
            [
                [cx - w * 0.20, cy - h * 0.46],
                [cx + w * 0.20, cy - h * 0.46],
                [cx + w * 0.30, cy + h * 0.15],
                [cx, cy + h * 0.46],
                [cx - w * 0.30, cy + h * 0.15],
            ],
            dtype=np.float32,
        )
    else:
        points = np.array(
            [
                [cx - w * 0.45, cy - h * 0.25],
                [cx + w * 0.45, cy - h * 0.25],
                [cx + w * 0.45, cy + h * 0.25],
                [cx - w * 0.45, cy + h * 0.25],
            ],
            dtype=np.float32,
        )

    points = rotate_points(points, center=(cx, cy), angle_deg=angle).astype(np.int32)
    cv2.fillPoly(canvas, [points], color)
    cv2.polylines(canvas, [points], isClosed=True, color=(30, 30, 30), thickness=2)

    x, y, w_box, h_box = cv2.boundingRect(points)
    return x, y, w_box, h_box


def draw_tool_image(class_name: str, size: int, rng: random.Random) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    canvas = make_background(size=size, rng=rng)
    center = (
        rng.randint(int(size * 0.3), int(size * 0.7)),
        rng.randint(int(size * 0.3), int(size * 0.7)),
    )
    box_w = rng.randint(int(size * 0.24), int(size * 0.52))
    box_h = rng.randint(int(size * 0.12), int(size * 0.36))
    if class_name in {"hob", "alloy_saw_blade", "drill_bit", "countersink", "counterbore_drill"}:
        box_h = rng.randint(int(size * 0.22), int(size * 0.42))
    angle = rng.uniform(-55.0, 55.0)
    color = (
        rng.randint(60, 130),
        rng.randint(90, 170),
        rng.randint(120, 220),
    )
    x, y, w, h = draw_polygon_tool(canvas, center, (box_w, box_h), angle, class_name, color)

    for _ in range(rng.randint(2, 5)):
        ox = rng.randint(max(0, x - 40), min(size - 1, x + w + 40))
        oy = rng.randint(max(0, y - 40), min(size - 1, y + h + 40))
        radius = rng.randint(8, 24)
        shade = rng.randint(130, 210)
        cv2.circle(canvas, (ox, oy), radius, (shade, shade, shade), -1)

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(size, x + w)
    y2 = min(size, y + h)
    x_center = ((x1 + x2) / 2.0) / size
    y_center = ((y1 + y2) / 2.0) / size
    norm_w = (x2 - x1) / size
    norm_h = (y2 - y1) / size
    return canvas, (x_center, y_center, norm_w, norm_h)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    images_dir = args.output_root / "images"
    labels_dir = args.output_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for class_id, class_name in enumerate(TOOL_CLASSES):
        for idx in range(args.per_class):
            image, box = draw_tool_image(class_name=class_name, size=args.image_size, rng=rng)
            image_path = images_dir / f"{class_name}_{idx:03d}.jpg"
            label_path = labels_dir / f"{class_name}_{idx:03d}.txt"
            cv2.imwrite(str(image_path), image)
            x_center, y_center, width, height = box
            label_path.write_text(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n",
                encoding="utf-8",
            )

    print(f"synthetic dataset saved to {args.output_root}")


if __name__ == "__main__":
    main()
