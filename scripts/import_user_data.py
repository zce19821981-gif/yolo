from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_yolo.constants import TOOL_CLASSES, TOOL_CLASSES_ZH

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}


def detect_bbox(image: Image.Image) -> tuple[float, float, float, float]:
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]

    border = np.concatenate(
        [
            rgb[0, :, :],
            rgb[-1, :, :],
            rgb[:, 0, :],
            rgb[:, -1, :],
        ],
        axis=0,
    )
    background = np.median(border, axis=0)
    distance = np.linalg.norm(rgb.astype(np.float32) - background.astype(np.float32), axis=2)
    gray = rgb.mean(axis=2)
    mask = (distance > 25.0) & (gray > 8) & (gray < 248)

    ys, xs = np.where(mask)
    if len(xs) < max(200, int(width * height * 0.01)):
        return 0.5, 0.5, 0.92, 0.92

    x1 = max(0, int(xs.min()) - int(width * 0.03))
    y1 = max(0, int(ys.min()) - int(height * 0.03))
    x2 = min(width - 1, int(xs.max()) + int(width * 0.03))
    y2 = min(height - 1, int(ys.max()) + int(height * 0.03))

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    return (
        (x1 + box_w / 2.0) / width,
        (y1 + box_h / 2.0) / height,
        min(box_w / width, 0.98),
        min(box_h / height, 0.98),
    )


def load_image(image_path: Path) -> Image.Image | None:
    suffix = image_path.suffix.lower()
    if suffix in {".heic", ".heif"}:
        temp_path = PROJECT_ROOT / "tmp" / f"{image_path.stem}.jpg"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["sips", "-s", "format", "jpeg", str(image_path), "--out", str(temp_path)],
            check=True,
            capture_output=True,
            text=False,
        )
        image = Image.open(temp_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        temp_path.unlink(missing_ok=True)
        return image

    image = Image.open(image_path)
    return ImageOps.exif_transpose(image).convert("RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import the user's class-folder tool photos into YOLO raw format.")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "data", help="Source folder with class subdirectories.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "raw_tool15_user",
        help="Output folder for normalized images and weak YOLO labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root
    output_root = args.output_root
    images_root = output_root / "images"
    labels_root = output_root / "labels"

    if output_root.exists():
        shutil.rmtree(output_root)
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, int] = {}

    for class_id, class_name in enumerate(TOOL_CLASSES):
        zh_name = TOOL_CLASSES_ZH[class_name]
        source_dir = source_root / zh_name
        class_images_dir = images_root / class_name
        class_labels_dir = labels_root / class_name
        class_images_dir.mkdir(parents=True, exist_ok=True)
        class_labels_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        if not source_dir.exists():
            print(f"[skip] missing source dir: {source_dir}", flush=True)
            summary[class_name] = 0
            continue

        for image_path in sorted(source_dir.iterdir()):
            if not image_path.is_file() or image_path.name.startswith("."):
                continue
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            try:
                image = load_image(image_path)
            except Exception as exc:
                print(f"[skip] {zh_name}/{image_path.name}: {exc}", flush=True)
                continue

            out_stem = f"{class_name}_{saved:04d}"
            out_image_path = class_images_dir / f"{out_stem}.jpg"
            out_label_path = class_labels_dir / f"{out_stem}.txt"

            image.save(out_image_path, format="JPEG", quality=95)
            x_center, y_center, box_w, box_h = detect_bbox(image)
            out_label_path.write_text(
                f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n",
                encoding="utf-8",
            )
            saved += 1

        summary[class_name] = saved
        print(f"[done] {class_name} <- {zh_name}: {saved}", flush=True)

    report_path = output_root / "summary.txt"
    report_lines = [f"{class_name}: {count}" for class_name, count in summary.items()]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"imported user data to {output_root}", flush=True)


if __name__ == "__main__":
    main()
