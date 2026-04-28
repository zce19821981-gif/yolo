from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_yolo.constants import TOOL_CLASSES, TOOL_CLASSES_ZH


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h))
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def pick_sample(class_name: str, real_dir: Path, synth_dir: Path) -> tuple[Path, str]:
    real_candidates = sorted(real_dir.glob(f"{class_name}_*"))
    if real_candidates:
        return real_candidates[0], "real"
    synth_candidates = sorted(synth_dir.glob(f"{class_name}_*"))
    if synth_candidates:
        return synth_candidates[0], "synthetic"
    raise FileNotFoundError(f"No image found for {class_name}")


def main() -> None:
    real_dir = PROJECT_ROOT / "datasets" / "raw_tool15_commons" / "images"
    synth_dir = PROJECT_ROOT / "datasets" / "raw_tool15_synth" / "images"
    photos_dir = PROJECT_ROOT / "photos"
    class_dir = photos_dir / "class_samples"
    real_out_dir = photos_dir / "real_samples"
    class_dir.mkdir(parents=True, exist_ok=True)
    real_out_dir.mkdir(parents=True, exist_ok=True)

    preview_cells: list[np.ndarray] = []
    readme_lines = [
        "# 图片查看目录",
        "",
        "这个目录是为了方便直接看图，不用再进入数据集深层目录。",
        "",
        "- `class_samples/`：15 类刀具的代表图，每类 1 张。",
        "- `real_samples/`：当前抓到的公开真实样本。",
        "- `tool15_contact_sheet.jpg`：15 类总览拼图。",
        "",
        "| 英文类名 | 中文名称 | 代表图来源 |",
        "| --- | --- | --- |",
    ]

    for source_file in sorted(real_dir.glob("*")):
        shutil.copy2(source_file, real_out_dir / source_file.name)

    cell_w = 320
    cell_h = 240
    label_h = 52
    columns = 3

    for class_name in TOOL_CLASSES:
        image_path, source_kind = pick_sample(class_name, real_dir=real_dir, synth_dir=synth_dir)
        target_path = class_dir / f"{class_name}{image_path.suffix.lower()}"
        shutil.copy2(image_path, target_path)

        image = cv2.imread(str(image_path))
        if image is None:
            continue
        body = fit_image(image, cell_w, cell_h)
        cell = np.full((cell_h + label_h, cell_w, 3), 250, dtype=np.uint8)
        cell[:cell_h] = body
        cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h + label_h - 1), (210, 210, 210), 1)
        zh_name = TOOL_CLASSES_ZH[class_name]
        cv2.putText(cell, zh_name, (12, cell_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(
            cell,
            f"{class_name} [{source_kind}]",
            (12, cell_h + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )
        preview_cells.append(cell)
        readme_lines.append(f"| {class_name} | {zh_name} | {source_kind} |")

    rows = (len(preview_cells) + columns - 1) // columns
    sheet = np.full((rows * (cell_h + label_h) + (rows + 1) * 16, columns * cell_w + (columns + 1) * 16, 3), 255, dtype=np.uint8)

    for index, cell in enumerate(preview_cells):
        row = index // columns
        col = index % columns
        y = 16 + row * (cell_h + label_h + 16)
        x = 16 + col * (cell_w + 16)
        sheet[y : y + cell.shape[0], x : x + cell.shape[1]] = cell

    cv2.imwrite(str(photos_dir / "tool15_contact_sheet.jpg"), sheet)
    (photos_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(f"gallery saved to {photos_dir}")


if __name__ == "__main__":
    main()
