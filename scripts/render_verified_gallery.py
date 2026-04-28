from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_yolo.constants import TOOL_CLASSES, TOOL_CLASSES_ZH

SOURCES = {
    "turning_tool": ("Wikimedia Commons", "https://commons.wikimedia.org/wiki/File:LatheCarbideTippedBoringThreadingBars.jpg"),
    "milling_cutter": ("Wikimedia Commons", "https://commons.wikimedia.org/wiki/File:20251217_Face_Mill_in_action.jpg"),
    "drill_bit": ("Wikimedia Commons", "https://commons.wikimedia.org/wiki/File:Drill_bits_2017_G1.jpg"),
    "reamer": ("Wikimedia Commons", "https://commons.wikimedia.org/wiki/File:Escariador.jpg"),
    "boring_tool": ("Wikimedia Commons", "https://commons.wikimedia.org/wiki/File:Boring_bar_001.jpg"),
    "tap": ("Wikimedia Commons", "https://commons.wikimedia.org/wiki/File:1982_Presto_Tools_Sheffield_England_HSGT_machine_second_tap_G1_8_inch_BSP_4-flute_series_60600.jpg"),
    "broach": ("Howmet Fastening Systems", "https://catalog.howmetfasteners.com/viewitems/fluid-fitting-tools/rfopb5000wba-broach-tool-wobble-broach-machining"),
    "hob": ("Wikimedia Commons", "https://commons.wikimedia.org/wiki/File:Gear-hob.jpg"),
    "slotting_tool": ("Mike's Workshop", "https://mikesworkshop.weebly.com/toolpost-slotting-tool.html"),
    "planer_tool": ("Wikimedia Commons", "https://commons.wikimedia.org/wiki/File:Tool_bits.jpg"),
    "twist_drill": ("Wikimedia Commons", "https://commons.wikimedia.org/wiki/File:2mm_diamond_drill_bits_macro.jpg"),
    "end_mill": ("Wikimedia Commons", "https://commons.wikimedia.org/wiki/File:MillingCutterSlotEndMillBallnose.jpg"),
    "alloy_saw_blade": ("Pixabay", "https://pixabay.com/photos/circular-blades-saw-industrial-822159/"),
    "countersink": ("Wikimedia Commons", "https://commons.wikimedia.org/wiki/File:%E6%B2%89%E9%A0%AD%E9%91%BD%E9%A0%AD90%E5%BA%A6_1.jpg"),
    "counterbore_drill": ("Wikimedia Commons", "https://commons.wikimedia.org/wiki/File:CounterBores.jpg"),
}


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


def main() -> None:
    photos_dir = PROJECT_ROOT / "photos"
    class_dir = photos_dir / "class_samples"

    readme_lines = [
        "# 真实照片目录",
        "",
        "这里现在只放人工确认后的真实刀具照片。",
        "",
        "- `class_samples/`：15 类刀具各 1 张真实照片。",
        "- `real_samples/`：同步查看目录。",
        "- `tool15_contact_sheet.jpg`：15 类真实照片总览。",
        "",
        "| 英文类名 | 中文名称 | 来源 | 页面 |",
        "| --- | --- | --- | --- |",
    ]

    cell_w = 320
    cell_h = 240
    label_h = 56
    columns = 3
    cells: list[np.ndarray] = []

    for class_name in TOOL_CLASSES:
        image_path = class_dir / f"{class_name}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        body = fit_image(image, cell_w, cell_h)
        cell = np.full((cell_h + label_h, cell_w, 3), 250, dtype=np.uint8)
        cell[:cell_h] = body
        cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h + label_h - 1), (210, 210, 210), 1)
        zh_name = TOOL_CLASSES_ZH[class_name]
        cv2.putText(cell, class_name, (12, cell_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(cell, zh_name, (12, cell_h + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.01, (250, 250, 250), 1, cv2.LINE_AA)
        cells.append(cell)
        source_name, page = SOURCES[class_name]
        readme_lines.append(f"| {class_name} | {zh_name} | {source_name} | {page} |")

    rows = (len(cells) + columns - 1) // columns
    sheet = np.full(
        (rows * (cell_h + label_h) + (rows + 1) * 16, columns * cell_w + (columns + 1) * 16, 3),
        255,
        dtype=np.uint8,
    )
    for index, cell in enumerate(cells):
        row = index // columns
        col = index % columns
        y = 16 + row * (cell_h + label_h + 16)
        x = 16 + col * (cell_w + 16)
        sheet[y : y + cell.shape[0], x : x + cell.shape[1]] = cell

    cv2.imwrite(str(photos_dir / "tool15_contact_sheet.jpg"), sheet)
    (photos_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(f"rendered gallery from existing images in {class_dir}")


if __name__ == "__main__":
    main()
