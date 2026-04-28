from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_yolo.constants import TOOL_CLASSES, TOOL_CLASSES_ZH

USER_AGENT = "Mozilla/5.0"

CURATED_IMAGES = {
    "turning_tool": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Single_edge_cutting_tools.jpg/960px-Single_edge_cutting_tools.jpg",
        "source": "Wikimedia Commons",
        "page": "https://commons.wikimedia.org/wiki/File:Single_edge_cutting_tools.jpg",
    },
    "milling_cutter": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/20251217_Face_Mill_in_action.jpg/960px-20251217_Face_Mill_in_action.jpg",
        "source": "Wikimedia Commons",
        "page": "https://commons.wikimedia.org/wiki/File:20251217_Face_Mill_in_action.jpg",
    },
    "drill_bit": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Drill_bits_2017_G1.jpg/960px-Drill_bits_2017_G1.jpg",
        "source": "Wikimedia Commons",
        "page": "https://commons.wikimedia.org/wiki/File:Drill_bits_2017_G1.jpg",
    },
    "reamer": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/Escariador.jpg/960px-Escariador.jpg",
        "source": "Wikimedia Commons",
        "page": "https://commons.wikimedia.org/wiki/File:Escariador.jpg",
    },
    "boring_tool": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Boring_bar_001.jpg/960px-Boring_bar_001.jpg",
        "source": "Wikimedia Commons",
        "page": "https://commons.wikimedia.org/wiki/File:Boring_bar_001.jpg",
    },
    "tap": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/1982_Presto_Tools_Sheffield_England_HSGT_machine_second_tap_G1_8_inch_BSP_4-flute_series_60600.jpg/960px-1982_Presto_Tools_Sheffield_England_HSGT_machine_second_tap_G1_8_inch_BSP_4-flute_series_60600.jpg",
        "source": "Wikimedia Commons",
        "page": "https://commons.wikimedia.org/wiki/File:1982_Presto_Tools_Sheffield_England_HSGT_machine_second_tap_G1_8_inch_BSP_4-flute_series_60600.jpg",
    },
    "broach": {
        "url": "https://catalog.howmetfasteners.com/ImgMedium/RFOPB5000WBA_primary.JPG",
        "source": "Howmet Fastening Systems",
        "page": "https://catalog.howmetfasteners.com/viewitems/fluid-fitting-tools/rfopb5000wba-broach-tool-wobble-broach-machining",
    },
    "hob": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/f4/Gear-hob.jpg",
        "source": "Wikimedia Commons",
        "page": "https://commons.wikimedia.org/wiki/File:Gear-hob.jpg",
    },
    "slotting_tool": {
        "url": "http://mikesworkshop.weebly.com/uploads/4/3/5/0/4350192/8424349.jpg?533",
        "source": "Mike's Workshop",
        "page": "https://mikesworkshop.weebly.com/toolpost-slotting-tool.html",
    },
    "planer_tool": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/a4/Tool_bits.jpg",
        "source": "Wikimedia Commons",
        "page": "https://commons.wikimedia.org/wiki/File:Tool_bits.jpg",
    },
    "twist_drill": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/2mm_diamond_drill_bits_macro.jpg/960px-2mm_diamond_drill_bits_macro.jpg",
        "source": "Wikimedia Commons",
        "page": "https://commons.wikimedia.org/wiki/File:2mm_diamond_drill_bits_macro.jpg",
    },
    "end_mill": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/MillingCutterSlotEndMillBallnose.jpg/960px-MillingCutterSlotEndMillBallnose.jpg",
        "source": "Wikimedia Commons",
        "page": "https://commons.wikimedia.org/wiki/File:MillingCutterSlotEndMillBallnose.jpg",
    },
    "alloy_saw_blade": {
        "url": "https://cdn.pixabay.com/photo/2015/06/26/05/19/circular-822159_1280.jpg",
        "source": "Pixabay",
        "page": "https://pixabay.com/photos/circular-blades-saw-industrial-822159/",
    },
    "countersink": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/%E6%B2%89%E9%A0%AD%E9%91%BD%E9%A0%AD90%E5%BA%A6_1.jpg/960px-%E6%B2%89%E9%A0%AD%E9%91%BD%E9%A0%AD90%E5%BA%A6_1.jpg",
        "source": "Wikimedia Commons",
        "page": "https://commons.wikimedia.org/wiki/File:%E6%B2%89%E9%A0%AD%E9%91%BD%E9%A0%AD90%E5%BA%A6_1.jpg",
    },
    "counterbore_drill": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/CounterBores.jpg/960px-CounterBores.jpg",
        "source": "Wikimedia Commons",
        "page": "https://commons.wikimedia.org/wiki/File:CounterBores.jpg",
    },
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


def download_bytes(url: str) -> bytes:
    temp_path = PROJECT_ROOT / "tmp" / "verified_download.bin"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "-L",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "-A",
            USER_AGENT,
            url,
            "-o",
            str(temp_path),
        ],
        check=True,
        capture_output=True,
        text=False,
    )
    data = temp_path.read_bytes()
    temp_path.unlink(missing_ok=True)
    return data


def main() -> None:
    photos_dir = PROJECT_ROOT / "photos"
    class_dir = photos_dir / "class_samples"
    real_dir = photos_dir / "real_samples"
    if class_dir.exists():
        shutil.rmtree(class_dir)
    if real_dir.exists():
        shutil.rmtree(real_dir)
    class_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    readme_lines = [
        "# 真实照片目录",
        "",
        "这里现在只放人工确认过的真实刀具照片，不再混入合成图。",
        "",
        "- `class_samples/`：15 类刀具各 1 张真实照片。",
        "- `real_samples/`：和 `class_samples/` 同步的一份查看目录。",
        "- `tool15_contact_sheet.jpg`：15 类真实照片总览。",
        "",
        "| 英文类名 | 中文名称 | 来源 | 页面 |",
        "| --- | --- | --- | --- |",
    ]

    cells: list[np.ndarray] = []
    cell_w = 320
    cell_h = 240
    label_h = 56
    columns = 3

    for class_name in TOOL_CLASSES:
        item = CURATED_IMAGES[class_name]
        content = download_bytes(item["url"])
        out_name = f"{class_name}.jpg"
        class_path = class_dir / out_name
        real_path = real_dir / out_name
        class_path.write_bytes(content)
        real_path.write_bytes(content)

        image = cv2.imread(str(class_path))
        if image is None:
            continue
        body = fit_image(image, cell_w, cell_h)
        cell = np.full((cell_h + label_h, cell_w, 3), 250, dtype=np.uint8)
        cell[:cell_h] = body
        cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h + label_h - 1), (210, 210, 210), 1)
        zh_name = TOOL_CLASSES_ZH[class_name]
        cv2.putText(cell, zh_name, (12, cell_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(cell, class_name, (12, cell_h + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
        cells.append(cell)
        readme_lines.append(f"| {class_name} | {zh_name} | {item['source']} | {item['page']} |")

    rows = (len(cells) + columns - 1) // columns
    sheet = np.full((rows * (cell_h + label_h) + (rows + 1) * 16, columns * cell_w + (columns + 1) * 16, 3), 255, dtype=np.uint8)
    for index, cell in enumerate(cells):
        row = index // columns
        col = index % columns
        y = 16 + row * (cell_h + label_h + 16)
        x = 16 + col * (cell_w + 16)
        sheet[y : y + cell.shape[0], x : x + cell.shape[1]] = cell

    cv2.imwrite(str(photos_dir / "tool15_contact_sheet.jpg"), sheet)
    (photos_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(f"verified gallery saved to {photos_dir}")


if __name__ == "__main__":
    main()
