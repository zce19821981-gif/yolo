from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_yolo.constants import TOOL_CLASSES

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "tool15-curated-downloader/1.0"

COMMONS_FILES: dict[str, list[str]] = {
    "turning_tool": [
        "File:Single_edge_cutting_tools.jpg",
        "File:Ceratizit Drehen.jpg",
        "File:WerkzeughalterProfil.jpg",
    ],
    "milling_cutter": [
        "File:20251217_Face_Mill_in_action.jpg",
        "File:MillingCutterCarbideTippedFaceMill-INT40.jpg",
        "File:Ceratizit Fräser.jpg",
    ],
    "drill_bit": [
        "File:Drill_bits_2017_G1.jpg",
        "File:2010-01-21 Craftsman Professional cobalt drill bit set.jpg",
        "File:Cobalt-Drill-Bit HSS-Co-10mm 61674-480x360 (5000494774).jpg",
    ],
    "reamer": [
        "File:Escariador.jpg",
        "File:Alésoir expansible.jpg",
        "File:Duplex-chucking-reamer.jpg",
    ],
    "boring_tool": [
        "File:Boring bar 001.jpg",
        "File:Lathe Boring.jpg",
    ],
    "tap": [
        "File:1982 Presto Tools Sheffield England HSGT machine second tap G1 8 inch BSP 4-flute series 60600.jpg",
        "File:Ca 2000 precision tap set DIN 352 HSS for M2 threads by LUX Werkzeuge Wermelskirchen Germany.jpg",
        "File:Gewindebohrer.jpg",
        "File:Tap and T-wrench.jpg",
    ],
    "broach": [
        "File:Wobble Broach Tool.jpg",
        "File:Hand-broachclamp.jpg",
        "File:Raemnadelschaft.JPG",
    ],
    "hob": [
        "File:Gear-hob.jpg",
        "File:AlCrTiN-CoatedHob NanoShieldPVD Thailand.jpg",
        "File:Spline hobs.JPG",
    ],
    "planer_tool": [
        "File:Tool bits.jpg",
        "File:ShaperSlideClapperBox.jpg",
    ],
    "twist_drill": [
        "File:2mm_diamond_drill_bits_macro.jpg",
        "File:2010-01-21 Craftsman Professional cobalt drill bit set.jpg",
        "File:Bohrer Hartmetall mit Innerer Kühlschmierstoffzufuhr.jpg",
    ],
    "end_mill": [
        "File:MillingCutterSlotEndMillBallnose.jpg",
        "File:AlTiNCoatedEndmill NanoShieldPVD Thailand.JPG",
        "File:HM-Fräser.jpg",
    ],
    "alloy_saw_blade": [
        "File:Circular Saw Blade.jpg",
        "File:Ca 1980 Amersaw 7 1 4 inch construction circular saw blade by American Saw and Tool Louisville Kentucky USA.jpg",
    ],
    "countersink": [
        "File:4FlutedCountersink.jpg",
        "File:Mèches à chambrer.jpg",
        "File:%E6%B2%89%E9%A0%AD%E9%91%BD%E9%A0%AD90%E5%BA%A6_1.jpg",
    ],
    "counterbore_drill": [
        "File:CounterBores.jpg",
        "File:Fraises à lamer.jpg",
    ],
}

DIRECT_IMAGES: dict[str, list[dict[str, str]]] = {
    "milling_cutter": [
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/20251217_Face_Mill_in_action.jpg/960px-20251217_Face_Mill_in_action.jpg",
            "page": "https://commons.wikimedia.org/wiki/File:20251217_Face_Mill_in_action.jpg",
            "source": "Wikimedia Commons",
        }
    ],
    "drill_bit": [
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Drill_bits_2017_G1.jpg/960px-Drill_bits_2017_G1.jpg",
            "page": "https://commons.wikimedia.org/wiki/File:Drill_bits_2017_G1.jpg",
            "source": "Wikimedia Commons",
        }
    ],
    "reamer": [
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/Escariador.jpg/960px-Escariador.jpg",
            "page": "https://commons.wikimedia.org/wiki/File:Escariador.jpg",
            "source": "Wikimedia Commons",
        }
    ],
    "boring_tool": [
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Boring_bar_001.jpg/960px-Boring_bar_001.jpg",
            "page": "https://commons.wikimedia.org/wiki/File:Boring_bar_001.jpg",
            "source": "Wikimedia Commons",
        }
    ],
    "slotting_tool": [
        {
            "url": "http://mikesworkshop.weebly.com/uploads/4/3/5/0/4350192/8424349.jpg?533",
            "page": "https://mikesworkshop.weebly.com/toolpost-slotting-tool.html",
            "source": "Mike's Workshop",
        },
        {
            "url": "https://cnchome-beyond.com/cdn/shop/files/QEFD2020R17-3.jpg?v=1719976628",
            "page": "https://cnchome-beyond.com/products/qefd2020r17-3",
            "source": "BEYOND Tools",
        },
    ],
    "twist_drill": [
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/2mm_diamond_drill_bits_macro.jpg/960px-2mm_diamond_drill_bits_macro.jpg",
            "page": "https://commons.wikimedia.org/wiki/File:2mm_diamond_drill_bits_macro.jpg",
            "source": "Wikimedia Commons",
        }
    ],
    "end_mill": [
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/MillingCutterSlotEndMillBallnose.jpg/960px-MillingCutterSlotEndMillBallnose.jpg",
            "page": "https://commons.wikimedia.org/wiki/File:MillingCutterSlotEndMillBallnose.jpg",
            "source": "Wikimedia Commons",
        }
    ],
    "alloy_saw_blade": [
        {
            "url": "https://cdn.pixabay.com/photo/2015/06/26/05/19/circular-822159_1280.jpg",
            "page": "https://pixabay.com/photos/circular-blades-saw-industrial-822159/",
            "source": "Pixabay",
        }
    ],
    "countersink": [
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/%E6%B2%89%E9%A0%AD%E9%91%BD%E9%A0%AD90%E5%BA%A6_1.jpg/960px-%E6%B2%89%E9%A0%AD%E9%91%BD%E9%A0%AD90%E5%BA%A6_1.jpg",
            "page": "https://commons.wikimedia.org/wiki/File:%E6%B2%89%E9%A0%AD%E9%91%BD%E9%A0%AD90%E5%BA%A6_1.jpg",
            "source": "Wikimedia Commons",
        }
    ],
    "planer_tool": [
        {
            "url": "https://www.shapiroltd.com/cdn/shop/files/iStock_000005676592Small_1200x1200.jpg?v=1614310684",
            "page": "https://www.shapiroltd.com/collections/boring-bars",
            "source": "Shapiro Supply",
        }
    ],
    "broach": [
        {
            "url": "https://catalog.howmetfasteners.com/ImgMedium/RFOPB5000WBA_primary.JPG",
            "page": "https://catalog.howmetfasteners.com/viewitems/fluid-fitting-tools/rfopb5000wba-broach-tool-wobble-broach-machining",
            "source": "Howmet Fastening Systems",
        },
        {
            "url": "https://genswiss.com/media/catalog/category/CAT2-broachingtools_1_1.jpg",
            "page": "https://genswiss.com/id-tools/broach-tools.html",
            "source": "GenSwiss",
        },
    ],
    "counterbore_drill": [
        {
            "url": "https://aoshiji.com/wp-content/uploads/2025/10/og-combination-drill-counterbore-tool-1200x627-1.jpg",
            "page": "https://aoshiji.com/custom-cutting-tool/combination-drill-counterbore-tool/",
            "source": "Aoshiji",
        },
        {
            "url": "https://aoshiji.com/wp-content/uploads/2025/08/combination-drill-counterbore-tool-aoshiji-1024x240.webp",
            "page": "https://aoshiji.com/custom-cutting-tool/combination-drill-counterbore-tool/",
            "source": "Aoshiji",
        },
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/CounterBores.jpg/960px-CounterBores.jpg",
            "page": "https://commons.wikimedia.org/wiki/File:CounterBores.jpg",
            "source": "Wikimedia Commons",
        },
    ],
}


def commons_request(params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(
        COMMONS_API,
        params={**params, "format": "json"},
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def fetch_commons_image_record(file_title: str) -> dict[str, str] | None:
    data = commons_request(
        {
            "action": "query",
            "titles": file_title,
            "prop": "imageinfo",
            "iiprop": "url",
        }
    )
    pages = list((data.get("query") or {}).get("pages", {}).values())
    if not pages:
        return None
    page = pages[0]
    imageinfo = (page.get("imageinfo") or [{}])[0]
    image_url = imageinfo.get("url")
    if not image_url:
        return None
    return {
        "url": image_url,
        "page": f"https://commons.wikimedia.org/wiki/{file_title.replace(' ', '_')}",
        "source": "Wikimedia Commons",
    }


def download_bytes(url: str) -> bytes:
    temp_path = PROJECT_ROOT / "tmp" / "curated_download.bin"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "-L",
            "-f",
            "--retry",
            "4",
            "--retry-delay",
            "3",
            "--max-time",
            "30",
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


def collect_records() -> dict[str, list[dict[str, str]]]:
    records: dict[str, list[dict[str, str]]] = {class_name: [] for class_name in TOOL_CLASSES}
    for class_name, titles in COMMONS_FILES.items():
        for title in titles:
            print(f"[commons] {class_name} <- {title}", flush=True)
            try:
                record = fetch_commons_image_record(title)
            except Exception as exc:
                print(f"[skip] commons lookup failed for {title}: {exc}", flush=True)
                record = None
            if record is not None:
                records[class_name].append(record)
    for class_name, items in DIRECT_IMAGES.items():
        records.setdefault(class_name, []).extend(items)
    return records


def main() -> None:
    photos_root = PROJECT_ROOT / "photos"
    gallery_root = photos_root / "curated_real"
    reports_root = gallery_root / "reports"
    if gallery_root.exists():
        shutil.rmtree(gallery_root)
    gallery_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    records = collect_records()
    manifest: dict[str, list[dict[str, str]]] = {class_name: [] for class_name in TOOL_CLASSES}

    readme_lines = [
        "# 多图真实照片库",
        "",
        "这里按 15 类刀具整理了多张真实图片，优先使用 Wikimedia Commons 和可直接下载的产品页主图。",
        "",
        "| 类别 | 数量 | 目录 |",
        "| --- | --- | --- |",
    ]

    for class_name in TOOL_CLASSES:
        class_dir = gallery_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        class_records = records.get(class_name, [])
        saved = 0
        for index, record in enumerate(class_records):
            print(f"[download] {class_name} #{index:02d} {record['url']}", flush=True)
            try:
                data = download_bytes(record["url"])
            except Exception as exc:
                print(f"[skip] download failed for {class_name} #{index:02d}: {exc}", flush=True)
                continue
            file_path = class_dir / f"{class_name}_{index:02d}.jpg"
            file_path.write_bytes(data)
            manifest[class_name].append(
                {
                    "file": file_path.name,
                    "page": record["page"],
                    "source": record["source"],
                    "image_url": record["url"],
                }
            )
            saved += 1
        print(f"[done] {class_name}: {saved}", flush=True)
        readme_lines.append(f"| {class_name} | {saved} | `photos/curated_real/{class_name}` |")

    manifest_path = reports_root / "curated_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    (gallery_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(f"curated real photos saved to {gallery_root}")
    print(f"manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
