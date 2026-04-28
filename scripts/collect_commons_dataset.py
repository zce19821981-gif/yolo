from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_yolo.constants import TOOL_CLASSES

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "tool15-yolov8-dataset-builder/1.0"

SEARCH_TERMS = {
    "turning_tool": ['"lathe tool"', '"turning tool bit"', '"metal lathe tool bit"'],
    "milling_cutter": ['"milling cutter"', '"face milling cutter"', '"metal milling cutter"'],
    "drill_bit": ['"drill bit metal"', '"drill bit tool"', '"twist drill bit metal"'],
    "reamer": ['"machine reamer"', '"reamer tool"', '"metal reamer cutter"'],
    "boring_tool": ['"boring tool"', '"boring bar"', '"lathe boring bar"'],
    "tap": ['"thread tap tool"', '"tap tool metalworking"', '"thread cutting tap"'],
    "broach": ['"broach tool"', '"broaching tool"', '"broach cutter metal"'],
    "hob": ['"gear hob"', '"gear hobbing cutter"', '"hobbing cutter"'],
    "slotting_tool": ['"slotting tool metal"', '"slotting machine tool"', '"slotter cutting tool"'],
    "planer_tool": ['"planer tool metalworking"', '"planer cutting tool"', '"shaper tool bit metal"'],
    "twist_drill": ['"twist drill"', '"twist drill bit"', '"spiral drill bit metal"'],
    "end_mill": ['"end mill cutter"', '"carbide end mill"', '"endmill cutter metal"'],
    "alloy_saw_blade": ['"carbide circular saw blade"', '"alloy saw blade"', '"tct saw blade"'],
    "countersink": ['"countersink cutter"', '"countersink bit"', '"countersink tool metal"'],
    "counterbore_drill": ['"counterbore tool"', '"counterbore drill"', '"counterbore cutter"'],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download tool images from Wikimedia Commons and build pseudo labels.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "raw_tool15_commons",
        help="Output root for downloaded raw images and labels.",
    )
    parser.add_argument("--per-class", type=int, default=30, help="Target image count per class.")
    parser.add_argument("--min-width", type=int, default=320)
    parser.add_argument("--min-height", type=int, default=320)
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between API requests.")
    return parser.parse_args()


def commons_request(params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(
        COMMONS_API,
        params={**params, "format": "json"},
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def search_files(query: str, limit: int = 40) -> list[dict[str, Any]]:
    payload = commons_request(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {query}",
            "gsrnamespace": 6,
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
            "iiurlwidth": 640,
        }
    )
    pages = payload.get("query", {}).get("pages", {})
    return list(pages.values())


def file_extension_from_url(url: str, mime: str | None) -> str:
    guessed = Path(url.split("?")[0]).suffix.lower()
    if guessed:
        return guessed
    mime_guess = mimetypes.guess_extension(mime or "")
    return mime_guess or ".jpg"


def detect_bbox(image: np.ndarray) -> tuple[float, float, float, float]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    _, thresh_dark = cv2.threshold(blur, 245, 255, cv2.THRESH_BINARY_INV)
    _, thresh_light = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(edges, thresh_dark)
    mask = cv2.bitwise_or(mask, thresh_light)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_boxes: list[tuple[int, int, int, int]] = []
    min_area = max(300, int(height * width * 0.01))

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_area:
            continue
        if w >= width * 0.98 and h >= height * 0.98:
            continue
        valid_boxes.append((x, y, w, h))

    if not valid_boxes:
        return 0.5, 0.5, 0.9, 0.9

    x1 = min(box[0] for box in valid_boxes)
    y1 = min(box[1] for box in valid_boxes)
    x2 = max(box[0] + box[2] for box in valid_boxes)
    y2 = max(box[1] + box[3] for box in valid_boxes)

    x1 = max(0, x1 - int(width * 0.02))
    y1 = max(0, y1 - int(height * 0.02))
    x2 = min(width, x2 + int(width * 0.02))
    y2 = min(height, y2 + int(height * 0.02))

    bbox_width = max(1, x2 - x1)
    bbox_height = max(1, y2 - y1)
    x_center = (x1 + bbox_width / 2.0) / width
    y_center = (y1 + bbox_height / 2.0) / height
    norm_w = bbox_width / width
    norm_h = bbox_height / height
    return x_center, y_center, min(norm_w, 0.98), min(norm_h, 0.98)


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def download_image(url: str) -> bytes | None:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type:
        return None
    return response.content


def main() -> None:
    args = parse_args()
    images_dir = args.output_root / "images"
    labels_dir = args.output_root / "labels"
    reports_dir = args.output_root / "reports"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, list[dict[str, Any]]] = {}
    seen_hashes: set[str] = set()

    for class_id, class_name in enumerate(TOOL_CLASSES):
        class_manifest: list[dict[str, Any]] = []
        manifest[class_name] = class_manifest
        collected = 0

        for term in SEARCH_TERMS[class_name]:
            if collected >= args.per_class:
                break
            try:
                pages = search_files(term, limit=max(args.per_class * 2, 40))
            except Exception as exc:
                print(f"[warn] search failed for {class_name} / {term}: {exc}", flush=True)
                continue
            time.sleep(args.sleep)

            for page in pages:
                if collected >= args.per_class:
                    break
                try:
                    imageinfo = (page.get("imageinfo") or [{}])[0]
                    url = imageinfo.get("thumburl") or imageinfo.get("url")
                    mime = imageinfo.get("mime")
                    width = int(imageinfo.get("width") or 0)
                    height = int(imageinfo.get("height") or 0)

                    if not url or width < args.min_width or height < args.min_height:
                        continue
                    if mime and mime not in {"image/jpeg", "image/png", "image/webp"}:
                        continue

                    try:
                        content = download_image(url)
                    except Exception as exc:
                        print(f"[warn] download failed for {url}: {exc}", flush=True)
                        continue
                    time.sleep(args.sleep)
                    if not content:
                        continue

                    digest = sha1_bytes(content)
                    if digest in seen_hashes:
                        continue

                    image_array = np.frombuffer(content, dtype=np.uint8)
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    if image is None:
                        continue

                    img_h, img_w = image.shape[:2]
                    if img_w < args.min_width or img_h < args.min_height:
                        continue

                    ext = file_extension_from_url(url, mime)
                    image_name = f"{class_name}_{collected:03d}{ext}"
                    image_path = images_dir / image_name
                    label_path = labels_dir / f"{class_name}_{collected:03d}.txt"

                    success = cv2.imwrite(str(image_path), image)
                    if not success:
                        continue

                    x_center, y_center, box_w, box_h = detect_bbox(image)
                    label_path.write_text(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n",
                        encoding="utf-8",
                    )

                    metadata = imageinfo.get("extmetadata", {}) or {}
                    class_manifest.append(
                        {
                            "image_name": image_name,
                            "source_page": f"https://commons.wikimedia.org/wiki/{page.get('title', '').replace(' ', '_')}",
                            "image_url": url,
                            "license_short_name": (metadata.get("LicenseShortName", {}) or {}).get("value"),
                            "artist": (metadata.get("Artist", {}) or {}).get("value"),
                            "credit": (metadata.get("Credit", {}) or {}).get("value"),
                            "search_term": term,
                        }
                    )
                    seen_hashes.add(digest)
                    collected += 1
                    report_path = reports_dir / "commons_manifest.json"
                    report_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"[ok] {class_name}: {collected}/{args.per_class} via {term}", flush=True)
                except Exception as exc:
                    print(f"[warn] page processing failed for {class_name}: {exc}", flush=True)
                    continue

        print(f"{class_name}: collected {collected}", flush=True)

    report_path = reports_dir / "commons_manifest.json"
    report_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest saved to {report_path}", flush=True)


if __name__ == "__main__":
    main()
