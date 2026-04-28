from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the YOLOv8 tool detection model.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "configs" / "data" / "tool15.yaml")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="YOLOv8 model checkpoint or model yaml.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "hyp" / "tool_article.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "detect")
    parser.add_argument("--name", type=str, default="tool_yolo_v8")
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("缺少 ultralytics，请先执行 `pip install -r requirements.txt`。") from exc

    args = parse_args()
    train_overrides = load_yaml(args.config)

    if args.epochs is not None:
        train_overrides["epochs"] = args.epochs
    if args.imgsz is not None:
        train_overrides["imgsz"] = args.imgsz
    if args.batch is not None:
        train_overrides["batch"] = args.batch
    if args.workers is not None:
        train_overrides["workers"] = args.workers

    train_overrides.update(
        {
            "data": str(args.data),
            "device": args.device,
            "project": str(args.project),
            "name": args.name,
        }
    )

    model = YOLO(args.model)
    model.train(**train_overrides)


if __name__ == "__main__":
    main()

