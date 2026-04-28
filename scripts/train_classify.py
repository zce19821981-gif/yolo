from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from tool_yolo.inference import default_device

DEFAULT_DATA = PROJECT_ROOT / 'datasets' / 'tool15_cls_user'
DEFAULT_PROJECT = PROJECT_ROOT / 'runs' / 'classify'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a YOLOv8 classification model on the tool dataset.')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--model', type=str, default='yolov8s-cls.pt')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--imgsz', type=int, default=224)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default=default_device())
    parser.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    parser.add_argument('--name', type=str, default='tool15_cls_user_v1')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--patience', type=int, default=20)
    return parser.parse_args()


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit('缺少 ultralytics，请先安装依赖。') from exc

    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.25,
        fliplr=0.5,
        flipud=0.0,
        erasing=0.2,
        auto_augment='randaugment',
        degrees=5.0,
        scale=0.25,
        translate=0.05,
    )


if __name__ == '__main__':
    main()
