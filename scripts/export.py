from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLOv8 weights for deployment.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to best.pt")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "engine", "torchscript", "openvino"])
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--half", action="store_true", help="Use FP16 export when supported.")
    parser.add_argument("--int8", action="store_true", help="Use INT8 export when supported.")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic input shape.")
    return parser.parse_args()


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("缺少 ultralytics，请先执行 `pip install -r requirements.txt`。") from exc

    args = parse_args()
    model = YOLO(str(args.weights))
    model.export(
        format=args.format,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        int8=args.int8,
        dynamic=args.dynamic,
    )


if __name__ == "__main__":
    main()
