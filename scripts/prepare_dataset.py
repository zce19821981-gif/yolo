from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_yolo.dataset_tools import prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLOv8 dataset for the 15-class tool project.")
    parser.add_argument("--source-images", type=Path, required=True, help="Path to raw images directory.")
    parser.add_argument("--source-labels", type=Path, required=True, help="Path to raw YOLO labels directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "tool15",
        help="Output YOLO dataset root.",
    )
    parser.add_argument("--target-train-count", type=int, default=640, help="Target image count for train split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--retinex-prob", type=float, default=0.25, help="Probability of Retinex augmentation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_dataset(
        source_images=args.source_images,
        source_labels=args.source_labels,
        output_root=args.output,
        target_train_count=args.target_train_count,
        seed=args.seed,
        retinex_prob=args.retinex_prob,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

