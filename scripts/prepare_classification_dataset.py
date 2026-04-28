from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
ZH_TO_EN = {
    '车刀': 'turning_tool',
    '铣刀': 'milling_cutter',
    '钻头': 'drill_bit',
    '铰刀': 'reamer',
    '镗刀': 'boring_tool',
    '丝锥': 'tap',
    '拉刀': 'broach',
    '滚刀': 'hob',
    '插刀': 'slotting_tool',
    '刨刀': 'planer_tool',
    '麻花钻': 'twist_drill',
    '立铣刀': 'end_mill',
    '合金锯片': 'alloy_saw_blade',
    '锪钻': 'countersink',
    '扩孔钻': 'counterbore_drill',
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / 'data'
DEFAULT_OUTPUT = PROJECT_ROOT / 'datasets' / 'tool15_cls_user'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare a folder-classification dataset for YOLOv8-cls.')
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def collect_images(class_dir: Path) -> list[Path]:
    return sorted(path for path in class_dir.rglob('*') if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if total < 3:
        raise ValueError(f'每个类别至少需要 3 张图片，当前只有 {total} 张。')

    train_count = max(1, int(round(total * train_ratio)))
    val_count = max(1, int(round(total * val_ratio)))
    test_count = total - train_count - val_count

    if test_count < 1:
        test_count = 1
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        else:
            val_count -= 1

    while train_count + val_count + test_count > total:
        if train_count > val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            test_count -= 1

    while train_count + val_count + test_count < total:
        train_count += 1

    return train_count, val_count, test_count


def copy_sample(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise SystemExit('train/val/test 比例之和必须等于 1。')

    if not args.source_root.exists():
        raise SystemExit(f'未找到原始数据目录: {args.source_root}')

    if args.output_root.exists() and args.overwrite:
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    split_summary: dict[str, dict[str, int]] = {}
    class_map: dict[str, str] = {}

    for class_dir in sorted(path for path in args.source_root.iterdir() if path.is_dir()):
        zh_name = class_dir.name
        if zh_name not in ZH_TO_EN:
            raise SystemExit(f'未识别的类别目录: {zh_name}')
        en_name = ZH_TO_EN[zh_name]
        class_map[en_name] = zh_name

        images = collect_images(class_dir)
        if not images:
            continue
        rng.shuffle(images)
        train_count, val_count, test_count = split_counts(len(images), args.train_ratio, args.val_ratio, args.test_ratio)

        train_samples = images[:train_count]
        val_samples = images[train_count:train_count + val_count]
        test_samples = images[train_count + val_count:train_count + val_count + test_count]

        for split_name, split_samples in (
            ('train', train_samples),
            ('val', val_samples),
            ('test', test_samples),
        ):
            for idx, sample in enumerate(split_samples):
                dst_name = f'{en_name}_{idx:03d}{sample.suffix.lower()}'
                copy_sample(sample, args.output_root / split_name / en_name / dst_name)

        split_summary[en_name] = {
            'zh_name': zh_name,
            'total': len(images),
            'train': len(train_samples),
            'val': len(val_samples),
            'test': len(test_samples),
        }

    report = {
        'source_root': str(args.source_root),
        'output_root': str(args.output_root),
        'seed': args.seed,
        'ratios': {
            'train': args.train_ratio,
            'val': args.val_ratio,
            'test': args.test_ratio,
        },
        'class_map': class_map,
        'splits': split_summary,
        'total_images': sum(item['total'] for item in split_summary.values()),
    }

    report_path = args.output_root / 'split_report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Prepared classification dataset at: {args.output_root}')
    print(f'Report saved to: {report_path}')


if __name__ == '__main__':
    main()
