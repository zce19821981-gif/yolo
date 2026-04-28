from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .constants import NUM_CLASSES, TOOL_CLASSES
from .retinex import msrcr

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    key: str
    image_path: Path
    label_path: Path
    class_ids: tuple[int, ...]


def read_yolo_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    labels: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return labels

    with label_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"{label_path} 第 {line_number} 行不是标准 YOLO 标注。")
            class_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
            labels.append((class_id, x, y, w, h))
    return labels


def validate_labels(label_path: Path, num_classes: int = NUM_CLASSES) -> None:
    for class_id, x, y, w, h in read_yolo_label_file(label_path):
        if class_id < 0 or class_id >= num_classes:
            raise ValueError(f"{label_path} 存在越界类别 id: {class_id}")
        for value_name, value in (("x", x), ("y", y), ("w", w), ("h", h)):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{label_path} 中 {value_name} 值 {value} 超出 0~1 范围")
        if w <= 0.0 or h <= 0.0:
            raise ValueError(f"{label_path} 中宽高必须大于 0")


def collect_samples(source_images: Path, source_labels: Path) -> list[Sample]:
    samples: list[Sample] = []
    for image_path in sorted(source_images.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = image_path.relative_to(source_images)
        label_path = source_labels / relative.with_suffix(".txt")
        if not label_path.exists():
            continue
        validate_labels(label_path)
        label_records = read_yolo_label_file(label_path)
        class_ids = tuple(record[0] for record in label_records)
        key = str(relative.with_suffix("")).replace("/", "__")
        samples.append(Sample(key=key, image_path=image_path, label_path=label_path, class_ids=class_ids))
    if not samples:
        raise FileNotFoundError("没有找到成对的图片和标签，请检查 source-images/source-labels 路径。")
    return samples


def class_distribution(samples: list[Sample]) -> dict[str, int]:
    counter: Counter[int] = Counter()
    for sample in samples:
        counter.update(sample.class_ids)
    return {TOOL_CLASSES[class_id]: counter.get(class_id, 0) for class_id in range(NUM_CLASSES)}


def split_samples(
    samples: list[Sample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    rng = random.Random(seed)

    per_class_samples: dict[int, list[Sample]] = {}
    multi_label_samples: list[Sample] = []
    for sample in samples:
        unique_classes = set(sample.class_ids)
        if len(unique_classes) == 1:
            class_id = next(iter(unique_classes))
            per_class_samples.setdefault(class_id, []).append(sample)
        else:
            multi_label_samples.append(sample)

    train_samples: list[Sample] = []
    val_samples: list[Sample] = []
    test_samples: list[Sample] = []

    for class_id in sorted(per_class_samples):
        grouped = per_class_samples[class_id][:]
        rng.shuffle(grouped)
        total = len(grouped)

        val_count = max(1, int(round(total * val_ratio))) if total >= 10 else max(0, int(total * val_ratio))
        test_count = max(1, int(round(total * (1.0 - train_ratio - val_ratio)))) if total >= 10 else max(0, int(total * (1.0 - train_ratio - val_ratio)))
        train_count = total - val_count - test_count

        if train_count <= 0:
            train_count = max(1, total - 2)
            remaining = total - train_count
            val_count = remaining // 2
            test_count = remaining - val_count

        train_samples.extend(grouped[:train_count])
        val_samples.extend(grouped[train_count : train_count + val_count])
        test_samples.extend(grouped[train_count + val_count : train_count + val_count + test_count])

    if multi_label_samples:
        rng.shuffle(multi_label_samples)
        total = len(multi_label_samples)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        train_samples.extend(multi_label_samples[:train_end])
        val_samples.extend(multi_label_samples[train_end:val_end])
        test_samples.extend(multi_label_samples[val_end:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)
    return train_samples, val_samples, test_samples


def ensure_output_dirs(output_root: Path) -> None:
    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)


def save_label_file(label_path: Path, labels: list[tuple[int, float, float, float, float]]) -> None:
    with label_path.open("w", encoding="utf-8") as file:
        for class_id, x, y, w, h in labels:
            file.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def copy_file_stream(src: Path, dst: Path, chunk_size: int = 1024 * 1024) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fsrc, dst.open("wb") as fdst:
        shutil.copyfileobj(fsrc, fdst, length=chunk_size)


def copy_original_sample(sample: Sample, split: str, output_root: Path) -> None:
    dst_image = output_root / "images" / split / f"{sample.key}{sample.image_path.suffix.lower()}"
    dst_label = output_root / "labels" / split / f"{sample.key}.txt"
    copy_file_stream(sample.image_path, dst_image)
    copy_file_stream(sample.label_path, dst_label)


def _transform_labels(
    labels: list[tuple[int, float, float, float, float]],
    transform_name: str,
) -> list[tuple[int, float, float, float, float]]:
    transformed: list[tuple[int, float, float, float, float]] = []
    for class_id, x, y, w, h in labels:
        if transform_name == "hflip":
            transformed.append((class_id, 1.0 - x, y, w, h))
        elif transform_name == "vflip":
            transformed.append((class_id, x, 1.0 - y, w, h))
        elif transform_name == "rot90":
            transformed.append((class_id, y, 1.0 - x, h, w))
        elif transform_name == "rot180":
            transformed.append((class_id, 1.0 - x, 1.0 - y, w, h))
        elif transform_name == "rot270":
            transformed.append((class_id, 1.0 - y, x, h, w))
        else:
            transformed.append((class_id, x, y, w, h))
    return transformed


def augment_sample(
    sample: Sample,
    rng: random.Random,
    retinex_prob: float = 0.25,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    image = cv2.imread(str(sample.image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {sample.image_path}")

    labels = read_yolo_label_file(sample.label_path)
    geometry_transform = rng.choice(["none", "hflip", "vflip", "rot90", "rot180", "rot270"])

    if geometry_transform == "hflip":
        image = cv2.flip(image, 1)
    elif geometry_transform == "vflip":
        image = cv2.flip(image, 0)
    elif geometry_transform == "rot90":
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif geometry_transform == "rot180":
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif geometry_transform == "rot270":
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    labels = _transform_labels(labels, geometry_transform)

    if rng.random() < 0.8:
        alpha = rng.uniform(0.8, 1.2)
        beta = rng.randint(-25, 25)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if rng.random() < 0.5:
        noise_std = rng.uniform(5.0, 18.0)
        noise = np.random.normal(0.0, noise_std, size=image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < retinex_prob:
        image = msrcr(image)

    return image, labels


def image_sampling_weights(samples: list[Sample]) -> list[float]:
    per_class_counts = Counter()
    for sample in samples:
        per_class_counts.update(sample.class_ids)

    weights: list[float] = []
    for sample in samples:
        if not sample.class_ids:
            weights.append(1.0)
            continue
        inverse_frequencies = [1.0 / max(per_class_counts[class_id], 1) for class_id in set(sample.class_ids)]
        weights.append(max(inverse_frequencies))
    return weights


def augment_training_set(
    train_samples: list[Sample],
    output_root: Path,
    target_train_count: int,
    seed: int = 42,
    retinex_prob: float = 0.25,
) -> int:
    current_count = len(train_samples)
    if current_count >= target_train_count:
        return 0

    rng = random.Random(seed)
    weights = image_sampling_weights(train_samples)
    generated = 0

    progress = tqdm(total=target_train_count - current_count, desc="augment-train", unit="img")
    while current_count < target_train_count:
        sample = rng.choices(train_samples, weights=weights, k=1)[0]
        aug_image, aug_labels = augment_sample(sample, rng=rng, retinex_prob=retinex_prob)

        suffix = f"__aug_{generated:04d}"
        dst_image = output_root / "images" / "train" / f"{sample.key}{suffix}{sample.image_path.suffix.lower()}"
        dst_label = output_root / "labels" / "train" / f"{sample.key}{suffix}.txt"

        cv2.imwrite(str(dst_image), aug_image)
        save_label_file(dst_label, aug_labels)

        generated += 1
        current_count += 1
        progress.update(1)

    progress.close()
    return generated


def scan_label_distribution(label_dir: Path) -> dict[str, int]:
    counter: Counter[int] = Counter()
    for label_path in sorted(label_dir.glob("*.txt")):
        for class_id, _, _, _, _ in read_yolo_label_file(label_path):
            counter[class_id] += 1
    return {TOOL_CLASSES[class_id]: counter.get(class_id, 0) for class_id in range(NUM_CLASSES)}


def prepare_dataset(
    source_images: Path,
    source_labels: Path,
    output_root: Path,
    target_train_count: int = 640,
    seed: int = 42,
    retinex_prob: float = 0.25,
) -> dict[str, object]:
    ensure_output_dirs(output_root)
    samples = collect_samples(source_images=source_images, source_labels=source_labels)

    train_samples, val_samples, test_samples = split_samples(samples=samples, seed=seed)

    for sample in tqdm(train_samples, desc="copy-train", unit="img"):
        copy_original_sample(sample, "train", output_root)
    for sample in tqdm(val_samples, desc="copy-val", unit="img"):
        copy_original_sample(sample, "val", output_root)
    for sample in tqdm(test_samples, desc="copy-test", unit="img"):
        copy_original_sample(sample, "test", output_root)

    generated_count = augment_training_set(
        train_samples=train_samples,
        output_root=output_root,
        target_train_count=target_train_count,
        seed=seed,
        retinex_prob=retinex_prob,
    )

    summary = {
        "original_sample_count": len(samples),
        "train_count_before_augment": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "generated_train_count": generated_count,
        "train_count_after_augment": len(list((output_root / "images" / "train").glob("*"))),
        "original_class_distribution": class_distribution(samples),
        "train_class_distribution_after_augment": scan_label_distribution(output_root / "labels" / "train"),
    }

    report_path = output_root / "reports" / "dataset_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary
