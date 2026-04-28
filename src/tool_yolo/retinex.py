from __future__ import annotations

import cv2
import numpy as np


def _replace_zeros(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image[image <= 0] = 1.0
    return image


def single_scale_retinex(image: np.ndarray, sigma: float = 30.0) -> np.ndarray:
    image = _replace_zeros(image)
    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    blur = _replace_zeros(blur)
    return np.log10(image) - np.log10(blur)


def multi_scale_retinex(image: np.ndarray, sigmas: tuple[float, ...] = (15.0, 80.0, 250.0)) -> np.ndarray:
    retinex = np.zeros_like(image, dtype=np.float32)
    for sigma in sigmas:
        retinex += single_scale_retinex(image, sigma)
    return retinex / float(len(sigmas))


def msrcr(
    image: np.ndarray,
    sigmas: tuple[float, ...] = (15.0, 80.0, 250.0),
    gain: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Multi-scale Retinex with simple dynamic-range restoration."""
    image = image.astype(np.float32) + 1.0
    retinex = multi_scale_retinex(image, sigmas)
    restored = gain * retinex + offset

    normalized_channels = []
    for channel in cv2.split(restored):
        channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
        normalized_channels.append(channel.astype(np.uint8))
    return cv2.merge(normalized_channels)

