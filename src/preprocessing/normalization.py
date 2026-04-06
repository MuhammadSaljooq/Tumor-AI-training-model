"""Normalization constants and helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image pixel values to [0, 1].

    Args:
        image: Input uint8/float image array.

    Returns:
        Float32 image array with values clipped to [0, 1].
    """
    normalized = image.astype(np.float32) / 255.0
    return np.clip(normalized, 0.0, 1.0)


def normalize_pixel_values(image: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for image normalization."""
    return normalize_image(image)


def get_normalization_stats() -> Tuple[list[float], list[float]]:
    """Return ImageNet mean and standard deviation.

    Returns:
        Tuple containing mean and std lists.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return mean, std
