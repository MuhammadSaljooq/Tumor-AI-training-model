"""Contrast enhancement utilities."""

from __future__ import annotations

import cv2
import numpy as np


def enhance_image(image: np.ndarray) -> np.ndarray:
    """Enhance MRI image contrast using CLAHE.

    Args:
        image: Input image in grayscale or RGB format.

    Returns:
        Contrast-enhanced image with same channel layout as input.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    if image.ndim == 2:
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Backward-compatible alias for CLAHE enhancement."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if image.ndim == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
