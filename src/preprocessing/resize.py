"""Image resizing helpers."""

from __future__ import annotations

import cv2
import numpy as np


def resize_image(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize an image to a square target size.

    Args:
        image: Input image.
        size: Target size in pixels.

    Returns:
        Resized image.
    """
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
