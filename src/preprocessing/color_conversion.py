"""Color-space conversion helpers."""

from __future__ import annotations

import cv2
import numpy as np


def convert_color(image: np.ndarray, mode: str = "rgb") -> np.ndarray:
    """Convert image color mode.

    Args:
        image: Input image in grayscale or RGB format.
        mode: Output mode, either ``"grayscale"`` or ``"rgb"``.

    Returns:
        Converted image. For grayscale mode, output is 3-channel grayscale RGB.

    Raises:
        ValueError: If an unsupported conversion mode is provided.
    """
    normalized_mode = mode.lower().strip()
    if normalized_mode == "grayscale":
        if image.ndim == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if normalized_mode == "rgb":
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image
    raise ValueError(f"Unsupported color conversion mode: {mode}")


def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for grayscale conversion."""
    return convert_color(image, mode="grayscale")
