"""Noise removal helpers for MRI preprocessing."""

from __future__ import annotations

import cv2
import numpy as np


def remove_noise(
    image: np.ndarray,
    method: str = "gaussian",
) -> np.ndarray:
    """Remove image noise using Gaussian blur or median filtering.

    Args:
        image: Input image as an HxW or HxWxC numpy array.
        method: Denoising method, either ``"gaussian"`` or ``"median"``.

    Returns:
        Denoised image with the same shape as the input.

    Raises:
        ValueError: If an invalid method is provided.
    """
    normalized_method = method.lower().strip()
    if normalized_method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    if normalized_method == "median":
        return cv2.medianBlur(image, 5)
    raise ValueError(f"Unsupported denoising method: {method}")
