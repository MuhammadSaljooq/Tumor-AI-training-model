"""End-to-end preprocessing pipeline for MRI datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .color_conversion import convert_color
from .enhancement import enhance_image
from .noise_removal import remove_noise
from .normalization import normalize_image
from .resize import resize_image


class PreprocessingPipeline:
    """Pipeline that preprocesses MRI images and full datasets."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize preprocessing pipeline.

        Args:
            config: Dictionary with preprocessing and image parameters.
        """
        self.config = config
        self.img_size = int(config.get("img_size", 224))
        self.noise_method = str(config.get("noise_method", "gaussian"))
        self.color_mode = str(config.get("color_mode", "grayscale"))

    def process_image(self, image_path: str) -> np.ndarray:
        """Read and preprocess a single image.

        Processing steps:
            1) read image
            2) noise removal
            3) contrast enhancement
            4) color conversion
            5) resize
            6) normalization to [0, 1]

        Args:
            image_path: Path to input image.

        Returns:
            Preprocessed image as float32 numpy array in [0, 1].

        Raises:
            FileNotFoundError: If image cannot be loaded.
        """
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = remove_noise(image_rgb, method=self.noise_method)
        image_rgb = enhance_image(image_rgb)
        image_rgb = convert_color(image_rgb, mode=self.color_mode)
        image_rgb = resize_image(image_rgb, size=self.img_size)
        image_rgb = normalize_image(image_rgb)
        return image_rgb

    def process_dataset(self, raw_dir: str, output_dir: str) -> None:
        """Preprocess all images from raw directory and save outputs.

        Args:
            raw_dir: Root directory containing class subfolders of raw images.
            output_dir: Destination root directory for processed images.
        """
        raw_path = Path(raw_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        image_paths = [p for p in raw_path.rglob("*") if p.is_file() and p.suffix.lower() in valid_ext]

        for image_path in tqdm(image_paths, desc="Preprocessing dataset"):
            rel_path = image_path.relative_to(raw_path)
            target_path = output_path / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)

            processed = self.process_image(str(image_path))
            processed_uint8 = np.clip(processed * 255.0, 0, 255).astype(np.uint8)
            processed_bgr = cv2.cvtColor(processed_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(target_path), processed_bgr)
