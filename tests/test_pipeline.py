"""Integration-style tests for core project pipeline components."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.evaluation import compute_metrics
from src.models import get_model
from src.preprocessing import convert_color, enhance_image, normalize_image, remove_noise, resize_image
from src.utils import get_data_loaders


class TestPipeline(unittest.TestCase):
    """Covers preprocessing, model forward, dataloaders, and metrics."""

    def test_preprocessing_functions_with_dummy_image(self) -> None:
        """Verify preprocessing helpers operate on a random image."""
        image = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)

        denoised = remove_noise(image, method="gaussian")
        enhanced = enhance_image(denoised)
        gray = convert_color(enhanced, mode="grayscale")
        resized = resize_image(enhanced, size=224)
        normalized = normalize_image(resized)

        self.assertEqual(denoised.shape, image.shape)
        self.assertEqual(enhanced.shape, image.shape)
        self.assertEqual(gray.shape[:2], image.shape[:2])
        self.assertEqual(resized.shape, image.shape)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertTrue(np.all(normalized >= 0.0))
        self.assertTrue(np.all(normalized <= 1.0))

    def test_model_forward_passes(self) -> None:
        """Verify each model produces logits of expected shape."""
        x = torch.randn(2, 3, 224, 224)
        for model_name in ("resnet50", "vit", "hybrid"):
            model = get_model(model_name, num_classes=4, pretrained=False)
            model.eval()
            with torch.no_grad():
                out = model(x)
            self.assertEqual(tuple(out.shape), (2, 4))

    def test_data_loader_instantiation(self) -> None:
        """Build dataloaders from temp image folders and validate splits."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]

            for class_name in class_names:
                class_dir = root / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                # Keep enough samples per class for 70/15/15 stratified split.
                for idx in range(20):
                    img = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
                    Image.fromarray(img).save(class_dir / f"{class_name}_{idx}.png")

            config = {"data": {"batch_size": 8, "img_size": 224}}
            train_loader, val_loader, test_loader, loaded_classes = get_data_loaders(
                data_dir=str(root),
                config=config,
            )

            self.assertEqual(set(loaded_classes), set(class_names))
            self.assertGreater(len(train_loader.dataset), 0)
            self.assertGreater(len(val_loader.dataset), 0)
            self.assertGreater(len(test_loader.dataset), 0)

    def test_data_loader_explicit_training_testing_layout(self) -> None:
        """Verify loader correctly handles Training/Testing split folders and aliases."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for split in ("Training", "Testing"):
                for class_name in ("glioma", "meningioma", "pituitary", "notumor"):
                    class_dir = root / split / class_name
                    class_dir.mkdir(parents=True, exist_ok=True)
                    for idx in range(8):
                        img = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
                        Image.fromarray(img).save(class_dir / f"{class_name}_{idx}.png")

            config = {
                "data": {"batch_size": 8, "img_size": 224},
                "model": {"classes": ["glioma", "meningioma", "pituitary", "no_tumor"]},
                "training": {"use_weighted_sampler": True},
            }
            train_loader, val_loader, test_loader, loaded_classes = get_data_loaders(
                data_dir=str(root),
                config=config,
            )

            self.assertEqual(loaded_classes, ["glioma", "meningioma", "pituitary", "no_tumor"])
            self.assertGreater(len(train_loader.dataset), 0)
            self.assertGreater(len(val_loader.dataset), 0)
            self.assertGreater(len(test_loader.dataset), 0)

    def test_metric_computation(self) -> None:
        """Compute metrics from dummy predictions and probabilities."""
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        y_pred = np.array([0, 1, 2, 2, 0, 1, 3, 3], dtype=np.int64)
        y_scores = np.array(
            [
                [0.95, 0.02, 0.02, 0.01],
                [0.01, 0.95, 0.02, 0.02],
                [0.01, 0.02, 0.95, 0.02],
                [0.05, 0.10, 0.70, 0.15],
                [0.90, 0.04, 0.03, 0.03],
                [0.05, 0.85, 0.05, 0.05],
                [0.05, 0.05, 0.10, 0.80],
                [0.03, 0.02, 0.05, 0.90],
            ],
            dtype=np.float32,
        )
        class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]

        metrics = compute_metrics(y_true, y_pred, y_scores, class_names)
        expected_keys = {
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "specificity",
            "auc_roc",
            "classification_report",
            "confusion_matrix",
        }
        self.assertTrue(expected_keys.issubset(metrics.keys()))
        self.assertEqual(metrics["confusion_matrix"].shape, (4, 4))
        self.assertIsInstance(metrics["classification_report"], str)


if __name__ == "__main__":
    unittest.main()
