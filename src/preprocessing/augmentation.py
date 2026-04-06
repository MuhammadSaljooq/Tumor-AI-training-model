"""Albumentations-based augmentation pipelines."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 224) -> A.Compose:
    """Build train-time augmentation transforms.

    Args:
        img_size: Target image size (height and width).

    Returns:
        Albumentations composed transform.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.03, 0.03),
                rotate=(-10, 10),
                p=0.4,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.3,
            ),
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def get_val_transforms(img_size: int = 224) -> A.Compose:
    """Build validation/test transforms.

    Args:
        img_size: Target image size (height and width).

    Returns:
        Albumentations composed transform.
    """
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def build_augmentation_transform(img_size: int = 224) -> A.Compose:
    """Backward-compatible alias for train transforms."""
    return get_train_transforms(img_size=img_size)