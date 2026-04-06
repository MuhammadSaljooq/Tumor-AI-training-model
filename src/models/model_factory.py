"""Factory utilities for model creation."""

from __future__ import annotations

import torch.nn as nn

from .hybrid import HybridResNetViT
from .resnet50 import ResNet50Classifier
from .vit import ViTClassifier


def get_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Return a model instance by name."""
    key = model_name.lower()
    if key == "resnet50":
        return ResNet50Classifier(num_classes=num_classes, pretrained=pretrained)
    if key == "vit":
        return ViTClassifier(num_classes=num_classes, pretrained=pretrained)
    if key == "hybrid":
        return HybridResNetViT(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model name '{model_name}'. Expected one of: resnet50, vit, hybrid.")
