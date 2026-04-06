"""Hybrid model combining ResNet50 and ViT backbones."""

from __future__ import annotations

import torch
import torch.nn as nn

from .resnet50 import ResNet50Classifier
from .vit import ViTClassifier


class HybridResNetViT(nn.Module):
    """Feature-level fusion of ResNet50 and ViT backbones."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True) -> None:
        super().__init__()

        self.resnet_backbone = ResNet50Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=False,
        ).get_feature_extractor()
        self.vit_backbone = ViTClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            model_name="vit_base_patch16_224",
        ).get_feature_extractor()

        self.resnet_features = 2048
        self.vit_features = 768
        self.combined_features = self.resnet_features + self.vit_features

        self.fusion_head = nn.Sequential(
            nn.Linear(self.combined_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resnet_features = self.resnet_backbone(x)
        if resnet_features.ndim == 4:
            resnet_features = torch.flatten(resnet_features, 1)

        vit_features = self.vit_backbone(x)
        if vit_features.ndim > 2:
            vit_features = vit_features[:, 0]

        combined = torch.cat([resnet_features, vit_features], dim=1)
        logits = self.fusion_head(combined)
        return logits
