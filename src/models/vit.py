"""Vision Transformer classifier model."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class ViTClassifier(nn.Module):
    """ViT backbone with a custom classification head."""

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        model_name: str = "vit_base_patch16_224",
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def get_feature_extractor(self) -> nn.Module:
        """Return ViT backbone that outputs raw features."""
        return self.backbone
