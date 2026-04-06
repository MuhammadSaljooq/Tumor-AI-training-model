"""ResNet50 classifier model."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50Classifier(nn.Module):
    """ResNet50-based classifier with a custom classification head."""

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = resnet50(weights=weights)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

            for layer in (self.model.layer3, self.model.layer4, self.model.fc):
                for param in layer.parameters():
                    param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_feature_extractor(self) -> nn.Module:
        """Return ResNet50 backbone without the final FC layer."""
        modules = list(self.model.children())[:-1]
        return nn.Sequential(*modules)
