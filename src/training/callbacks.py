"""Training callbacks."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class EarlyStopping:
    """Early stopping callback."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = "min") -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def __call__(self, metric: float) -> bool:
        """Return True when training should stop."""
        if self.mode == "min":
            improved = metric < (self.best_metric - self.min_delta)
        else:
            improved = metric > (self.best_metric + self.min_delta)

        if improved:
            self.best_metric = metric
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


class ModelCheckpoint:
    """Save and load best model checkpoint."""

    def __init__(self, save_dir: str, model_name: str, mode: str = "min") -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.mode = mode
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.save_dir / f"{self.model_name}_best.pth"
        self.best_metric = float("inf") if mode == "min" else float("-inf")

    def __call__(self, metric: float, model: nn.Module, epoch: int) -> bool:
        """Save model if metric improved. Returns True if saved."""
        if self.mode == "min":
            improved = metric < self.best_metric
        else:
            improved = metric > self.best_metric

        if not improved:
            return False

        self.best_metric = metric
        torch.save(
            {
                "epoch": epoch,
                "metric": metric,
                "model_state_dict": model.state_dict(),
            },
            self.checkpoint_path,
        )
        return True

    def load_best(self, model: nn.Module) -> nn.Module:
        """Load best checkpoint weights into model and return it."""
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        return model
