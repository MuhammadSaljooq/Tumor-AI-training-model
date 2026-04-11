"""Training callbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..utils.checkpoint_io import load_checkpoint, save_checkpoint


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
    """Save best and optional last checkpoints with full training state."""

    def __init__(
        self,
        save_dir: str,
        model_name: str,
        mode: str = "min",
        min_delta: float = 0.0,
        metric_name: str = "val_loss",
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.mode = mode
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.save_dir / f"{self.model_name}_best.pth"
        self.last_checkpoint_path = self.save_dir / f"{self.model_name}_last.pth"
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.best_epoch: int = 0

    def _is_improvement(self, metric: float) -> bool:
        if self.mode == "min":
            return metric < (self.best_metric - self.min_delta)
        return metric > (self.best_metric + self.min_delta)

    def _build_payload(
        self,
        *,
        metric: float,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        scaler: torch.amp.GradScaler | None = None,
        use_amp: bool = False,
        training_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metric": metric,
            "best_metric": self.best_metric,
            "metric_at_save": metric,
            "metric_name": self.metric_name,
            "model_name": self.model_name,
            "best_epoch": self.best_epoch,
            "torch_version": torch.__version__,
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None and use_amp:
            payload["scaler_state_dict"] = scaler.state_dict()
        if training_snapshot:
            payload["training_snapshot"] = dict(training_snapshot)
        return payload

    def save_if_improved(
        self,
        metric: float,
        epoch: int,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        scaler: torch.amp.GradScaler | None = None,
        use_amp: bool = False,
        training_snapshot: dict[str, Any] | None = None,
    ) -> bool:
        """Persist best checkpoint if metric improved. Returns True if saved."""
        if not self._is_improvement(metric):
            return False

        self.best_metric = metric
        self.best_epoch = epoch
        payload = self._build_payload(
            metric=metric,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            use_amp=use_amp,
            training_snapshot=training_snapshot,
        )
        save_checkpoint(self.checkpoint_path, payload, atomic=True)
        return True

    def save_last(
        self,
        metric: float,
        epoch: int,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        scaler: torch.amp.GradScaler | None = None,
        use_amp: bool = False,
        training_snapshot: dict[str, Any] | None = None,
    ) -> None:
        """Overwrite last-epoch checkpoint (crash recovery)."""
        payload = self._build_payload(
            metric=metric,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            use_amp=use_amp,
            training_snapshot=training_snapshot,
        )
        save_checkpoint(self.last_checkpoint_path, payload, atomic=True)

    def load_best(self, model: nn.Module, map_location: str | torch.device | None = None) -> nn.Module:
        """Load best checkpoint weights into model and return it."""
        loc = map_location if map_location is not None else next(model.parameters()).device
        checkpoint = load_checkpoint(self.checkpoint_path, map_location=loc, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        return model

    def restore_callback_state(self, checkpoint: dict[str, Any]) -> None:
        """Restore best_metric and best_epoch from a loaded training checkpoint."""
        if "best_metric" in checkpoint:
            self.best_metric = float(checkpoint["best_metric"])
        elif "metric" in checkpoint:
            self.best_metric = float(checkpoint["metric"])
        self.best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0)))
