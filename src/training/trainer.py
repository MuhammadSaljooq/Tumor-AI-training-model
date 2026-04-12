"""Model training loop implementation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ..utils import ExperimentLogger
from ..utils.checkpoint_io import load_checkpoint
from ..utils.data_loader import _read_img_size

from .callbacks import EarlyStopping, ModelCheckpoint


class ModelTrainer:
    """High-level trainer for model training/validation."""

    def __init__(self, model, config: dict, device: str, logger: ExperimentLogger) -> None:
        """Initialize trainer with model, config, runtime device, and logger."""
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = logger
        self.use_amp = device.startswith("cuda") and torch.cuda.is_available()

    def _resolve_num_classes_for_loss(self) -> int:
        """Match CrossEntropyLoss weight length to model logits (never infer from labels alone)."""
        model_cfg = self.config.get("model") or {}
        if model_cfg.get("num_classes") is not None:
            return int(model_cfg["num_classes"])
        classes = model_cfg.get("classes")
        if isinstance(classes, list) and len(classes) > 0:
            return len(classes)
        img_size = _read_img_size(self.config)
        dummy = torch.zeros(1, 3, img_size, img_size, device=self.device, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(dummy)
        self.model.train()
        n = int(logits.shape[1])
        self.logger.logger.warning(
            "Config missing model.num_classes and model.classes; inferred num_classes=%s from a dummy forward pass.",
            n,
        )
        return n

    def train_one_epoch(self, train_loader, optimizer, criterion, scaler) -> tuple[float, float]:
        """Run one training epoch with mixed precision and tqdm progress bar."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(
            train_loader,
            desc="Train",
            leave=False,
            file=sys.stdout,
            dynamic_ncols=True,
            mininterval=0.2,
        )
        for images, labels in progress:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / max(total, 1)
        avg_accuracy = correct / max(total, 1)
        return avg_loss, avg_accuracy

    def validate(self, val_loader, criterion) -> tuple[float, float]:
        """Run validation loop under torch.no_grad()."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            progress = tqdm(
                val_loader,
                desc="Val",
                leave=False,
                file=sys.stdout,
                dynamic_ncols=True,
                mininterval=0.2,
            )
            for images, labels in progress:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += batch_size

                progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / max(total, 1)
        avg_accuracy = correct / max(total, 1)
        return avg_loss, avg_accuracy

    def train(
        self,
        train_loader,
        val_loader,
        resume_path: str | None = None,
    ) -> dict:
        """Run full training with scheduler, early stopping, checkpointing, and logging."""
        training_cfg = self.config.get("training", {})
        epochs = int(training_cfg.get("epochs", 50))
        lr = float(training_cfg.get("lr", 1e-4))
        weight_decay = float(training_cfg.get("weight_decay", 1e-4))
        patience = int(training_cfg.get("patience", 10))
        min_delta = float(training_cfg.get("min_delta", 0.001))
        save_last_checkpoint = bool(training_cfg.get("save_last_checkpoint", False))

        model_name = self.config.get("model_name", getattr(self.logger, "model_name", "model"))
        checkpoint_dir = self.config.get("paths", {}).get("checkpoints", "results/checkpoints")

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        use_class_weights = bool(training_cfg.get("use_class_weights", True))
        criterion = nn.CrossEntropyLoss()
        if use_class_weights and hasattr(train_loader.dataset, "labels"):
            labels = np.asarray(train_loader.dataset.labels, dtype=np.int64)
            if labels.size > 0:
                num_classes = self._resolve_num_classes_for_loss()
                if labels.max() >= num_classes or labels.min() < 0:
                    raise ValueError(
                        f"Training labels out of range for num_classes={num_classes}: "
                        f"min={labels.min()}, max={labels.max()}. Check data layout and model.classes order."
                    )
                counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
                counts = np.clip(counts, a_min=1.0, a_max=None)
                weights = counts.sum() / (num_classes * counts)
                class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
                if class_weights.numel() != num_classes:
                    raise RuntimeError(
                        f"Internal error: class weight length {class_weights.numel()} != num_classes {num_classes}"
                    )
                criterion = nn.CrossEntropyLoss(weight=class_weights)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode="min")
        checkpoint = ModelCheckpoint(
            save_dir=checkpoint_dir,
            model_name=model_name,
            mode="min",
            min_delta=min_delta,
            metric_name="val_loss",
        )

        training_snapshot = {"lr": lr, "weight_decay": weight_decay, "epochs": epochs}
        start_epoch = 1

        if resume_path:
            ckpt_path = Path(resume_path)
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
            ckpt = load_checkpoint(ckpt_path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict):
                ckpt_model = ckpt.get("model_name")
                if ckpt_model is not None and ckpt_model != model_name:
                    self.logger.logger.warning(
                        "Checkpoint model_name=%s does not match current model_name=%s.",
                        ckpt_model,
                        model_name,
                    )
                if "model_state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                if "scaler_state_dict" in ckpt and self.use_amp:
                    scaler.load_state_dict(ckpt["scaler_state_dict"])
                checkpoint.restore_callback_state(ckpt)
                last_done = int(ckpt.get("epoch", 0))
                start_epoch = last_done + 1
                self.logger.logger.info(
                    "Resumed from %s (completed epoch %s, starting epoch %s).",
                    ckpt_path,
                    last_done,
                    start_epoch,
                )
            else:
                self.model.load_state_dict(ckpt)

        history = {
            "train_losses": [],
            "val_losses": [],
            "train_accs": [],
            "val_accs": [],
        }

        if start_epoch > epochs:
            self.logger.logger.info(
                "Resume start_epoch %s exceeds training.epochs %s; skipping training loop.",
                start_epoch,
                epochs,
            )
            if epochs > 0 and checkpoint.checkpoint_path.is_file():
                self.load_checkpoint(str(checkpoint.checkpoint_path))
            return history

        for epoch in range(start_epoch, epochs + 1):
            train_loss, train_acc = self.train_one_epoch(train_loader, optimizer, criterion, scaler)
            val_loss, val_acc = self.validate(val_loader, criterion)
            scheduler.step()

            history["train_losses"].append(train_loss)
            history["val_losses"].append(val_loss)
            history["train_accs"].append(train_acc)
            history["val_accs"].append(val_acc)

            checkpoint.save_if_improved(
                val_loss,
                epoch,
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                use_amp=self.use_amp,
                training_snapshot=training_snapshot,
            )
            if save_last_checkpoint:
                checkpoint.save_last(
                    val_loss,
                    epoch,
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    use_amp=self.use_amp,
                    training_snapshot=training_snapshot,
                )
            self.logger.log_metrics(epoch, train_loss, val_loss, train_acc, val_acc)

            if early_stopping(val_loss):
                self.logger.logger.info("Early stopping triggered at epoch %s.", epoch)
                break

        if epochs > 0 and checkpoint.checkpoint_path.is_file():
            self.load_checkpoint(str(checkpoint.checkpoint_path))

        return history

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load saved model weights."""
        checkpoint = load_checkpoint(Path(checkpoint_path), map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
