"""Model training loop implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ..utils import ExperimentLogger

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

    def train_one_epoch(self, train_loader, optimizer, criterion, scaler) -> tuple[float, float]:
        """Run one training epoch with mixed precision and tqdm progress bar."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(train_loader, desc="Train", leave=False)
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
            progress = tqdm(val_loader, desc="Val", leave=False)
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

    def train(self, train_loader, val_loader) -> dict:
        """Run full training with scheduler, early stopping, checkpointing, and logging."""
        training_cfg = self.config.get("training", {})
        epochs = int(training_cfg.get("epochs", 50))
        lr = float(training_cfg.get("lr", 1e-4))
        weight_decay = float(training_cfg.get("weight_decay", 1e-4))
        patience = int(training_cfg.get("patience", 10))

        model_name = self.config.get("model_name", getattr(self.logger, "model_name", "model"))
        checkpoint_dir = self.config.get("paths", {}).get("checkpoints", "results/checkpoints")

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        use_class_weights = bool(training_cfg.get("use_class_weights", True))
        criterion = nn.CrossEntropyLoss()
        if use_class_weights and hasattr(train_loader.dataset, "labels"):
            labels = np.asarray(train_loader.dataset.labels, dtype=np.int64)
            if labels.size > 0:
                num_classes = int(labels.max()) + 1
                counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
                counts = np.clip(counts, a_min=1.0, a_max=None)
                weights = counts.sum() / (num_classes * counts)
                class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        early_stopping = EarlyStopping(patience=patience, min_delta=0.001, mode="min")
        checkpoint = ModelCheckpoint(save_dir=checkpoint_dir, model_name=model_name, mode="min")

        history = {
            "train_losses": [],
            "val_losses": [],
            "train_accs": [],
            "val_accs": [],
        }

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_one_epoch(train_loader, optimizer, criterion, scaler)
            val_loss, val_acc = self.validate(val_loader, criterion)
            scheduler.step()

            history["train_losses"].append(train_loss)
            history["val_losses"].append(val_loss)
            history["train_accs"].append(train_acc)
            history["val_accs"].append(val_acc)

            checkpoint.__call__(metric=val_loss, model=self.model, epoch=epoch)
            self.logger.log_metrics(epoch, train_loss, val_loss, train_acc, val_acc)

            if early_stopping(val_loss):
                self.logger.logger.info("Early stopping triggered at epoch %s.", epoch)
                break

        return history

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load saved model weights."""
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
