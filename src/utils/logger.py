"""Experiment logging utilities."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


class ExperimentLogger:
    """Logs experiment metrics to console/file and optionally TensorBoard."""

    def __init__(self, log_dir: str, model_name: str) -> None:
        self.log_dir = Path(log_dir)
        self.model_name = model_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"brain_tumor_classifier.{model_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            file_handler = logging.FileHandler(self.log_dir / f"{model_name}.log")
            stream_handler = logging.StreamHandler()
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)

        self.writer = None
        if SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard" / model_name))

    def log_metrics(self, epoch, train_loss, val_loss, train_acc, val_acc) -> None:
        """Log per-epoch training and validation metrics."""
        self.logger.info(
            "Epoch %s | train_loss=%.4f val_loss=%.4f train_acc=%.4f val_acc=%.4f",
            epoch,
            train_loss,
            val_loss,
            train_acc,
            val_acc,
        )
        if self.writer is not None:
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Validation", val_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/Validation", val_acc, epoch)

    def log_test_results(self, metrics_dict) -> None:
        """Log test/evaluation metrics."""
        metrics_text = ", ".join(f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in metrics_dict.items())
        self.logger.info("Test Results | %s", metrics_text)

        if self.writer is not None:
            for metric_name, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"Test/{metric_name}", value)

    def save_results_csv(self, results_dict, save_path) -> None:
        """Save all model results into a CSV file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        metric_names: set[str] = set()
        for model_metrics in results_dict.values():
            metric_names.update(model_metrics.keys())

        ordered_metrics = sorted(metric_names)
        with save_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["model_name", *ordered_metrics])
            for model_name, model_metrics in results_dict.items():
                row = [model_name, *[model_metrics.get(metric, "") for metric in ordered_metrics]]
                writer.writerow(row)

        self.logger.info("Saved comparison CSV to %s", save_path)
