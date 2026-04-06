"""Evaluation metrics and reporting utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


def compute_specificity(y_true, y_pred, num_classes) -> float:
    """Compute macro-average specificity across classes."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities: list[float] = []

    for class_idx in range(num_classes):
        tp = cm[class_idx, class_idx]
        fp = cm[:, class_idx].sum() - tp
        fn = cm[class_idx, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        denom = tn + fp
        specificities.append(float(tn / denom) if denom > 0 else 0.0)

    return float(np.mean(specificities))


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray, class_names: list
) -> dict:
    """Compute full classification metrics for multi-class evaluation."""
    num_classes = len(class_names)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "specificity": compute_specificity(y_true, y_pred, num_classes),
        "classification_report_dict": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))),
    }

    try:
        metrics["auc_roc"] = roc_auc_score(
            y_true,
            y_scores,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        metrics["auc_roc"] = float("nan")

    return metrics


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str, class_names: list) -> dict:
    """Run model over test set and return computed metrics."""
    model = model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    y_scores: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_scores.extend(probs.cpu().numpy())

    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    y_scores_np = np.asarray(y_scores)

    return compute_metrics(y_true_np, y_pred_np, y_scores_np, class_names)


def print_results_table(all_results: dict) -> None:
    """Print model comparison table for key metrics."""
    headers = ["Model", "Accuracy", "Precision", "Recall", "Specificity", "F1", "AUC-ROC"]
    rows = []
    for model_name, metrics_dict in all_results.items():
        rows.append(
            [
                model_name,
                f"{metrics_dict.get('accuracy', float('nan')):.4f}",
                f"{metrics_dict.get('precision', float('nan')):.4f}",
                f"{metrics_dict.get('recall', float('nan')):.4f}",
                f"{metrics_dict.get('specificity', float('nan')):.4f}",
                f"{metrics_dict.get('f1_score', float('nan')):.4f}",
                f"{metrics_dict.get('auc_roc', float('nan')):.4f}",
            ]
        )

    try:
        from tabulate import tabulate

        print(tabulate(rows, headers=headers, tablefmt="grid"))
    except ImportError:
        widths = [max(len(str(item)) for item in col) for col in zip(headers, *rows)]
        line = "+".join("-" * (w + 2) for w in widths)
        print(line)
        print(" | ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers)))
        print(line)
        for row in rows:
            print(" | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)))
        print(line)
