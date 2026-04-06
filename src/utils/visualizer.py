"""Visualization utilities for model evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_path) -> None:
    """Plot confusion matrix with seaborn heatmap and save PNG."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, format="png", dpi=300)
    plt.close()


def plot_roc_curves(y_true, y_scores, class_names, model_names, save_path) -> None:
    """Plot multi-class ROC curves for all models in one figure."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_true_np = np.asarray(y_true)
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true_np, classes=list(range(n_classes)))

    # Supports either {"model_name": scores} or list/tuple aligned with model_names.
    if isinstance(y_scores, dict):
        scores_by_model = {name: np.asarray(y_scores[name]) for name in model_names if name in y_scores}
    else:
        scores_by_model = {
            model_name: np.asarray(scores)
            for model_name, scores in zip(model_names, y_scores, strict=False)
        }

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(model_names))))
    plt.figure(figsize=(10, 8))

    for idx, model_name in enumerate(model_names):
        if model_name not in scores_by_model:
            continue
        model_scores = scores_by_model[model_name]

        fpr_grid = np.linspace(0.0, 1.0, 200)
        mean_tpr = np.zeros_like(fpr_grid)

        for class_idx in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], model_scores[:, class_idx])
            mean_tpr += np.interp(fpr_grid, fpr, tpr)

        mean_tpr /= n_classes
        model_auc = auc(fpr_grid, mean_tpr)
        plt.plot(
            fpr_grid,
            mean_tpr,
            color=colors[idx],
            linewidth=2,
            label=f"{model_name} (AUC={model_auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.title("Multi-model ROC Curves (Macro-average OVR)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, format="png", dpi=300)
    plt.close()


def plot_training_history(train_losses, val_losses, train_accs, val_accs, model_name, save_path) -> None:
    """Plot side-by-side loss and accuracy curves and save PNG."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, val_losses, label="Val Loss")
    axes[0].set_title(f"{model_name} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_accs, label="Train Accuracy")
    axes[1].plot(epochs, val_accs, label="Val Accuracy")
    axes[1].set_title(f"{model_name} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, format="png", dpi=300)
    plt.close(fig)


def plot_comparison_bar(results_dict, metric_names, save_path) -> None:
    """Plot grouped bar chart for model metric comparison and save PNG."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model_names = list(results_dict.keys())
    x = np.arange(len(metric_names))
    width = 0.8 / max(1, len(model_names))

    plt.figure(figsize=(12, 6))
    for idx, model_name in enumerate(model_names):
        model_metrics = results_dict.get(model_name, {})
        values = [model_metrics.get(metric, 0.0) for metric in metric_names]
        offset = (idx - (len(model_names) - 1) / 2) * width
        plt.bar(x + offset, values, width=width, label=model_name)

    plt.xticks(x, metric_names)
    plt.ylabel("Score")
    plt.title("Model Comparison Across Metrics")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, format="png", dpi=300)
    plt.close()
