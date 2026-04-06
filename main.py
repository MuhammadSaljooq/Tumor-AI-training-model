"""Main entry point for brain tumor classification pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from src.evaluation import evaluate_model, print_results_table
from src.models import get_model
from src.preprocessing import PreprocessingPipeline
from src.training import ModelTrainer
from src.utils import (
    ExperimentLogger,
    get_data_loaders,
    plot_comparison_bar,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_history,
)


def load_config(config_path: str) -> dict:
    """Load YAML configuration from disk."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def setup_directories(config: dict) -> None:
    """Create all output directories declared in config."""
    for _, dir_path in config.get("paths", {}).items():
        os.makedirs(dir_path, exist_ok=True)


def train_and_evaluate_model(
    model_name,
    train_loader,
    val_loader,
    test_loader,
    config,
    device,
    class_names,
) -> dict:
    """Train/evaluate one model and generate per-model artifacts."""
    num_classes = int(config["model"]["num_classes"])
    model = get_model(model_name, num_classes=num_classes, pretrained=True).to(device)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"{model_name} parameters: total={total_params:,}, trainable={trainable_params:,}")

    paths_cfg = config["paths"]
    checkpoint_path = Path(paths_cfg["checkpoints"]) / f"{model_name}_best.pth"

    model_logger = ExperimentLogger(log_dir=paths_cfg["logs"], model_name=model_name)
    history = {
        "train_losses": [],
        "val_losses": [],
        "train_accs": [],
        "val_accs": [],
    }

    if config.get("evaluate_only", False):
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for evaluate_only mode: {checkpoint_path}"
            )
        print(f"Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
    else:
        trainer_config = dict(config)
        trainer_config["model_name"] = model_name
        trainer = ModelTrainer(model=model, config=trainer_config, device=device, logger=model_logger)
        history = trainer.train(train_loader=train_loader, val_loader=val_loader)

    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
    )
    model_logger.log_test_results(
        {
            "accuracy": metrics.get("accuracy", float("nan")),
            "precision": metrics.get("precision", float("nan")),
            "recall": metrics.get("recall", float("nan")),
            "f1_score": metrics.get("f1_score", float("nan")),
            "specificity": metrics.get("specificity", float("nan")),
            "auc_roc": metrics.get("auc_roc", float("nan")),
        }
    )

    # Recompute predictions for confusion plotting from test pass outputs.
    # Using evaluate_model internals keeps metrics authoritative.
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            probs = torch.softmax(model(images), dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        model_name=model_name,
        save_path=Path(paths_cfg["plots"]) / f"{model_name}_confusion_matrix.png",
    )
    report_dir = Path(paths_cfg["plots"])
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{model_name}_classification_report.txt").write_text(
        metrics.get("classification_report", ""),
        encoding="utf-8",
    )
    (report_dir / f"{model_name}_classification_report.json").write_text(
        json.dumps(metrics.get("classification_report_dict", {}), indent=2),
        encoding="utf-8",
    )

    if not config.get("evaluate_only", False) and len(history["train_losses"]) > 0:
        plot_training_history(
            train_losses=history["train_losses"],
            val_losses=history["val_losses"],
            train_accs=history["train_accs"],
            val_accs=history["val_accs"],
            model_name=model_name,
            save_path=Path(paths_cfg["plots"]) / f"{model_name}_training_history.png",
        )

    # Store arrays needed for global ROC plotting.
    y_scores = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device, non_blocking=True)
            probs = torch.softmax(model(images), dim=1)
            y_scores.extend(probs.cpu().numpy())
    metrics["y_true"] = torch.tensor(y_true).numpy()
    metrics["y_scores"] = torch.tensor(y_scores).numpy()

    return {
        "model": model,
        "history": history,
        "metrics": metrics,
    }


def main():
    """Parse CLI args and run full training/evaluation workflow."""
    parser = argparse.ArgumentParser(description="Brain Tumor Classification")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet50", "vit", "hybrid"],
        choices=["resnet50", "vit", "hybrid"],
    )
    parser.add_argument("--data_dir", default=None, help="Override data directory from config")
    parser.add_argument("--skip_preprocessing", action="store_true")
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        help="Skip training, load checkpoints and evaluate",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_dir:
        config["data"]["raw_dir"] = args.data_dir
    config["evaluate_only"] = args.evaluate_only

    setup_directories(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_cfg = config["data"]
    raw_dir = data_cfg["raw_dir"]
    processed_dir = data_cfg["processed_dir"]

    if not args.skip_preprocessing:
        print(f"Running preprocessing: {raw_dir} -> {processed_dir}")
        preprocessing = PreprocessingPipeline(config=data_cfg)
        preprocessing.process_dataset(raw_dir=raw_dir, output_dir=processed_dir)

    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        data_dir=processed_dir,
        config=config,
    )

    all_results: dict[str, dict] = {}
    roc_scores: dict[str, torch.Tensor] = {}
    roc_y_true = None

    for model_name in args.models:
        print("=" * 50 + f"\nTraining {model_name}\n" + "=" * 50)
        model_output = train_and_evaluate_model(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=device,
            class_names=class_names,
        )

        metrics = model_output["metrics"]
        all_results[model_name] = metrics
        roc_scores[model_name] = metrics["y_scores"]
        if roc_y_true is None:
            roc_y_true = metrics["y_true"]

    # Metric bar comparison for all models.
    metric_names = ["accuracy", "precision", "recall", "specificity", "f1_score", "auc_roc"]
    compact_results = {
        name: {metric: values.get(metric, float("nan")) for metric in metric_names}
        for name, values in all_results.items()
    }
    plot_comparison_bar(
        results_dict=compact_results,
        metric_names=metric_names,
        save_path=Path(config["paths"]["plots"]) / "model_comparison.png",
    )

    # One ROC plot with all selected models.
    if roc_y_true is not None:
        plot_roc_curves(
            y_true=roc_y_true,
            y_scores=roc_scores,
            class_names=class_names,
            model_names=args.models,
            save_path=Path(config["paths"]["plots"]) / "roc_all_models.png",
        )

    print_results_table(compact_results)

    final_logger = ExperimentLogger(log_dir=config["paths"]["logs"], model_name="final_results")
    final_logger.save_results_csv(
        results_dict=compact_results,
        save_path=Path("results") / "final_results.csv",
    )


if __name__ == "__main__":
    main()
