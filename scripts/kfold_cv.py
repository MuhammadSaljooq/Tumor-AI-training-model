#!/usr/bin/env python3
"""Run stratified K-fold cross-validation for brain tumor classifiers."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import evaluate_model, print_results_table
from src.models import get_model
from src.preprocessing import get_train_transforms, get_val_transforms
from src.training import ModelTrainer
from src.utils import BrainTumorDataset, ExperimentLogger

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_ALIASES = {"notumor": "no_tumor", "no_tumor": "no_tumor"}


def canonical(name: str) -> str:
    return CLASS_ALIASES.get(name.strip().lower().replace("-", "_").replace(" ", "_"), name.strip().lower())


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_num_workers(config: dict) -> int:
    requested = int(config.get("data", {}).get("num_workers", 4))
    if torch.cuda.is_available():
        return requested
    return 0 if os.uname().sysname.lower() == "darwin" else requested


def collect_image_paths(data_dir: Path, class_names: list[str]) -> tuple[list[str], list[int]]:
    """Collect image paths and encoded labels from either split or flat layout."""
    image_paths: list[str] = []
    labels: list[int] = []

    candidate_roots = []
    training_dir = data_dir / "Training"
    testing_dir = data_dir / "Testing"
    if training_dir.is_dir() and testing_dir.is_dir():
        candidate_roots.extend([training_dir, testing_dir])
    else:
        candidate_roots.append(data_dir)

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for root in candidate_roots:
        for folder in root.iterdir():
            if not folder.is_dir():
                continue
            class_name = canonical(folder.name)
            if class_name not in class_to_idx:
                continue
            label_idx = class_to_idx[class_name]
            for p in folder.rglob("*"):
                if p.is_file() and p.suffix.lower() in VALID_EXTS:
                    image_paths.append(str(p))
                    labels.append(label_idx)

    if not image_paths:
        raise ValueError(f"No images found under '{data_dir}'.")
    return image_paths, labels


def make_fold_loaders(
    image_paths: list[str],
    labels: list[int],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: dict,
) -> tuple[DataLoader, DataLoader]:
    data_cfg = config.get("data", {})
    img_size = int(data_cfg.get("img_size", 224))
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = resolve_num_workers(config)
    pin_memory = torch.cuda.is_available()

    train_paths = [image_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_ds = BrainTumorDataset(train_paths, train_labels, transform=get_train_transforms(img_size))
    val_ds = BrainTumorDataset(val_paths, val_labels, transform=get_val_transforms(img_size))

    sampler = None
    if bool(config.get("training", {}).get("use_weighted_sampler", True)) and train_labels:
        counts = np.bincount(np.asarray(train_labels), minlength=max(train_labels) + 1)
        counts = np.clip(counts, a_min=1, a_max=None)
        class_weights = 1.0 / counts
        sample_weights = class_weights[np.asarray(train_labels)]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_labels),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def summarize_metrics(fold_metrics: list[dict], keys: list[str]) -> dict:
    summary = {}
    for key in keys:
        values = [float(m.get(key, float("nan"))) for m in fold_metrics]
        arr = np.asarray(values, dtype=np.float64)
        summary[f"{key}_mean"] = float(np.nanmean(arr))
        summary[f"{key}_std"] = float(np.nanstd(arr))
    return summary


def run_kfold_for_model(
    model_name: str,
    config: dict,
    image_paths: list[str],
    labels: list[int],
    class_names: list[str],
    k_folds: int,
    device: str,
) -> dict:
    splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_metrics: list[dict] = []
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "specificity", "auc_roc"]

    for fold, (train_idx, val_idx) in enumerate(splitter.split(image_paths, labels), start=1):
        print(f"\n{'=' * 55}\n{model_name} | Fold {fold}/{k_folds}\n{'=' * 55}")
        train_loader, val_loader = make_fold_loaders(image_paths, labels, train_idx, val_idx, config)

        run_cfg = copy.deepcopy(config)
        run_cfg["model_name"] = f"{model_name}_fold{fold}"
        ckpt_root = Path(config.get("paths", {}).get("checkpoints", "results/checkpoints")) / "kfold"
        log_root = Path(config.get("paths", {}).get("logs", "results/logs")) / "kfold"
        run_cfg.setdefault("paths", {})
        run_cfg["paths"]["checkpoints"] = str(ckpt_root)
        run_cfg["paths"]["logs"] = str(log_root)
        ckpt_root.mkdir(parents=True, exist_ok=True)
        log_root.mkdir(parents=True, exist_ok=True)

        model = get_model(model_name, num_classes=len(class_names), pretrained=True).to(device)
        logger = ExperimentLogger(log_dir=str(log_root), model_name=f"{model_name}_fold{fold}")
        trainer = ModelTrainer(model=model, config=run_cfg, device=device, logger=logger)
        trainer.train(train_loader, val_loader)

        checkpoint_path = ckpt_root / f"{model_name}_fold{fold}_best.pth"
        if checkpoint_path.exists():
            trainer.load_checkpoint(str(checkpoint_path))

        metrics = evaluate_model(model=model, test_loader=val_loader, device=device, class_names=class_names)
        logger.log_test_results({k: metrics.get(k) for k in metric_keys})

        clean_metrics = {k: float(metrics.get(k, float("nan"))) for k in metric_keys}
        clean_metrics["fold"] = fold
        fold_metrics.append(clean_metrics)

    mean_std = summarize_metrics(fold_metrics, metric_keys)
    return {"model": model_name, "folds": fold_metrics, "summary": mean_std}


def main() -> None:
    parser = argparse.ArgumentParser(description="K-fold cross-validation for brain tumor models.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--models", nargs="+", default=["resnet50"], choices=["resnet50", "vit", "hybrid"])
    parser.add_argument("--k_folds", type=int, default=5)
    args = parser.parse_args()

    config_path = Path(args.config)
    data_dir = Path(args.data_dir)
    config = load_config(config_path)
    class_names = [canonical(c) for c in config.get("model", {}).get("classes", [])]
    if not class_names:
        raise SystemExit("No model.classes found in config.")

    image_paths, labels = collect_image_paths(data_dir, class_names)
    print(f"Found {len(image_paths)} images across {len(class_names)} classes.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    all_results = {}
    compact_results = {}
    for model_name in args.models:
        result = run_kfold_for_model(
            model_name=model_name,
            config=config,
            image_paths=image_paths,
            labels=labels,
            class_names=class_names,
            k_folds=args.k_folds,
            device=device,
        )
        all_results[model_name] = result
        compact_results[model_name] = {
            "accuracy": result["summary"]["accuracy_mean"],
            "precision": result["summary"]["precision_mean"],
            "recall": result["summary"]["recall_mean"],
            "specificity": result["summary"]["specificity_mean"],
            "f1_score": result["summary"]["f1_score_mean"],
            "auc_roc": result["summary"]["auc_roc_mean"],
        }

    print("\nK-fold mean metrics")
    print_results_table(compact_results)

    output_dir = Path("results") / "kfold"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"kfold_{args.k_folds}_results.json"
    csv_path = output_dir / f"kfold_{args.k_folds}_summary.csv"
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8") as f:
        headers = [
            "model",
            "accuracy_mean",
            "accuracy_std",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
            "f1_score_mean",
            "f1_score_std",
            "specificity_mean",
            "specificity_std",
            "auc_roc_mean",
            "auc_roc_std",
        ]
        f.write(",".join(headers) + "\n")
        for model_name, payload in all_results.items():
            s = payload["summary"]
            row = [
                model_name,
                s["accuracy_mean"],
                s["accuracy_std"],
                s["precision_mean"],
                s["precision_std"],
                s["recall_mean"],
                s["recall_std"],
                s["f1_score_mean"],
                s["f1_score_std"],
                s["specificity_mean"],
                s["specificity_std"],
                s["auc_roc_mean"],
                s["auc_roc_std"],
            ]
            f.write(",".join(str(x) for x in row) + "\n")

    print(f"\nSaved K-fold outputs:\n- {json_path}\n- {csv_path}")


if __name__ == "__main__":
    main()
