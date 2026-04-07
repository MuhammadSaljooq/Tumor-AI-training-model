# Brain Tumor Classifier - Complete Project Documentation

## 1. Project Overview

This project implements an end-to-end MRI brain tumor classification pipeline using deep learning.  
It supports three model families under a single training and evaluation framework:

- `resnet50`
- `vit`
- `hybrid` (ResNet + ViT feature fusion)

The pipeline is designed to be reproducible, configurable, and automation-friendly for GitHub users.

---

## 2. Core Objectives

- Train and evaluate multiple deep learning architectures under a common workflow.
- Generate consistent artifacts: checkpoints, logs, plots, and tabular summaries.
- Support practical experimentation (quick 2-epoch runs, evaluate-only runs, multi-model runs).
- Improve robustness with class balancing techniques and standardized preprocessing.

---

## 3. High-Level Pipeline

1. Load YAML config (`configs/config.yaml`).
2. Build dataset loaders from `data/processed`.
3. Initialize selected model(s).
4. Train model(s) with:
   - weighted loss (optional)
   - weighted sampler (optional)
   - early stopping
   - checkpointing
5. Evaluate on test split.
6. Save artifacts in `results/`.

Entry point: `main.py`

---

## 4. Project Structure

```text
brain_tumor_classifier/
├── configs/
│   ├── config.yaml
│   └── config_*_2epochs.yaml
├── data/
│   ├── raw/
│   └── processed/
│       ├── Training/
│       └── Testing/
├── docs/
│   └── PROJECT_DOCUMENTATION.md
├── results/
│   ├── checkpoints/
│   ├── logs/
│   └── plots/
├── src/
│   ├── evaluation/
│   ├── models/
│   ├── preprocessing/
│   ├── training/
│   └── utils/
├── tests/
│   └── test_pipeline.py
├── main.py
├── quick_start.py
├── setup_and_run.sh
└── README.md
```

---

## 5. Dataset Format

Preferred processed layout:

```text
data/processed/
  Training/
    glioma/
    meningioma/
    pituitary/
    no_tumor/   (or notumor; normalized internally)
  Testing/
    glioma/
    meningioma/
    pituitary/
    no_tumor/
```

### Class Name Normalization

The loader canonicalizes class names and handles alias mapping:

- `notumor` -> `no_tumor`

This prevents accidental metric inconsistency due to folder naming differences.

---

## 6. Model Details

### 6.1 ResNet50

- Backbone: ImageNet-pretrained ResNet50
- Use case: strong local-feature extraction and transfer-learning baseline

### 6.2 Vision Transformer (ViT)

- Backbone: `vit_base_patch16_224` (via `timm`)
- Use case: global context modeling through patch-based self-attention

### 6.3 Hybrid

- Combines CNN and transformer features
- Use case: leverage local + global representations together

---

## 7. Preprocessing and Augmentation

Module: `src/preprocessing/augmentation.py`

### Train-time transforms (moderate)

- Horizontal flip
- Mild affine transform (small rotate/scale/translate)
- Mild brightness/contrast jitter
- Resize to target size
- ImageNet normalization
- Tensor conversion

### Validation/Test transforms

- Resize
- ImageNet normalization
- Tensor conversion

ImageNet stats used:

- mean: `[0.485, 0.456, 0.406]`
- std: `[0.229, 0.224, 0.225]`

---

## 8. Training Strategy

Module: `src/training/trainer.py`

- Optimizer: `AdamW`
- Scheduler: `CosineAnnealingLR`
- Loss: `CrossEntropyLoss`
  - optional class weights from train-label frequencies
- Early stopping: monitored on validation loss
- Checkpointing: save best model by validation loss
- Mixed precision: enabled on CUDA environments

### Class Imbalance Handling

Implemented in two ways (configurable):

1. `use_class_weights: true` -> class-weighted cross entropy
2. `use_weighted_sampler: true` -> `WeightedRandomSampler` for train loader

---

## 9. Configuration

Primary config: `configs/config.yaml`

Important fields:

- `data.raw_dir`
- `data.processed_dir`
- `data.img_size`
- `data.batch_size`
- `model.num_classes`
- `model.classes`
- `training.epochs`
- `training.lr`
- `training.weight_decay`
- `training.patience`
- `training.use_class_weights`
- `training.use_weighted_sampler`
- `paths.checkpoints`
- `paths.logs`
- `paths.plots`

---

## 10. Running the Project

## 10.1 Manual setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 10.2 One-command automation (recommended)

```bash
bash setup_and_run.sh
```

### Useful script variants

```bash
# Train resnet50 for 2 epochs (temporary config override)
bash setup_and_run.sh --models resnet50 --epochs 2 --train

# Evaluate only using existing checkpoint(s)
bash setup_and_run.sh --models resnet50 --evaluate-only

# Run multiple models
bash setup_and_run.sh --models "resnet50 vit"
```

## 10.3 Direct `main.py` command

```bash
python3 -u main.py --config configs/config.yaml --data_dir data/processed --skip_preprocessing --models resnet50
```

---

## 11. Evaluation and Metrics

Module: `src/evaluation/metrics.py`

Metrics computed:

- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- Specificity (macro)
- AUC-ROC (OVR macro)
- Confusion matrix
- Classification report (text + dict)

### Key Output Logs

In `results/logs/<model>.log`, the final summary line format:

```text
Test Results | accuracy=..., precision=..., recall=..., f1_score=..., specificity=..., auc_roc=...
```

---

## 12. Generated Artifacts

### Per model

- `results/checkpoints/<model>_best.pth`
- `results/plots/<model>_confusion_matrix.png`
- `results/plots/<model>_training_history.png` (training mode only)
- `results/plots/<model>_classification_report.txt`
- `results/plots/<model>_classification_report.json`
- `results/logs/<model>.log`

### Cross-model

- `results/plots/model_comparison.png`
- `results/plots/roc_all_models.png`
- `results/final_results.csv`

---

## 13. About Metric Mismatches

If you see different metric values across files/charts, usually the evaluation set is different.

Common cases:

- **Official run metrics** from `main.py` test loader (log line `Test Results | ...`)
- **Manual folder evaluation** on `data/processed/Testing/...` done by ad-hoc scripts

To compare fairly, always use the same evaluation method and split definition.

---

## 14. Testing

Run integration-style tests:

```bash
PYTHONPATH=. python3 tests/test_pipeline.py
```

Coverage includes:

- preprocessing helpers
- model forward shapes
- dataloader behavior
- metric computation
- explicit `Training/Testing` layout handling

---

## 15. Troubleshooting

### `ModuleNotFoundError: src`

Use:

```bash
PYTHONPATH=. python3 ...
```

### Slow training

- CPU training is significantly slower than GPU.
- Use lower epochs for smoke tests.

### `urllib3` OpenSSL warning

This is typically non-blocking for training/evaluation runs.

### Empty/incorrect metrics

- Verify folder structure in `data/processed`
- Verify class names and aliases
- Verify you are reading metrics from the intended evaluation split

---

## 16. Reproducibility Checklist

- Use the same `config.yaml`
- Keep dataset layout fixed
- Use the same model checkpoint type (`*_best.pth`)
- Compare metrics from the same evaluation workflow
- Track commit hash when reporting results

---

## 17. Suggested Next Improvements

- Add per-class ROC and PR curves as separate plots
- Add deterministic seed controls in config and CLI
- Add model export options (TorchScript/ONNX)
- Add CI workflow for tests + lint + docs checks

