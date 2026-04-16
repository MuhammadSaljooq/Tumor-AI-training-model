# Brain Tumor Classifier

Automated brain tumor classification project using MRI images and deep learning.
The pipeline is designed to compare three model families under a unified training and evaluation flow.

## Models Used

- **ResNet50**: Transfer learning with a CNN backbone pretrained on ImageNet.
- **Vision Transformer (ViT-B/16)**: Transformer-based vision model for patch-level representation learning.
- **Hybrid ResNet50 + ViT**: Feature-fusion architecture combining CNN texture features and transformer global context.

## Dataset

Use the Brain Tumor MRI Dataset from Kaggle:
[https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Expected classes:

- `glioma`
- `meningioma`
- `pituitary`
- `no_tumor`

Place the dataset in either layout:

**Flat (class folders at the root of `data/raw`):**

```text
data/raw/
  glioma/
  meningioma/
  pituitary/
  no_tumor/
```

**Kaggle-style (splits under `data/raw`):**

```text
data/raw/
  Training/
    glioma/
    meningioma/
    pituitary/
    notumor/
  Testing/
    ...
```

The folder name `notumor` is treated as `no_tumor`.

---

## Google Colab (one-click style)

1. Upload **[`notebooks/Colab_Tumor_AI_Training.ipynb`](notebooks/Colab_Tumor_AI_Training.ipynb)** to Colab (or open from GitHub with Colab if you enable that).
2. **Runtime → Change runtime type → GPU.**
3. Run the **single code cell** (after the intro markdown). Edit the variables at the top (**`EPOCHS`**, **`MODELS`**, **`RESULTS_DRIVE_DIR`**, hyperparameters, **`DATA_SOURCE`**, etc.), then execute.

**Full walkthrough** (how the cell works, checkpoints on Drive, resume, data modes, troubleshooting): **[`notebooks/README_COLAB.md`](notebooks/README_COLAB.md)**.

Short version: the cell mounts Drive, clones/updates this repo (branch **`checkpoint_added`**), merges your settings into **`configs/config_colab_runtime.yaml`**, prepares data, and runs **`main.py`**. Use **`RESULTS_DRIVE_DIR`** on Google Drive so checkpoints survive new Colab sessions; **`RESUME_IF_CHECKPOINT_EXISTS`** adds **`--resume`** when **`{model}_last.pth`** exists (single model only). **`SAVE_LAST_CHECKPOINT = True`** refreshes **`_last.pth` every epoch**.

---

## CLI quick start: setup, training, and checkpoints

Use these commands from the **repository root** (replace the `cd` path with your clone location).

### 1. Environment setup (from scratch)

```bash
cd "/path/to/Tumor-AI-training-model"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Windows (PowerShell): `\.venv\Scripts\Activate.ps1` instead of `source .venv/bin/activate`.

### 2. Point data at your dataset (optional)

By default `configs/config.yaml` uses `data/raw`. To use another folder (e.g. Kaggle extract), pass **`--data_dir`** to the same commands below.

**First time** with that folder (must preprocess into `data/processed`):

```bash
python main.py --data_dir "/path/to/dataset_root"
```

**After** `data/processed` is populated, add **`--skip_preprocessing`** to save time:

```bash
python main.py --data_dir "/path/to/dataset_root" --skip_preprocessing
```

### 3. First full run: preprocess + train + evaluate

Trains **all three** models (`resnet50`, `vit`, `hybrid`) using `configs/config.yaml`:

```bash
source .venv/bin/activate
python main.py --config configs/config.yaml
```

Train **one** model only (example: hybrid):

```bash
python main.py --config configs/config.yaml --models hybrid
```

Shorter trial (2 epochs, ResNet only):

```bash
python main.py --config configs/config.yaml --models resnet50 --epochs 2
```

### 4. Later runs (processed data already exists)

```bash
source .venv/bin/activate
python main.py --config configs/config.yaml --skip_preprocessing
```

Hybrid only, custom epoch count:

```bash
python main.py --models hybrid --skip_preprocessing --epochs 10
```

### 4b. Save a checkpoint after every epoch (resume next run)

In `configs/config.yaml`, **`training.save_last_checkpoint`** is **`true`** by default in this repo. Each finished epoch overwrites:

`results/checkpoints/{model}_last.pth`

with the same full training state as the best checkpoint (weights, optimizer, scheduler, scaler on CUDA, etc.).  
`{model}_best.pth` is still updated only when validation loss improves.

**Continue training** from the last epoch (use **one** model, same config/architecture):

```bash
python main.py --models hybrid --resume results/checkpoints/hybrid_last.pth --skip_preprocessing
```

If the previous run already reached `training.epochs`, increase epochs in the YAML or pass e.g. `--epochs 50` before resuming.

Set `save_last_checkpoint: false` in config if you want to skip the extra disk write each epoch (large models = large files).

### 5. Confirm checkpoints and outputs

Check that best weights exist (names match `--models`):

```bash
ls -la results/checkpoints/
```

You should see files such as:

- `resnet50_best.pth`, `vit_best.pth`, `hybrid_best.pth` — best validation loss so far
- `resnet50_last.pth`, … — **only if** `save_last_checkpoint: true`; latest epoch for resume

Optional: print checkpoint metadata (requires the same venv / PyTorch):

```bash
python -c "
from pathlib import Path
import torch
for p in sorted(Path('results/checkpoints').glob('*_best.pth')):
    ck = torch.load(p, map_location='cpu')
    if isinstance(ck, dict):
        print(p.name, '| epoch=', ck.get('epoch'), '| model_name=', ck.get('model_name'), '| keys:', 'ok' if 'model_state_dict' in ck else 'missing model_state_dict')
    else:
        print(p.name, '| legacy format (state_dict only)')
"
```

Checkpoints are trusted local files; only inspect files you created.

Other artifacts:

- Plots: `results/plots/` (confusion matrices, ROC, `model_comparison.png`, …)
- Logs: `results/logs/`
- Summary CSV: `results/final_results.csv`

### 6. Evaluate only (load checkpoints, no training)

```bash
python main.py --skip_preprocessing --evaluate_only --models resnet50
```

Use one model at a time if you only have that checkpoint.

### 7. Sanity-check tests

```bash
pytest tests/ -q
```

---

## Project Structure

```text
brain_tumor_classifier/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── preprocessing/__init__.py
│   ├── models/__init__.py
│   ├── training/__init__.py
│   ├── evaluation/__init__.py
│   └── utils/__init__.py
├── notebooks/
├── configs/config.yaml
├── results/
│   ├── plots/
│   ├── checkpoints/
│   └── logs/
├── main.py
├── requirements.txt
└── README.md
```

## Setup Instructions

Follow **[CLI quick start](#cli-quick-start-setup-training-and-checkpoints)** (venv, `pip install`, then `python main.py`). Verify paths and classes in `configs/config.yaml` before long runs.

## How to Run

Default entry (preprocess + train all models + evaluate):

```bash
python main.py
```

Same as `python main.py --config configs/config.yaml` with default `--models resnet50 vit hybrid`.

### One-command setup + run (recommended for GitHub users)

If someone clones this repo and wants to run it quickly, use:

```bash
bash setup_and_run.sh
```

This script will:

1. Create `.venv` if needed
2. Install/update dependencies from `requirements.txt`
3. Run `main.py` with defaults
4. Auto-switch to `--evaluate_only` when checkpoints already exist

Useful variants:

```bash
# Force 2-epoch retraining for ResNet50
bash setup_and_run.sh --models resnet50 --epochs 2 --train

# Evaluate existing checkpoint only
bash setup_and_run.sh --models resnet50 --evaluate-only

# Run multiple models
bash setup_and_run.sh --models "resnet50 vit"
```

Typical flow:

1. Load dataset and split into train/validation/test.
2. Train selected model(s).
3. Evaluate performance.
4. Save checkpoints, plots, and logs in `results/`.

## Configuration

All paths and hyperparameters are centralized in:

- `configs/config.yaml`

Includes:

- Data paths and split ratios
- Model class metadata
- Training hyperparameters
- Output directories for checkpoints, plots, and logs
- `training.device`: `auto` (CUDA, else MPS, else CPU), or force `cuda` / `mps` / `cpu`
- `training.seed` and optional `training.cudnn_deterministic` (CUDA reproducibility; can reduce speed)
- `training.min_delta`: minimum validation-loss improvement for a new best checkpoint (aligned with early stopping)
- `training.save_last_checkpoint`: default **`true`** — writes `results/checkpoints/{model}_last.pth` after every epoch so you can **`--resume`**; set `false` to save disk I/O on huge models

## Checkpoints and resume

Checkpoints are **trusted local files** (full Python pickles). Only load files you created.

**Artifacts**

- `{model}_best.pth` — best validation loss so far, written atomically (temp file then rename). Contains `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict` (CUDA AMP only), `epoch` (last completed epoch when saved), `best_metric`, `best_epoch`, `model_name`, `torch_version`, and a small `training_snapshot`.
- `{model}_last.pth` — optional; same schema, overwritten each epoch when `save_last_checkpoint: true`.

**Evaluation without training**

```bash
python main.py --models resnet50 --evaluate_only
```

**Resume training** (same config and architecture as the run that produced the file; use **exactly one** model):

```bash
python main.py --models hybrid --resume results/checkpoints/hybrid_last.pth --skip_preprocessing
```

**Override epochs**

```bash
python main.py --models resnet50 --epochs 5 --skip_preprocessing
```

**Tests**

```bash
pytest tests/ -q
# or
python -m unittest discover tests -v
```
