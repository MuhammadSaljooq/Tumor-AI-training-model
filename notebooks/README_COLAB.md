# Google Colab: one-cell training guide

This document describes **[`Colab_Tumor_AI_Training.ipynb`](Colab_Tumor_AI_Training.ipynb)** — a **single Python cell** (plus a short intro markdown cell) that prepares the environment and runs the full training pipeline on [Google Colab](https://colab.research.google.com/).

## Prerequisites

1. **GPU runtime:** **Runtime → Change runtime type →** pick a **GPU** (e.g. T4). Training is much slower on CPU.
2. **Google account:** The cell calls **`drive.mount("/content/drive")`**, so you will authorize Google Drive when the cell runs.
3. **Dataset access:** Either a Kaggle API token (`kaggle.json`) or a Drive folder that already contains your MRI dataset (see [Data sources](#data-sources)).

## What the code cell does (order of operations)

The cell runs **top to bottom** in one shot. Rough flow:

| Step | What happens |
|------|----------------|
| 1 | **Mount Google Drive** so you can read `kaggle.json`, dataset folders, and (optionally) save checkpoints outside the ephemeral VM disk. |
| 2 | **`nvidia-smi`** (informational) to confirm a GPU is visible. |
| 3 | **Clone or update** the repo into **`PROJECT_DIR`** (default `/content/Tumor-AI-training-model`) from **`REPO_URL`** / **`BRANCH`**. |
| 4 | **`pip install`** `requirements.txt` (and **`kaggle`** only if **`DATA_SOURCE == "kaggle"`**). |
| 5 | **Load** [`configs/config_colab.yaml`](../configs/config_colab.yaml), **merge** your Python variables (hyperparameters, optional Drive results paths), and **write** **`configs/config_colab_runtime.yaml`**. That file is **regenerated every run** and is what `main.py` reads. |
| 6 | **Prepare data:** either download/extract from Kaggle into `data/raw/`, or copy / point at Drive (see below). |
| 7 | **Build the `main.py` command:** `--config` runtime YAML, `--models`, `--epochs`, optional `--data_dir`, optional **`--resume`** (see [Checkpoints and resume](#checkpoints-and-resume)). |
| 8 | **Run training** in the repo root with **`python -u`** and **`PYTHONUNBUFFERED=1`** so **tqdm batch progress** and log lines **stream live** in the Colab output (not buffered until the end). When finished, the cell prints **`final_results.csv`** if it exists (path depends on `RESULTS_DRIVE_DIR`). |

There is no separate “setup cell”: everything lives in that one code cell.

## Top-of-cell settings (what to edit)

### Run and repository

| Variable | Role |
|----------|------|
| **`EPOCHS`** | Passed to **`main.py --epochs`** and written into **`training.epochs`** in the runtime YAML. |
| **`MODELS`** | Space-separated list, e.g. `"resnet50"`, `"vit"`, `"resnet50 vit"`. Each name is passed to **`--models`**. |
| **`REPO_URL`**, **`BRANCH`**, **`PROJECT_DIR`** | Where the code is cloned and executed. |
| **`CONFIG_BASE`** | Base YAML merged with your overrides (usually `configs/config_colab.yaml`). |
| **`RUNTIME_CONFIG`** | Output path for the merged config (usually `configs/config_colab_runtime.yaml`). |
| **`EXTRA_ARGS`** | Extra CLI tokens for `main.py` (string, split with `shlex`). Example: `'--skip_preprocessing'`. Do not add a second **`--resume`** if you rely on auto-resume (see below). |

### Checkpoints and resume

Colab’s **`/content`** disk is **reset** when you get a **new runtime**. Checkpoints under the default `results/checkpoints/` inside the clone **disappear** unless you redirect outputs to Drive.

| Variable | Role |
|----------|------|
| **`RESULTS_DRIVE_DIR`** | If set (e.g. `"/content/drive/MyDrive/TumorAI_results"`), the runtime config points **`paths.checkpoints`**, **`paths.plots`**, **`paths.logs`**, and **`paths.final_results_parent`** under that folder so artifacts **persist** across sessions. |
| **`RESUME_IF_CHECKPOINT_EXISTS`** | If **`True`** and **`MODELS`** contains **exactly one** model, the cell checks for **`{model}_last.pth`** under the configured checkpoints directory and, if found, appends **`--resume <path>`** so training continues from the saved **epoch, optimizer, and scheduler** state. |
| **`SAVE_LAST_CHECKPOINT`** | Written into the runtime YAML. When **`True`**, training saves **`{model}_last.pth` after every completed epoch** (same file overwritten each time). **`{model}_best.pth`** still updates only when validation loss improves. |

**Multi-model runs:** `main.py` only allows **`--resume`** with a **single** `--models` entry. If you list several models, auto-resume is skipped; train one model per run or pass **`--resume`** manually via **`EXTRA_ARGS`** for a single-model command.

### Hyperparameters (training and data)

These are merged into the runtime YAML under **`training`** and **`data`**:

- **`LR`**, **`WEIGHT_DECAY`**, **`PATIENCE`**, **`MIN_DELTA`**, **`SEED`**
- **`USE_CLASS_WEIGHTS`**, **`USE_WEIGHTED_SAMPLER`**, **`SAVE_LAST_CHECKPOINT`**
- **`BATCH_SIZE`**, **`NUM_WORKERS`**, **`IMG_SIZE`**
- **`TRAIN_SPLIT`**, **`VAL_SPLIT`**, **`TEST_SPLIT`**

**Note:** Changing split ratios only affects the next run if preprocessing rebuilds **`data/processed/`** (remove that folder first if you changed splits).

**Quick tuning hints**

- **ViT:** Often benefits from a **lower** learning rate (e.g. `1e-4` or `5e-5`) and **more epochs** (e.g. 30–50) with your existing early stopping.
- **ResNet50:** `LR = 3e-4` and **`BATCH_SIZE`** 16–32 are strong starting points on a T4 if memory allows.
- **OOM:** Lower **`BATCH_SIZE`** (e.g. 8 or 12).

## Data sources

### `DATA_SOURCE = "kaggle"`

1. Ensure **`KAGGLE_JSON_DRIVE`** points to `kaggle.json` on Drive, **or** upload when the cell prompts you.
2. The cell runs **`kaggle datasets download`** for the brain-tumor MRI dataset used by this project and extracts it under **`data/raw/`**.

### `DATA_SOURCE = "drive"`

1. Set **`DRIVE_DATA_PATH`** to the mounted path of a folder that contains **`Training/`** (and **`Testing/`** if applicable).
2. **`DRIVE_USE_IN_PLACE = False`:** copies that folder’s contents into **`data/raw/`** (often faster training I/O).
3. **`DRIVE_USE_IN_PLACE = True`:** passes **`--data_dir`** to `main.py` so raw images are read **from Drive** (saves VM disk; can be slower).

Shared drives use paths like `/content/drive/Shareddrives/<Name>/...`.

## What gets produced

With default paths (no **`RESULTS_DRIVE_DIR`**):

- Checkpoints: `results/checkpoints/{model}_best.pth`, `{model}_last.pth`
- Plots and logs: `results/plots/`, `results/logs/`
- Summary CSV: `results/final_results.csv`

With **`RESULTS_DRIVE_DIR = "/content/drive/MyDrive/TumorAI_results"`** (example):

- `.../TumorAI_results/checkpoints/`
- `.../TumorAI_results/plots/`
- `.../TumorAI_results/logs/`
- `.../TumorAI_results/final_results.csv`

## How this relates to the rest of the repo

- Training logic, models, and evaluation live in the cloned repository (same as local **`python main.py`**).
- The notebook does **not** replace `main.py`; it **configures** and **invokes** it with a generated YAML file.

## Troubleshooting

| Symptom | What to check |
|---------|----------------|
| Training never resumes | Set **`RESULTS_DRIVE_DIR`** so `*_last.pth` survives; use **one** model in **`MODELS`**; ensure **`RESUME_IF_CHECKPOINT_EXISTS`** is **`True`**. |
| “Starting from epoch 1” every time | Expected if checkpoints were only on the VM disk and the runtime was recycled. Use Drive for **`RESULTS_DRIVE_DIR`**. |
| Duplicate **`--resume`** error | Remove **`--resume`** from **`EXTRA_ARGS`** when using auto-resume, or set **`RESUME_IF_CHECKPOINT_EXISTS = False`** and pass **`--resume`** only in **`EXTRA_ARGS`**. |
| No training progress until the run finishes | Pull the latest repo (notebook uses **`python -u`** + unbuffered env). The trainer prints **tqdm** bars per batch to **stdout**. |
| Preprocessing / split changes ignored | Delete **`data/processed/`** in the clone before re-running so splits are rebuilt. |
| ViT accuracy low with few epochs | Increase **`EPOCHS`**, lower **`LR`**, and see the hints above. |

For command-line usage on your own machine, see the main **[`README.md`](../README.md)**.
