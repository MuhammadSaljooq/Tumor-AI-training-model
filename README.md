# Brain Tumor Classifier

Automated brain tumor classification project using MRI images and deep learning.
The pipeline is designed to compare three model families under a unified training and evaluation flow.

## Full Documentation

For complete project documentation, see:

- `docs/PROJECT_DOCUMENTATION.md`

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

Place the dataset in:

```text
data/raw/
  glioma/
  meningioma/
  pituitary/
  no_tumor/
```

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

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Verify configuration in `configs/config.yaml`.

## How to Run

Run the main pipeline:

```bash
python main.py
```

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
