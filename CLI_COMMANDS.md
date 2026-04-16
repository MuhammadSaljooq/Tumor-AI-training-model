# CLI Commands Reference

Run all commands from the repo root:

```bash
cd "/Users/shireenafzal/Desktop/hybrid bug /Tumor-AI-training-model"
```

## 1) Setup environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Start the model (train + evaluate)

Default run (all models, full pipeline):

```bash
python main.py --config configs/config.yaml
```

Run only one model:

```bash
python main.py --config configs/config.yaml --models resnet50
```

If processed data already exists:

```bash
python main.py --config configs/config.yaml --models resnet50 --skip_preprocessing
```

## 3) GPU run (recommended)

Set `training.device` in `configs/config.yaml` to one of:
- `cuda` (NVIDIA GPU)
- `mps` (Apple Silicon GPU)
- `auto` (tries CUDA, then MPS, then CPU)

Then run:

```bash
python main.py --config configs/config.yaml --models resnet50 --skip_preprocessing
```

## 4) Evaluate only (no training)

```bash
python main.py --config configs/config.yaml --models resnet50 --evaluate_only --skip_preprocessing
```

## 5) Resume training from checkpoint

```bash
python main.py --config configs/config.yaml --models resnet50 --resume results/checkpoints/resnet50_last.pth --skip_preprocessing
```

## 6) Quick test run (short epochs)

```bash
python main.py --config configs/config.yaml --models resnet50 --epochs 2 --skip_preprocessing
```
## 7) Generate metric/probability chart for any image

Use the helper script to create a per-image class probability chart PNG:

```bash
python scripts/plot_single_image_prob_chart.py \
  --image "/absolute/path/to/your/image.jpg" \
  --checkpoint "results/checkpoints/resnet50_best.pth" \
  --model resnet50 \
  --config configs/config.yaml \
  --out "results/plots/my_image_prediction_metrics.png"
```

If `--out` is omitted, output is saved to:

```text
results/plots/<image_name>_prediction_metrics.png
```

## 8) Useful checks

Check checkpoints:

```bash
ls -lah results/checkpoints
```

Check generated plots:

```bash
ls -lah results/plots
```

Check logs:

```bash
ls -lah results/logs
```

## 9) One-command runner (alternative)

```bash
bash setup_and_run.sh
```

Examples:

```bash
bash setup_and_run.sh --models resnet50 --epochs 2 --train
bash setup_and_run.sh --models "resnet50 vit" --evaluate-only
```
