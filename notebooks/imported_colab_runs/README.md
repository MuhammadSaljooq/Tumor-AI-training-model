# Imported Colab runs (ResNet50 + ViT)

This folder stores **exported Jupyter notebooks** from Google Colab sessions that trained **one model each** using `colab_one_click.py` on branch **`5-fold-optimization`**, with processed data at **`MyDrive/MriDataset/processed`**.

## Important: what is (and is not) saved here

| Saved in repo | Not in the notebooks |
|---------------|----------------------|
| Full Colab output logs inside **`BrainTumor_resnet50.ipynb`** and **`BrainTumor_VIT.ipynb`** | **`.pth` checkpoint files** (weights, optimizer, epoch). Those only exist as binary files on disk. |

You **cannot** “rebuild” a checkpoint from the notebook text. To continue training from the same weights, you must **copy the real `.pth` files** from wherever they still exist (see below).

## Where the checkpoints probably are

On the Colab run shown in the notebooks, training wrote to:

```text
/content/brain_tumor_classifier/results/checkpoints/
```

That path is on the **Colab VM** and is **lost** when the runtime ends **unless** you (or `colab_one_click.py`) also stored copies on **Google Drive**.

Check Drive for folders such as:

- `MyDrive/.../results/checkpoints/`
- Any folder you configured for experiment output on that branch

Look for:

- **`resnet50_best.pth`**, **`resnet50_last.pth`**
- **`vit_best.pth`**, **`vit_last.pth`**

`*_last.pth` is best for **resume** (full training state through the last finished epoch). `*_best.pth` is best for **evaluation** (lowest val loss).

## Copy checkpoints into this repository

From your machine (after downloading from Drive or copying from an environment that still has the files):

```bash
cd "/path/to/Tumor-AI-training-model"
mkdir -p results/checkpoints
cp /path/to/resnet50_last.pth results/checkpoints/
cp /path/to/vit_last.pth results/checkpoints/
# optional: also copy *_best.pth for eval-only workflows
```

## Resume training (more epochs)

Use **one** model per command and **`--resume`** pointing at `*_last.pth`:

```bash
python main.py --config configs/config.yaml --models resnet50 \
  --resume results/checkpoints/resnet50_last.pth --epochs 30 \
  --skip_preprocessing
```

```bash
python main.py --config configs/config.yaml --models vit \
  --resume results/checkpoints/vit_last.pth --epochs 40 \
  --skip_preprocessing
```

Adjust **`--config`**, **`--epochs`**, and paths to match your setup. **`--skip_preprocessing`** is appropriate if **`data/processed`** is already built and matches the run you resumed from.

**Branch / code compatibility:** These runs used **`5-fold-optimization`**. If your current checkout is **`checkpoint_added`**, the checkpoint format should still match if the training script saved the same keys (`model_state_dict`, `optimizer_state_dict`, `epoch`, etc.). If resume fails with a state_dict error, train on the **same branch** that created the checkpoint or ask to align architectures.

## Machine-readable summary

See **[`colab_run_manifest.json`](colab_run_manifest.json)** for parsed metrics and filenames.
