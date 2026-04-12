"""Plot per-class probability bar chart for one image using a saved checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import get_model
from src.preprocessing import get_val_transforms
from src.utils.checkpoint_io import load_checkpoint


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bar chart of class probabilities for one MRI image.")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "vit", "hybrid"])
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    classes: list[str] = list(cfg["model"]["classes"])
    num_classes = int(cfg["model"]["num_classes"])
    img_size = int(cfg["data"]["img_size"])

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        hint = (
            "\n  Use the real path to your .jpg on this Mac, not a /path/to/... placeholder.\n"
            "  Tip (Finder): right-click the file, hold Option, choose “Copy … as Pathname”, paste after --image "
            "(wrap in quotes if the path has spaces).\n"
            "  Example: --image \"$HOME/Desktop/archive (1)/Testing/glioma/Te-gl_3.jpg\""
        )
        raise FileNotFoundError(f"Image not found:\n  {image_path}{hint}")

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found:\n  {ckpt_path}\n"
            "  Run from the repo root or pass a full path to your *_best.pth file."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args.model, num_classes=num_classes, pretrained=False).to(device)
    state = load_checkpoint(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    transform = get_val_transforms(img_size=img_size)
    pil = Image.open(image_path).convert("RGB")
    image_np = np.array(pil)
    t = transform(image=image_np)
    if isinstance(t, dict) and "image" in t:
        x = t["image"]
    else:
        x = t
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_label = classes[pred_idx]

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else ROOT / "results" / "plots" / f"{image_path.stem}_prediction_metrics.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#2ecc71" if i == pred_idx else "#95a5a6" for i in range(len(classes))]
    bars = ax.barh(classes, probs, color=colors, edgecolor="#2c3e50", linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title(f"{image_path.name} — predicted: {pred_label} ({probs[pred_idx]:.1%})")
    for bar, p in zip(bars, probs):
        ax.text(min(p + 0.02, 0.98), bar.get_y() + bar.get_height() / 2, f"{p:.3f}", va="center", fontsize=10)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")
    print("Class probabilities:", dict(zip(classes, probs.round(4).tolist())))


if __name__ == "__main__":
    main()
