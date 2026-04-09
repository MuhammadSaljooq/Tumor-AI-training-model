#!/usr/bin/env python3
"""One-file Google Colab bootstrap + run script.

This script is designed to be run in one Colab cell and handle:
- optional Google Drive mount
- clone/update repository
- dependency install
- dataset path linking (raw/processed)
- optional config overrides for Colab
- train/evaluate run

Example:
  !python colab_one_click.py \
      --mount-drive \
      --processed-data "/content/drive/MyDrive/brain_mri/processed" \
      --branch google-colab \
      --models resnet50 hybrid \
      --epochs 10
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"+ {printable}", flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def is_colab() -> bool:
    return bool(os.environ.get("COLAB_RELEASE_TAG")) or Path("/content").exists()


def mount_drive_if_requested(mount: bool) -> None:
    if not mount:
        return
    try:
        from google.colab import drive  # type: ignore[import-not-found]
    except ImportError:
        print("google.colab not available; skipping Drive mount.", flush=True)
        return
    print("Mounting Google Drive...", flush=True)
    drive.mount("/content/drive")


def clone_or_update(repo: str, branch: str, clone_dir: Path) -> Path:
    clone_dir = clone_dir.resolve()
    if (clone_dir / ".git").is_dir():
        print(f"Updating existing clone at {clone_dir}", flush=True)
        run(["git", "-C", str(clone_dir), "fetch", "--depth", "1", "origin", branch])
        run(["git", "-C", str(clone_dir), "checkout", branch])
        run(["git", "-C", str(clone_dir), "pull", "origin", branch])
        return clone_dir

    if clone_dir.exists():
        raise SystemExit(
            f"{clone_dir} already exists but is not a git clone. "
            f"Remove it or pass a different --clone-dir."
        )

    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", "--branch", branch, repo, str(clone_dir)])
    return clone_dir


def install_requirements(project_root: Path) -> None:
    req = project_root / "requirements.txt"
    if not req.is_file():
        raise SystemExit(f"requirements.txt not found at {req}")
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])


def safe_link(dest: Path, source: Path, force: bool) -> None:
    source = source.expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Dataset path does not exist: {source}")

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_symlink() and dest.resolve() == source:
        print(f"Link already set: {dest} -> {source}", flush=True)
        return

    if dest.is_symlink():
        dest.unlink()
    elif dest.is_dir():
        if any(dest.iterdir()):
            if not force:
                raise SystemExit(
                    f"{dest} is non-empty. Use --force-link or clean it first."
                )
            shutil.rmtree(dest)
        else:
            dest.rmdir()
    elif dest.exists():
        raise SystemExit(f"{dest} exists and is not a directory/symlink.")

    dest.symlink_to(source, target_is_directory=True)
    print(f"Linked {dest} -> {source}", flush=True)


def _has_any_images(root: Path) -> bool:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
    for pat in patterns:
        if any(root.rglob(pat)):
            return True
    return False


def validate_dataset_availability(project_root: Path) -> bool:
    """Return True if processed data exists, otherwise ensure raw data is available."""
    processed_dir = project_root / "data" / "processed"
    raw_dir = project_root / "data" / "raw"

    if processed_dir.exists() and _has_any_images(processed_dir):
        print(f"Using processed dataset: {processed_dir}", flush=True)
        return True

    if raw_dir.exists() and _has_any_images(raw_dir):
        print(f"Processed data missing; using raw dataset for preprocessing: {raw_dir}", flush=True)
        return False

    raise SystemExit(
        "Dataset not found. Provide one of:\n"
        "  1) --processed-data \"/content/drive/MyDrive/.../processed\" (preferred)\n"
        "  2) --raw-data \"/content/drive/MyDrive/.../raw\"\n"
        "Current paths checked:\n"
        f"  - {processed_dir}\n"
        f"  - {raw_dir}\n"
        "No image files were detected."
    )


def write_runtime_config(
    project_root: Path,
    *,
    epochs: int | None,
    num_workers: int,
    batch_size: int | None,
    lr: float | None,
) -> Path:
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML missing after install; cannot write runtime config.")

    base_cfg = project_root / "configs" / "config.yaml"
    if not base_cfg.is_file():
        raise SystemExit(f"Base config not found: {base_cfg}")

    cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit("Invalid YAML config format.")

    data_cfg = cfg.setdefault("data", {})
    train_cfg = cfg.setdefault("training", {})
    if not isinstance(data_cfg, dict) or not isinstance(train_cfg, dict):
        raise SystemExit("Config sections 'data'/'training' are malformed.")

    data_cfg["num_workers"] = int(num_workers)
    if batch_size is not None:
        data_cfg["batch_size"] = int(batch_size)
    if epochs is not None:
        train_cfg["epochs"] = int(epochs)
    if lr is not None:
        train_cfg["lr"] = float(lr)

    out_cfg = project_root / "configs" / "config_colab_runtime.yaml"
    out_cfg.write_text(
        yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Wrote runtime config: {out_cfg}", flush=True)
    return out_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="One-file Colab setup + run for brain_tumor_classifier.")
    parser.add_argument("--repo", default="https://github.com/MuhammadSaljooq/Tumor-AI-training-model.git")
    parser.add_argument("--branch", default="google-colab")
    parser.add_argument("--clone-dir", type=Path, default=Path("/content/brain_tumor_classifier"))
    parser.add_argument("--mount-drive", action="store_true")
    parser.add_argument("--processed-data", type=Path, default=None)
    parser.add_argument("--raw-data", type=Path, default=None)
    parser.add_argument("--force-link", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--models", nargs="+", default=["resnet50"], choices=["resnet50", "vit", "hybrid"])
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    if is_colab():
        print("Colab-like environment detected.", flush=True)
    else:
        print("Warning: not running in Colab; continuing anyway.", flush=True)

    mount_drive_if_requested(args.mount_drive)

    project_root = clone_or_update(args.repo, args.branch, args.clone_dir)
    os.chdir(project_root)
    os.environ["BRAIN_TUMOR_CLASSIFIER_ROOT"] = str(project_root)

    if not args.skip_install:
        install_requirements(project_root)

    if args.processed_data is not None:
        safe_link(project_root / "data" / "processed", args.processed_data, args.force_link)
    if args.raw_data is not None:
        safe_link(project_root / "data" / "raw", args.raw_data, args.force_link)

    cfg_path = write_runtime_config(
        project_root,
        epochs=args.epochs,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    has_processed = validate_dataset_availability(project_root)
    cmd = [
        sys.executable,
        "main.py",
        "--config",
        str(cfg_path),
        "--models",
        *args.models,
    ]
    if has_processed:
        cmd.append("--skip_preprocessing")
    if args.evaluate_only:
        cmd.append("--evaluate_only")

    print(f"Project root: {project_root}", flush=True)
    print("Launching pipeline...", flush=True)
    run(cmd, cwd=project_root)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
