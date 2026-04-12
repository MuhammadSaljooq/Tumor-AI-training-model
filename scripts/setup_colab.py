#!/usr/bin/env python3
"""Bootstrap this project inside Google Colab (clone, deps, dataset links, sanity checks).

Fastest path: open ``notebooks/Colab_Quickstart.ipynb`` in Colab, choose GPU, **Run all**.

Typical Colab workflow (GPU runtime recommended):

  from google.colab import drive
  drive.mount("/content/drive")

  # Option A — clone via this script (from an empty /content cell)
  !wget -q https://raw.githubusercontent.com/MuhammadSaljooq/Tumor-AI-training-model/main/scripts/setup_colab.py
  !python setup_colab.py --mount-drive \\
      --processed-data "/content/drive/MyDrive/path/to/processed"

  # Option B — already cloned
  %cd /content/brain_tumor_classifier
  !python scripts/setup_colab.py --skip-clone \\
      --processed-data "/content/drive/MyDrive/path/to/processed"

  # Then train (adjust models / config as needed). Data should live at data/processed
  # (this script can symlink it). Loaders use processed_dir from configs/config.yaml.
  !python main.py --config configs/config.yaml --skip_preprocessing --models resnet50

``processed`` should follow this layout (either split you already use in the repo)::

  processed/
    Training/<class>/*.jpg
    Testing/<class>/*.jpg

or flat class folders under ``processed/`` when not using the Training/Testing layout.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def _is_colab_runtime() -> bool:
    return bool(os.environ.get("COLAB_RELEASE_TAG")) or os.path.isfile("/content/.default_colab_shell")


def _try_mount_google_drive() -> None:
    try:
        from google.colab import drive  # type: ignore[import-not-found]
    except ImportError:
        print("google.colab not available — skipping Drive mount (not in Colab?).", flush=True)
        return
    print("Mounting Google Drive (follow the link if prompted)…", flush=True)
    drive.mount("/content/drive")


def _resolve_project_root(explicit: Path | None, clone_dir: Path) -> Path:
    if explicit is not None:
        root = explicit.resolve()
        if not (root / "main.py").is_file() or not (root / "src").is_dir():
            raise SystemExit(
                f"--project-root is not the repo root (need main.py + src/): {root}"
            )
        return root
    here = Path.cwd().resolve()
    if (here / "main.py").is_file() and (here / "src").is_dir():
        return here
    candidate = clone_dir.resolve()
    if (candidate / "main.py").is_file() and (candidate / "src").is_dir():
        return candidate
    raise SystemExit(
        "Could not find project root. Run from the repo directory, or pass "
        "--project-root, or clone with --clone-dir first."
    )


def _clone_or_update(*, repo: str, branch: str, target: Path) -> None:
    target = target.resolve()
    if (target / ".git").is_dir():
        print(f"Updating existing repo at {target}", flush=True)
        _run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", branch])
        _run(["git", "-C", str(target), "checkout", branch])
        _run(["git", "-C", str(target), "pull", "origin", branch])
        return
    if target.exists():
        raise SystemExit(f"Refusing to clone: path exists and is not a git repo: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            branch,
            repo,
            str(target),
        ]
    )


def _install_dependencies(project_root: Path, *, quiet: bool) -> None:
    req = project_root / "requirements.txt"
    if not req.is_file():
        raise SystemExit(f"Missing requirements.txt at {req}")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
    if quiet:
        cmd.append("-q")
    _run(cmd)
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req)]
    if quiet:
        cmd.insert(-2, "-q")
    _run(cmd, cwd=project_root)


def _replace_with_symlink(dest: Path, source: Path, *, force: bool = False) -> None:
    source = source.resolve()
    if not source.exists():
        raise SystemExit(f"Dataset path does not exist: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() and dest.resolve() == source:
        print(f"Symlink already set: {dest} -> {source}", flush=True)
        return
    if dest.is_symlink():
        dest.unlink()
    elif dest.is_dir():
        if any(dest.iterdir()):
            if force:
                shutil.rmtree(dest)
                dest.symlink_to(source, target_is_directory=True)
                print(f"Linked {dest} -> {source}", flush=True)
                return
            raise SystemExit(
                f"{dest} already exists and is not empty. Remove it, pick another "
                f"--processed-data / --raw-data path, or pass --force-link."
            )
        dest.rmdir()
    elif dest.exists():
        raise SystemExit(f"{dest} exists and is not a directory — remove it manually.")
    dest.symlink_to(source, target_is_directory=True)
    print(f"Linked {dest} -> {source}", flush=True)


def _dataset_sanity_check(processed_root: Path) -> None:
    training = processed_root / "Training"
    testing = processed_root / "Testing"
    if training.is_dir() and testing.is_dir():
        print(f"Detected Training/Testing layout under {processed_root}", flush=True)
        return
    # flat layout: at least one class subfolder with an image
    subs = [p for p in processed_root.iterdir() if p.is_dir()]
    image_globs = ("*/*.jpg", "*/*.jpeg", "*/*.png", "*/*.bmp", "*/*.tif", "*/*.tiff", "*/*.webp")
    images: list[Path] = []
    for pattern in image_globs:
        images.extend(processed_root.glob(pattern))
    if subs and images:
        print(f"Detected flat class-folder layout under {processed_root}", flush=True)
        return
    print(
        "WARNING: Could not confirm dataset layout under data/processed. "
        "Expected Training/Testing class folders or flat class directories. "
        "See README dataset section.",
        flush=True,
    )


def _maybe_write_colab_config(project_root: Path) -> Path | None:
    """Fewer dataloader workers avoids occasional Colab worker crashes."""
    try:
        import yaml
    except ImportError:
        print(
            "PyYAML is not installed, so configs/config_colab.yaml was not generated. "
            "Install dependencies first or rerun without --skip-install.",
            flush=True,
        )
        return None

    src = project_root / "configs" / "config.yaml"
    if not src.is_file():
        return None
    dst = project_root / "configs" / "config_colab.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        return None
    data_cfg = cfg.setdefault("data", {})
    if not isinstance(data_cfg, dict):
        cfg["data"] = {"num_workers": 2}
    else:
        data_cfg["num_workers"] = 2
    dst.write_text(
        yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Wrote {dst} with num_workers: 2 (use --config configs/config_colab.yaml)", flush=True)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up brain_tumor_classifier on Google Colab.")
    parser.add_argument(
        "--repo",
        default="https://github.com/MuhammadSaljooq/Tumor-AI-training-model.git",
        help="Git remote to clone.",
    )
    parser.add_argument("--branch", default="main", help="Branch to check out after clone.")
    parser.add_argument(
        "--clone-dir",
        type=Path,
        default=Path("/content/brain_tumor_classifier"),
        help="Where to clone the repository (Colab default).",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Do not clone; use current directory or --project-root.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Explicit path to repo root (folder containing main.py).",
    )
    parser.add_argument(
        "--mount-drive",
        action="store_true",
        help="Call google.colab.drive.mount before linking dataset paths.",
    )
    parser.add_argument(
        "--processed-data",
        type=Path,
        default=None,
        help="Host path to preprocessed data; symlinked to <repo>/data/processed.",
    )
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=None,
        help="Host path to raw class folders; symlinked to <repo>/data/raw.",
    )
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install -r requirements.txt.")
    parser.add_argument("--quiet-pip", action="store_true", help="Pass -q to pip.")
    parser.add_argument(
        "--force-link",
        action="store_true",
        help="Replace existing non-empty data/raw or data/processed when creating dataset symlinks.",
    )
    parser.add_argument(
        "--write-colab-config",
        action="store_true",
        help="Write configs/config_colab.yaml with num_workers: 2 for Colab stability.",
    )
    args = parser.parse_args()

    if _is_colab_runtime():
        print("Colab runtime detected.", flush=True)
    else:
        print("Not running in Colab — continuing anyway (local smoke test).", flush=True)

    if args.mount_drive:
        _try_mount_google_drive()

    if not args.skip_clone:
        _clone_or_update(repo=args.repo, branch=args.branch, target=args.clone_dir)
        os.chdir(args.clone_dir)
        print(f"Working directory: {Path.cwd()}", flush=True)

    project_root = _resolve_project_root(args.project_root, args.clone_dir)

    if not args.skip_install:
        _install_dependencies(project_root, quiet=args.quiet_pip)

    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.processed_data is not None:
        _replace_with_symlink(
            project_root / "data" / "processed",
            args.processed_data,
            force=args.force_link,
        )
    if args.raw_data is not None:
        _replace_with_symlink(
            project_root / "data" / "raw",
            args.raw_data,
            force=args.force_link,
        )

    processed_path = project_root / "data" / "processed"
    if processed_path.exists():
        _dataset_sanity_check(processed_path.resolve())
    else:
        print(
            f"NOTE: {processed_path} is missing. Upload data or pass --processed-data from Drive.",
            flush=True,
        )

    cfg_hint = "configs/config.yaml"
    if args.write_colab_config:
        colab_cfg = _maybe_write_colab_config(project_root)
        if colab_cfg:
            cfg_hint = str(colab_cfg.relative_to(project_root))

    try:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"PyTorch CUDA available: {torch.cuda.is_available()} (using device: {dev})", flush=True)
    except ImportError:
        print("PyTorch import check skipped.", flush=True)

    print("\nDone. Example commands from the repo root:\n", flush=True)
    print(f"  cd {project_root}", flush=True)
    has_processed = (project_root / "data" / "processed").exists()
    if has_processed:
        print(
            f"  python main.py --config {cfg_hint} --skip_preprocessing --models resnet50",
            flush=True,
        )
    else:
        print(
            f"  python main.py --config {cfg_hint} --models resnet50",
            flush=True,
        )
        print("  # data/processed not detected, so preprocessing step is enabled.", flush=True)
    print(
        "\nColab notebooks: before `from src...`, run `%cd` to the repo root, e.g.:\n"
        "  %cd /content/brain_tumor_classifier\n"
        "Or set:\n"
        "  import os, sys\n"
        "  os.environ['BRAIN_TUMOR_CLASSIFIER_ROOT'] = '/content/brain_tumor_classifier'\n"
        "  sys.path.insert(0, os.environ['BRAIN_TUMOR_CLASSIFIER_ROOT'])\n"
        "`notebooks/experiments.ipynb` auto-detects the root once cwd is inside the repo.\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
