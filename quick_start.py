"""Quick start script to fetch data and run the project pipeline."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


DATASET_REF = "masoudnickparvar/brain-tumor-mri-dataset"


def has_kaggle_credentials() -> bool:
    """Return True if Kaggle credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    env_ready = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    return kaggle_json.exists() or env_ready


def print_kaggle_instructions() -> None:
    """Print setup steps when Kaggle credentials are missing."""
    print("Kaggle API key not found.")
    print("Setup instructions:")
    print("1) Create API token from your Kaggle account settings.")
    print("2) Save it to ~/.kaggle/kaggle.json")
    print("3) Run: chmod 600 ~/.kaggle/kaggle.json")
    print("   OR set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")


def download_dataset(target_dir: Path) -> Path:
    """Download Kaggle dataset zip into target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "brain-tumor-mri-dataset.zip"

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET_REF,
        "-p",
        str(target_dir),
    ]
    subprocess.run(cmd, check=True)
    return zip_path


def unzip_dataset(zip_path: Path, output_dir: Path) -> None:
    """Extract dataset archive to data/raw directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)


def run_main() -> None:
    """Run main.py with default arguments."""
    subprocess.run([sys.executable, "main.py"], check=True)


def main() -> None:
    """Download dataset, unpack it, then execute training pipeline."""
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"

    if not has_kaggle_credentials():
        print_kaggle_instructions()
        sys.exit(1)

    try:
        print("Downloading dataset from Kaggle...")
        zip_path = download_dataset(data_dir)

        if raw_dir.exists():
            shutil.rmtree(raw_dir)

        print(f"Extracting dataset to {raw_dir} ...")
        unzip_dataset(zip_path, raw_dir)
        print("Dataset ready.")

        print("Launching main.py with default arguments...")
        run_main()
    except FileNotFoundError:
        print("Kaggle CLI is not installed or not found in PATH.")
        print("Install with: pip install kaggle")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"Failed while running command: {exc}")
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
