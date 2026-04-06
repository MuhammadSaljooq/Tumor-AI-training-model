"""Dataset and dataloader utilities."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms.functional import to_tensor

from ..preprocessing import get_train_transforms, get_val_transforms

_VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_CLASS_ALIASES = {
    "no_tumor": "no_tumor",
    "notumor": "no_tumor",
}


class BrainTumorDataset(Dataset):
    """Torch dataset for brain tumor image classification."""

    def __init__(self, image_paths: list, labels: list, transform=None) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path = self.image_paths[index]
        label = int(self.labels[index])

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        if self.transform is not None:
            transformed: Any = self.transform(image=image_np)
            if isinstance(transformed, dict) and "image" in transformed:
                image_tensor = transformed["image"]
            else:
                image_tensor = transformed
        else:
            image_tensor = to_tensor(image)

        return image_tensor, label


def _read_batch_size(config: dict) -> int:
    """Read batch size from either root-level or nested data config."""
    if "batch_size" in config:
        return int(config["batch_size"])
    if "data" in config and isinstance(config["data"], dict) and "batch_size" in config["data"]:
        return int(config["data"]["batch_size"])
    return 32


def _read_img_size(config: dict) -> int:
    """Read image size from either root-level or nested data config."""
    if "img_size" in config:
        return int(config["img_size"])
    if "data" in config and isinstance(config["data"], dict) and "img_size" in config["data"]:
        return int(config["data"]["img_size"])
    return 224


def _read_num_workers(config: dict) -> int:
    """Read dataloader worker count from config."""
    if "num_workers" in config:
        return int(config["num_workers"])
    if "data" in config and isinstance(config["data"], dict) and "num_workers" in config["data"]:
        return int(config["data"]["num_workers"])
    return 4


def _canonical_class_name(name: str) -> str:
    """Normalize folder/class naming variations to a canonical form."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    return _CLASS_ALIASES.get(key, key)


def _read_expected_class_names(config: dict) -> list[str] | None:
    """Read canonical class names from config when provided."""
    model_cfg = config.get("model", {})
    class_names = model_cfg.get("classes")
    if not class_names:
        return None
    return [_canonical_class_name(name) for name in class_names]


def _is_explicit_split_layout(data_dir: str) -> bool:
    """Detect dataset layout containing top-level Training/Testing folders."""
    return os.path.isdir(os.path.join(data_dir, "Training")) and os.path.isdir(
        os.path.join(data_dir, "Testing")
    )


def _collect_paths_from_class_dirs(
    base_dir: str,
    class_names: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Collect image paths + canonical labels from direct class folders under base_dir."""
    image_paths: list[str] = []
    labels: list[str] = []

    if not os.path.isdir(base_dir):
        return image_paths, labels

    if class_names:
        class_dir_names = class_names
    else:
        class_dir_names = [
            _canonical_class_name(name)
            for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name))
        ]
        class_dir_names = sorted(set(class_dir_names))

    for class_name in class_dir_names:
        possible_dirs = [class_name]
        # Keep backward compatibility with legacy folder naming.
        if class_name == "no_tumor":
            possible_dirs.append("notumor")
        if class_name == "notumor":
            possible_dirs.append("no_tumor")

        class_dir = None
        for candidate in possible_dirs:
            candidate_dir = os.path.join(base_dir, candidate)
            if os.path.isdir(candidate_dir):
                class_dir = candidate_dir
                break
        if class_dir is None:
            continue

        for file_name in os.listdir(class_dir):
            ext = os.path.splitext(file_name)[1].lower()
            if ext in _VALID_IMAGE_EXTS:
                image_paths.append(os.path.join(class_dir, file_name))
                labels.append(class_name)

    return image_paths, labels


def get_data_loaders(
    data_dir: str, config: dict
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train/val/test dataloaders from either explicit or flat class layout."""
    expected_class_names = _read_expected_class_names(config)

    if _is_explicit_split_layout(data_dir):
        train_base = os.path.join(data_dir, "Training")
        test_base = os.path.join(data_dir, "Testing")
        train_val_paths, train_val_class_names = _collect_paths_from_class_dirs(
            train_base, class_names=expected_class_names
        )
        test_paths, test_class_names = _collect_paths_from_class_dirs(
            test_base, class_names=expected_class_names
        )
        class_folder_names = train_val_class_names + test_class_names
    else:
        train_val_paths, train_val_class_names = _collect_paths_from_class_dirs(
            data_dir, class_names=expected_class_names
        )
        test_paths = []
        test_class_names = []
        class_folder_names = train_val_class_names

    if not train_val_paths and not test_paths:
        raise ValueError(f"No images found under '{data_dir}'.")

    if expected_class_names:
        class_names = [name for name in expected_class_names if name in set(class_folder_names)]
    else:
        class_names = sorted(set(class_folder_names))
    label_encoder = {name: idx for idx, name in enumerate(class_names)}

    train_val_labels = [label_encoder[name] for name in train_val_class_names]
    test_labels = [label_encoder[name] for name in test_class_names] if test_class_names else []

    if _is_explicit_split_layout(data_dir):
        # Keep provided test split untouched; split only train into train/val.
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths,
            train_val_labels,
            test_size=0.15,
            random_state=42,
            stratify=train_val_labels,
        )
    else:
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            train_val_paths,
            train_val_labels,
            test_size=0.30,
            random_state=42,
            stratify=train_val_labels,
        )
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths,
            temp_labels,
            test_size=0.50,
            random_state=42,
            stratify=temp_labels,
        )

    img_size = _read_img_size(config)
    train_transform = get_train_transforms(img_size=img_size)
    eval_transform = get_val_transforms(img_size=img_size)

    train_dataset = BrainTumorDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = BrainTumorDataset(val_paths, val_labels, transform=eval_transform)
    test_dataset = BrainTumorDataset(test_paths, test_labels, transform=eval_transform)

    batch_size = _read_batch_size(config)
    requested_workers = _read_num_workers(config)
    # On macOS CPU/MPS runs, multiprocessing workers can leak semaphores and destabilize runs.
    if torch.cuda.is_available():
        num_workers = requested_workers
    else:
        num_workers = 0 if os.uname().sysname.lower() == "darwin" else requested_workers
    pin_memory = torch.cuda.is_available()

    use_weighted_sampler = bool(config.get("training", {}).get("use_weighted_sampler", True))
    sample_weights = None
    sampler = None
    if use_weighted_sampler and train_labels:
        counts = np.bincount(train_labels, minlength=len(class_names))
        counts = np.clip(counts, a_min=1, a_max=None)
        class_weights = 1.0 / counts
        sample_weights = class_weights[np.array(train_labels)]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_labels),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, class_names
