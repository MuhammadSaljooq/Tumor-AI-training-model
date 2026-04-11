"""Atomic checkpoint save/load for trusted local training artifacts."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, state: dict[str, Any], *, atomic: bool = True) -> None:
    """Persist ``state`` to ``path``. Uses a temp file + replace when ``atomic`` is True."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if atomic:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, tmp_path)
        tmp_path.replace(path)
    else:
        torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device | None = None,
    *,
    weights_only: bool | None = False,
) -> Any:
    """Load a checkpoint. Default ``weights_only=False``: full training dicts are trusted local files only."""
    path = Path(path)
    kwargs: dict[str, Any] = {}
    if map_location is not None:
        kwargs["map_location"] = map_location
    if weights_only is not None and "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = weights_only
    return torch.load(path, **kwargs)
