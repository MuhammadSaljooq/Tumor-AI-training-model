"""Tests for checkpoint save/load and ROC alignment."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.training.callbacks import ModelCheckpoint
from src.utils.checkpoint_io import load_checkpoint, save_checkpoint
from src.utils.visualizer import plot_roc_curves


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestCheckpointIO(unittest.TestCase):
    def test_atomic_save_and_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pth"
            state = {"model_state_dict": {"fc.weight": torch.zeros(2, 4)}, "epoch": 3}
            save_checkpoint(path, state, atomic=True)
            loaded = load_checkpoint(path, map_location="cpu", weights_only=False)
            self.assertEqual(loaded["epoch"], 3)
            self.assertTrue(torch.equal(loaded["model_state_dict"]["fc.weight"], torch.zeros(2, 4)))


class TestModelCheckpoint(unittest.TestCase):
    def test_full_training_state_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            model = TinyNet()
            optimizer = AdamW(model.parameters(), lr=1e-3)
            scheduler = CosineAnnealingLR(optimizer, T_max=10)
            scaler = torch.amp.GradScaler("cuda", enabled=False)

            x = torch.randn(2, 4)
            y = torch.tensor([0, 1])
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            mc = ModelCheckpoint(save_dir=str(ckpt_dir), model_name="tiny", mode="min", min_delta=0.001)
            saved = mc.save_if_improved(
                0.5,
                1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                use_amp=False,
                training_snapshot={"lr": 1e-3},
            )
            self.assertTrue(saved)

            model2 = TinyNet()
            opt2 = AdamW(model2.parameters(), lr=1e-3)
            sched2 = CosineAnnealingLR(opt2, T_max=10)

            payload = load_checkpoint(mc.checkpoint_path, map_location="cpu", weights_only=False)
            model2.load_state_dict(payload["model_state_dict"])
            opt2.load_state_dict(payload["optimizer_state_dict"])
            sched2.load_state_dict(payload["scheduler_state_dict"])

            w0 = model.fc.weight.detach().cpu()
            w1 = model2.fc.weight.detach().cpu()
            self.assertTrue(torch.allclose(w0, w1))


class TestPlotRocCurves(unittest.TestCase):
    def test_mismatched_lengths_raise(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        scores_a = np.random.rand(4, 2).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "roc.png"
            with self.assertRaises(ValueError):
                plot_roc_curves(
                    y_true=y_true,
                    y_scores=[scores_a],
                    class_names=["a", "b"],
                    model_names=["m1", "m2"],
                    save_path=out,
                )


if __name__ == "__main__":
    unittest.main()
