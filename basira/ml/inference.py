"""Utilisation d'un modèle entraîné par l'API."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from basira.ml.dataset import to_tensor
from basira.ml.model import build_model
from basira.preprocessing import preprocess


class TorchClassifier:
    def __init__(self, model: torch.nn.Module, img_size: int, threshold: float, name: str):
        self.model = model.eval()
        self.img_size = img_size
        self.threshold = threshold
        self.name = name

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "TorchClassifier":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = build_model(ckpt["arch"], pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, int(ckpt["img_size"]), float(ckpt["threshold"]), f"{ckpt['arch']}:{Path(path).name}")

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> Optional[float]:
        x = to_tensor(preprocess(image_bgr, self.img_size)).unsqueeze(0)
        return float(torch.sigmoid(self.model(x)).item())
