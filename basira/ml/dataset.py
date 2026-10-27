"""Jeu de données au format APTOS 2019 (Kaggle).

Format attendu :
- un fichier CSV avec les colonnes `id_code` et `diagnosis` ;
- un dossier d'images nommées `<id_code>.png` (ou .jpg / .jpeg).

`diagnosis` suit l'échelle internationale de la rétinopathie diabétique :
0 = aucune, 1 = minime, 2 = modérée, 3 = sévère, 4 = proliférante.
Pour le dépistage, on prédit si la rétinopathie est « à adresser » à
l'ophtalmologue, c'est-à-dire de stade 2 ou plus.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from basira.preprocessing import preprocess

REFERABLE_GRADE = 2
EXTENSIONS = (".png", ".jpg", ".jpeg")


def read_labels(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = {"id_code", "diagnosis"} - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {csv_path} : {sorted(missing)}")
    df = df[["id_code", "diagnosis"]].copy()
    df["id_code"] = df["id_code"].astype(str)
    df["referable"] = (df["diagnosis"] >= REFERABLE_GRADE).astype(int)
    return df


def find_image(images_dir: str | Path, id_code: str) -> Path:
    for ext in EXTENSIONS:
        p = Path(images_dir) / f"{id_code}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Image introuvable pour {id_code} dans {images_dir}")


def stratified_split(labels: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Découpe entraînement / validation en gardant la même proportion de malades."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return np.array(sorted(train_idx)), np.array(sorted(val_idx))


def to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float() / 255.0
    return (x - 0.5) / 0.25


def augment(image_rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Variations réalistes : la rétine peut être tournée ou retournée, l'éclairage varie."""
    if rng.random() < 0.5:
        image_rgb = image_rgb[:, ::-1]
    if rng.random() < 0.5:
        image_rgb = image_rgb[::-1, :]
    h, w = image_rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(0, 360)), 1.0)
    image_rgb = cv2.warpAffine(np.ascontiguousarray(image_rgb), matrix, (w, h), borderValue=(0, 0, 0))
    gain = rng.uniform(0.9, 1.1)
    return np.clip(image_rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8)


class FundusDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str | Path, size: int = 384, train: bool = False, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.size = size
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        image = cv2.imread(str(find_image(self.images_dir, row["id_code"])), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Image illisible : {row['id_code']}")
        x = preprocess(image, self.size)
        if self.train:
            x = augment(x, self.rng)
        return to_tensor(x), torch.tensor(float(row["referable"]))
