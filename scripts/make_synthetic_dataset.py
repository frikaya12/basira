"""Crée un petit jeu de données synthétique au format APTOS, pour tester le pipeline.

ATTENTION : ces images sont dessinées par le programme. Un modèle entraîné
dessus n'apprend rien de médical ; il sert seulement à vérifier que le code
d'entraînement fonctionne de bout en bout, sans télécharger de données de santé.

    python -m scripts.make_synthetic_dataset --out data/synthetic --n 40
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from basira.synthetic import make_fundus


def make_dataset(out: str | Path, n: int = 40, seed: int = 0) -> Path:
    out = Path(out)
    images = out / "train_images"
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        sick = i % 2 == 1
        grade = int(rng.integers(2, 5)) if sick else int(rng.integers(0, 2))
        img = make_fundus("lesions" if sick else "saine", 1 if rng.random() < 0.5 else -1, seed=seed * 1000 + i)
        code = f"synth{i:04d}"
        cv2.imwrite(str(images / f"{code}.png"), img)
        rows.append((code, grade))
    csv_path = out / "train.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id_code", "diagnosis"])
        writer.writerows(rows)
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    a = parser.parse_args()
    print(f"Jeu synthétique écrit : {make_dataset(a.out, a.n, a.seed)}")
