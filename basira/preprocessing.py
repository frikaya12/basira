"""Prétraitement des photos du fond d'œil avant l'IA.

Étapes classiques pour les images de rétine (utilisées notamment sur la base
publique APTOS 2019) :
1. recadrer sur la rétine en supprimant le fond noir ;
2. compléter en carré puis redimensionner ;
3. normaliser l'éclairage (méthode de Ben Graham) : on soustrait une version
   très floue de l'image pour faire ressortir les petits détails (vaisseaux,
   micro-anévrismes, exsudats), quelle que soit la caméra ;
4. masquer le bord du cercle, où la normalisation crée des artefacts.
"""

from __future__ import annotations

import cv2
import numpy as np

from basira.quality import FIELD_LUMINANCE


def crop_to_field(image_bgr: np.ndarray, threshold: int = FIELD_LUMINANCE) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(gray > threshold)
    if ys.size == 0:
        return image_bgr
    return image_bgr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def pad_to_square(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    left = (side - w) // 2
    return cv2.copyMakeBorder(image_bgr, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def ben_graham(image_bgr: np.ndarray, sigma_ratio: float = 1 / 30) -> np.ndarray:
    sigma = max(1.0, image_bgr.shape[0] * sigma_ratio)
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(image_bgr, 4, blurred, -4, 128)


def circular_mask(image: np.ndarray, ratio: float = 0.95) -> np.ndarray:
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) / 2 * ratio), 1, -1)
    return image * mask[..., None]


def preprocess(image_bgr: np.ndarray, size: int = 384) -> np.ndarray:
    """Image BGR quelconque -> image RGB carrée (size x size), prête pour le modèle."""
    img = pad_to_square(crop_to_field(image_bgr))
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = circular_mask(ben_graham(img))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
