"""Interface entre l'application et le modèle d'IA.

L'application ne dépend pas directement de PyTorch : elle appelle un objet
qui renvoie, pour une photo, la probabilité d'une rétinopathie « à adresser »
(stade modéré ou plus), ou None quand aucun modèle n'est chargé.
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np


class Classifier(Protocol):
    name: str
    threshold: float

    def predict(self, image_bgr: np.ndarray) -> Optional[float]: ...


class NoModel:
    """Aucun modèle chargé : chaque examen est envoyé à l'ophtalmologue sans avis de l'IA."""

    name = "aucun-modele"
    threshold = 0.5

    def predict(self, image_bgr: np.ndarray) -> Optional[float]:
        return None


def load_classifier(checkpoint_path: Optional[str]) -> Classifier:
    if not checkpoint_path:
        return NoModel()
    from basira.ml.inference import TorchClassifier  # import tardif : PyTorch reste optionnel

    return TorchClassifier.from_checkpoint(checkpoint_path)
