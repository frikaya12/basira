"""Tri des examens à partir de l'avis de l'IA sur chaque œil.

Règle de prudence : un seul œil suspect suffit pour que l'examen soit
prioritaire. Si l'IA n'a pas donné d'avis (pas de modèle chargé), l'examen
est quand même lu par l'ophtalmologue. Dans tous les cas, c'est
l'ophtalmologue qui décide ; le tri sert seulement à ordonner sa file.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Optional


class EyeLabel(str, Enum):
    SUSPECT = "suspect"
    NORMAL = "normal"
    SANS_AVIS = "sans_avis"


class Triage(str, Enum):
    SUSPECT = "suspect"
    A_LIRE = "a_lire"
    NORMAL = "normal"


# Ordre de lecture dans la file de l'ophtalmologue (0 = en premier).
PRIORITY = {Triage.SUSPECT: 0, Triage.A_LIRE: 1, Triage.NORMAL: 2}


def label_eye(probability: Optional[float], threshold: float) -> EyeLabel:
    """Probabilité de rétinopathie « à adresser » -> étiquette pour un œil."""
    if probability is None:
        return EyeLabel.SANS_AVIS
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Probabilité hors de [0, 1] : {probability}")
    return EyeLabel.SUSPECT if probability >= threshold else EyeLabel.NORMAL


def combine(labels: Iterable[EyeLabel]) -> Triage:
    labels = list(labels)
    if not labels:
        raise ValueError("Au moins un œil est nécessaire.")
    if EyeLabel.SUSPECT in labels:
        return Triage.SUSPECT
    if EyeLabel.SANS_AVIS in labels:
        return Triage.A_LIRE
    return Triage.NORMAL
