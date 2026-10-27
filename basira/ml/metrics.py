"""Métriques d'un test de dépistage.

Pour un dépistage, la métrique la plus importante est la sensibilité : la
part des patients malades que l'outil détecte. Un malade non détecté (faux
négatif) est rassuré à tort et risque de perdre la vue. On choisit donc le
seuil de décision pour atteindre une sensibilité cible (par exemple 90 %),
puis on regarde la spécificité obtenue : la part des patients sains que l'on
n'envoie pas inutilement chez l'ophtalmologue.

Ce module n'utilise que NumPy pour rester testable sans PyTorch.
"""

from __future__ import annotations

import numpy as np


def roc_auc(y_true, scores) -> float | None:
    """Aire sous la courbe ROC (statistique de Mann-Whitney). None si une seule classe."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):  # rangs moyens en cas d'égalité
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def threshold_for_sensitivity(y_true, scores, target: float = 0.9) -> float:
    """Seuil le plus élevé qui atteint la sensibilité cible."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    pos = np.sort(s[y])[::-1]
    if pos.size == 0:
        return 0.5
    k = int(np.ceil(target * pos.size - 1e-9)) - 1  # il faut au moins k+1 positifs au-dessus du seuil
    return float(pos[max(0, min(k, pos.size - 1))])


def screening_metrics(y_true, scores, threshold: float) -> dict:
    y = np.asarray(y_true).astype(bool)
    pred = np.asarray(scores, dtype=float) >= threshold
    tp = int((pred & y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    fp = int((pred & ~y).sum())

    def ratio(a, b):
        return round(a / b, 4) if b else None

    return {
        "seuil": round(float(threshold), 4),
        "sensibilite": ratio(tp, tp + fn),
        "specificite": ratio(tn, tn + fp),
        "valeur_predictive_positive": ratio(tp, tp + fp),
        "valeur_predictive_negative": ratio(tn, tn + fn),
        "auc": None if roc_auc(y, scores) is None else round(roc_auc(y, scores), 4),
        "vrais_positifs": tp,
        "faux_negatifs": fn,
        "vrais_negatifs": tn,
        "faux_positifs": fp,
    }
