import pytest

from basira.ml.metrics import roc_auc, screening_metrics, threshold_for_sensitivity


def test_auc_parfaite_et_aleatoire():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert roc_auc([1, 1], [0.2, 0.3]) is None


def test_seuil_atteint_la_sensibilite_cible():
    y = [1] * 10 + [0] * 10
    scores = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.2] + [0.5, 0.4, 0.3, 0.25, 0.1, 0.1, 0.05, 0.05, 0.02, 0.01]
    t = threshold_for_sensitivity(y, scores, target=0.9)
    m = screening_metrics(y, scores, t)
    assert m["sensibilite"] >= 0.9
    assert t == pytest.approx(0.55)
    assert m["specificite"] == 1.0


def test_compte_des_erreurs():
    m = screening_metrics([1, 1, 0, 0], [0.9, 0.1, 0.8, 0.2], 0.5)
    assert (m["vrais_positifs"], m["faux_negatifs"], m["faux_positifs"], m["vrais_negatifs"]) == (1, 1, 1, 1)
    assert m["sensibilite"] == 0.5 and m["specificite"] == 0.5
