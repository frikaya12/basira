"""Test de bout en bout du pipeline IA sur données synthétiques (ignoré si PyTorch n'est pas installé)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("pandas")

from basira.classifier import load_classifier  # noqa: E402
from basira.ml import train  # noqa: E402
from basira.synthetic import make_fundus  # noqa: E402
from scripts.make_synthetic_dataset import make_dataset  # noqa: E402


def test_entrainement_et_inference(tmp_path):
    csv_path = make_dataset(tmp_path / "data", n=16)
    out = tmp_path / "model.pt"
    report = train.main([
        "--csv", str(csv_path), "--images", str(tmp_path / "data" / "train_images"), "--out", str(out),
        "--arch", "resnet18", "--no-pretrained", "--epochs", "1", "--batch-size", "4", "--img-size", "96", "--workers", "0",
    ])
    assert out.exists() and out.with_suffix(".json").exists()
    assert len(report["history"]) == 1

    clf = load_classifier(str(out))
    p = clf.predict(make_fundus("lesions", 1, 99))
    assert 0.0 <= p <= 1.0
    assert 0.0 <= clf.threshold <= 1.0
