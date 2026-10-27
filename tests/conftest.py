from datetime import date

import pytest
from fastapi.testclient import TestClient

from basira.api import create_app
from basira.synthetic import encode_png, make_fundus


class FixedClassifier:
    """Faux modèle pour les tests : renvoie une probabilité choisie à l'avance."""

    name = "faux-modele"
    threshold = 0.5

    def __init__(self, probability):
        self.probability = probability

    def predict(self, image_bgr):
        return self.probability


@pytest.fixture
def png():
    def _png(variant="saine", side=1, seed=0):
        return encode_png(make_fundus(variant, side, seed))

    return _png


@pytest.fixture
def make_client(tmp_path):
    def _make(classifier=None, today=date(2026, 10, 28)):
        app = create_app(tmp_path / "test.db", tmp_path / "photos", classifier=classifier, today=lambda: today)
        return TestClient(app)

    return _make


@pytest.fixture
def post_screening(png):
    def _post(client, od="saine", og="saine", consentement="true", **overrides):
        data = {
            "initiales": "M.B.",
            "age": "61",
            "duree_diabete": "9",
            "type_diabete": "2",
            "telephone": "0600000000",
            "consentement": consentement,
        }
        data.update(overrides)
        files = {
            "photo_od": ("od.png", png(od, 1), "image/png"),
            "photo_og": ("og.png", png(og, -1), "image/png"),
        }
        return client.post("/screenings", data=data, files=files)

    return _post
