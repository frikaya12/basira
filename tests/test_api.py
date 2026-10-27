from conftest import FixedClassifier


def test_health(make_client):
    r = make_client().get("/health")
    assert r.status_code == 200
    assert r.json()["modele_ia"] == "aucun-modele"


def test_parcours_complet_sans_modele(make_client, post_screening):
    client = make_client()
    r = post_screening(client)
    assert r.status_code == 201, r.text
    exam = r.json()
    assert exam["id"] == "BAS-0001"
    assert exam["tri"] == "a_lire"  # aucun modèle : l'ophtalmologue lit quand même
    assert exam["qualite"]["od"]["ok"] and exam["qualite"]["og"]["ok"]
    assert "telephone" not in exam  # le numéro n'apparaît pas côté ophtalmologue

    queue = client.get("/queue").json()
    assert [e["id"] for e in queue] == ["BAS-0001"]

    photo = client.get("/screenings/BAS-0001/photos/od")
    assert photo.status_code == 200 and photo.headers["content-type"] == "image/jpeg"

    d = client.post("/screenings/BAS-0001/decision", json={"decision": "rdv", "note": "microanévrismes OD"})
    assert d.status_code == 200
    assert d.json()["rdv"] == "2026-11-04"
    assert client.get("/queue").json() == []

    msgs = client.get("/messages").json()
    assert msgs[0]["telephone"] == "0600000000"
    assert "04/11/2026" in msgs[0]["texte"] and msgs[0]["envoye"] is False

    assert client.post("/screenings/BAS-0001/vu", json={"vu": True}).status_code == 200
    stats = client.get("/stats").json()
    assert stats["a_risque"] == 1 and stats["vus_par_ophtalmo"] == 1 and stats["taux_suivi"] == 1.0


def test_photo_floue_bloque_l_envoi(make_client, post_screening):
    client = make_client()
    r = post_screening(client, og="floue")
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["qualite"]["og"]["ok"] is False
    assert "Photo floue" in detail["qualite"]["og"]["reasons"]
    assert client.get("/queue").json() == []


def test_consentement_obligatoire(make_client, post_screening):
    r = post_screening(make_client(), consentement="false")
    assert r.status_code == 400


def test_fichier_illisible(make_client, png):
    client = make_client()
    r = client.post(
        "/screenings",
        data={"initiales": "A", "age": "50", "duree_diabete": "3", "type_diabete": "2", "telephone": "0600000000", "consentement": "true"},
        files={"photo_od": ("a.txt", b"pas une image", "text/plain"), "photo_og": ("b.png", png(), "image/png")},
    )
    assert r.status_code == 422


def test_age_invalide(make_client, post_screening):
    assert post_screening(make_client(), age="5").status_code == 422


def test_suspects_en_premier(tmp_path, post_screening):
    from datetime import date

    from fastapi.testclient import TestClient

    from basira.api import create_app

    normal_model = FixedClassifier(0.1)
    suspect_model = FixedClassifier(0.9)
    # Deux applications sur la même base : l'une voit des yeux normaux, l'autre des yeux suspects.
    a = TestClient(create_app(tmp_path / "db.sqlite", tmp_path / "ph", classifier=normal_model, today=lambda: date(2026, 10, 28)))
    b = TestClient(create_app(tmp_path / "db.sqlite", tmp_path / "ph", classifier=suspect_model, today=lambda: date(2026, 10, 28)))
    assert post_screening(a).json()["tri"] == "normal"
    assert post_screening(b).json()["tri"] == "suspect"
    queue = a.get("/queue").json()
    assert [e["tri"] for e in queue] == ["suspect", "normal"]


def test_double_decision_refusee(make_client, post_screening):
    client = make_client(classifier=FixedClassifier(0.9))
    post_screening(client)
    assert client.post("/screenings/BAS-0001/decision", json={"decision": "normal"}).status_code == 200
    assert client.post("/screenings/BAS-0001/decision", json={"decision": "rdv"}).status_code == 409


def test_vu_reserve_aux_patients_orientes(make_client, post_screening):
    client = make_client()
    post_screening(client)
    client.post("/screenings/BAS-0001/decision", json={"decision": "normal"})
    assert client.post("/screenings/BAS-0001/vu", json={"vu": True}).status_code == 409


def test_examen_inconnu(make_client):
    client = make_client()
    assert client.get("/screenings/BAS-9999").status_code == 404
    assert client.get("/screenings/n-importe-quoi").status_code == 404
