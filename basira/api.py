"""API Basira (FastAPI).

Parcours couvert par la v0.1 :
1. la pharmacie envoie les deux photos et les informations du patient ;
2. l'API contrôle la qualité des photos et refuse celles qui sont inexploitables ;
3. l'IA (si un modèle est chargé) donne un avis par œil, puis l'examen est trié ;
4. l'ophtalmologue lit sa file, priorisée, et décide ;
5. un message est généré pour le patient ;
6. on suit la part des patients à risque qui ont vu l'ophtalmologue.

Lancer : uvicorn basira.server:app --reload   puis ouvrir http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Literal, Optional

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from basira import __version__
from basira.classifier import Classifier, NoModel
from basira.messages import Decision, patient_message, rdv_date
from basira.quality import DEFAULT_THRESHOLDS, QualityThresholds, assess_quality, decode_image
from basira.storage import NewScreening, Store
from basira.triage import PRIORITY, Triage, combine, label_eye

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
EYES = ("od", "og")  # OD : œil droit, OG : œil gauche


class DecisionIn(BaseModel):
    decision: Decision
    note: str = Field(default="", max_length=500)


class SeenIn(BaseModel):
    vu: bool


def _public(s: dict) -> dict:
    """Vue d'un examen pour l'ophtalmologue (sans le téléphone du patient)."""
    return {
        "id": s["id"],
        "cree_le": s["created_at"],
        "patient": {
            "initiales": s["initiales"],
            "age": s["age"],
            "duree_diabete": s["duree_diabete"],
            "type_diabete": s["type_diabete"],
        },
        "qualite": {"od": s["qualite_od"], "og": s["qualite_og"]},
        "ia": {
            "modele": s["modele"],
            "od": {"avis": s["avis_od"], "probabilite": s["proba_od"]},
            "og": {"avis": s["avis_og"], "probabilite": s["proba_og"]},
        },
        "tri": s["tri"],
        "decision": s["decision"],
        "note": s["note"],
        "decide_le": s["decided_at"],
        "rdv": s["rdv_date"],
        "vu_ophtalmo": s["vu_ophtalmo"],
        "photos": {eye: f"/screenings/{s['id']}/photos/{eye}" for eye in EYES},
    }


def create_app(
    db_path: str | Path = "data/basira.db",
    upload_dir: str | Path = "data/photos",
    classifier: Optional[Classifier] = None,
    thresholds: QualityThresholds = DEFAULT_THRESHOLDS,
    today: Callable[[], date] = date.today,
) -> FastAPI:
    store = Store(db_path)
    photos = Path(upload_dir)
    photos.mkdir(parents=True, exist_ok=True)
    model = classifier or NoModel()

    app = FastAPI(
        title="Basira API",
        version=__version__,
        description="Dépistage de la rétinopathie diabétique en pharmacie. Prototype de recherche, pas un dispositif médical.",
    )

    def _get_or_404(code: str) -> dict:
        s = store.get(code)
        if s is None:
            raise HTTPException(status_code=404, detail=f"Examen {code} introuvable.")
        return s

    @app.get("/health", tags=["système"])
    def health() -> dict:
        return {"statut": "ok", "version": __version__, "modele_ia": model.name}

    @app.post("/screenings", status_code=201, tags=["1. pharmacie"])
    async def create_screening(
        initiales: str = Form(..., min_length=1, max_length=6),
        age: int = Form(..., ge=10, le=110),
        duree_diabete: int = Form(..., ge=0, le=90, description="Ancienneté du diabète, en années"),
        type_diabete: Literal["1", "2"] = Form(...),
        telephone: str = Form(..., min_length=6, max_length=20),
        consentement: bool = Form(..., description="Le patient accepte l'analyse par IA et la lecture à distance"),
        photo_od: UploadFile = File(..., description="Photo du fond d'œil droit"),
        photo_og: UploadFile = File(..., description="Photo du fond d'œil gauche"),
    ) -> dict:
        if not consentement:
            raise HTTPException(status_code=400, detail="Le consentement du patient est obligatoire.")

        images, reports = {}, {}
        for eye, upload in (("od", photo_od), ("og", photo_og)):
            data = await upload.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"Photo {eye.upper()} trop lourde (15 Mo maximum).")
            try:
                images[eye] = decode_image(data)
            except ValueError as err:
                raise HTTPException(status_code=422, detail=f"Photo {eye.upper()} : {err}") from err
            reports[eye] = assess_quality(images[eye], thresholds)

        if not all(r.ok for r in reports.values()):
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Photo à reprendre avant l'envoi.",
                    "qualite": {eye: r.to_dict() for eye, r in reports.items()},
                },
            )

        probas, avis, files = {}, {}, {}
        for eye, image in images.items():
            probas[eye] = model.predict(image)
            avis[eye] = label_eye(probas[eye], model.threshold)
            files[eye] = f"{uuid.uuid4().hex}.jpg"
            cv2.imwrite(str(photos / files[eye]), image, [cv2.IMWRITE_JPEG_QUALITY, 92])

        tri = combine(avis.values())
        code = store.insert(
            NewScreening(
                initiales=initiales.strip(), age=age, duree_diabete=duree_diabete, type_diabete=type_diabete,
                telephone=telephone.strip(), photo_od=files["od"], photo_og=files["og"],
                qualite_od=reports["od"].to_dict(), qualite_og=reports["og"].to_dict(), modele=model.name,
                proba_od=probas["od"], proba_og=probas["og"], avis_od=avis["od"].value, avis_og=avis["og"].value,
                tri=tri.value,
            )
        )
        return _public(store.get(code))

    @app.get("/queue", tags=["2. ophtalmologue"])
    def queue() -> list[dict]:
        """Examens à lire, les cas suspects en premier, puis par ancienneté."""
        pending = [s for s in store.list_all() if s["decision"] is None]
        pending.sort(key=lambda s: (PRIORITY[Triage(s["tri"])], s["created_at"]))
        return [_public(s) for s in pending]

    @app.get("/screenings/{code}", tags=["2. ophtalmologue"])
    def get_screening(code: str) -> dict:
        return _public(_get_or_404(code))

    @app.get("/screenings/{code}/photos/{eye}", tags=["2. ophtalmologue"])
    def get_photo(code: str, eye: Literal["od", "og"]) -> FileResponse:
        s = _get_or_404(code)
        return FileResponse(photos / s[f"photo_{eye}"], media_type="image/jpeg")

    @app.post("/screenings/{code}/decision", tags=["2. ophtalmologue"])
    def decide(code: str, body: DecisionIn) -> dict:
        s = _get_or_404(code)
        day = today()
        rdv = rdv_date(day) if body.decision is Decision.RDV else None
        exam_day = datetime.fromisoformat(s["created_at"]).date()
        message = patient_message(body.decision, exam_day, rdv)
        if not store.decide(code, body.decision.value, body.note, datetime.combine(day, datetime.now().time()), rdv, message):
            raise HTTPException(status_code=409, detail=f"L'examen {code} a déjà été lu.")
        return _public(store.get(code))

    @app.get("/messages", tags=["3. patient"])
    def messages() -> list[dict]:
        """Messages prêts à envoyer (aucun envoi réel dans la v0.1)."""
        decided = [s for s in store.list_all() if s["decision"]]
        decided.sort(key=lambda s: s["decided_at"], reverse=True)
        return [{"id": s["id"], "telephone": s["telephone"], "texte": s["message"], "envoye": False} for s in decided]

    @app.post("/screenings/{code}/vu", tags=["3. patient"])
    def mark_seen(code: str, body: SeenIn) -> dict:
        s = _get_or_404(code)
        if s["decision"] != Decision.RDV.value:
            raise HTTPException(status_code=409, detail="Seuls les patients orientés vers l'ophtalmologue peuvent être marqués comme vus.")
        store.set_seen(code, body.vu)
        return _public(store.get(code))

    @app.get("/stats", tags=["4. suivi"])
    def stats() -> dict:
        """Indicateur clé : part des patients à risque qui ont vu l'ophtalmologue (69 % à Fès, 2012-2013)."""
        all_s = store.list_all()
        risk = [s for s in all_s if s["decision"] == Decision.RDV.value]
        seen = sum(1 for s in risk if s["vu_ophtalmo"])
        return {
            "depistes": len(all_s),
            "suspects_ia": sum(1 for s in all_s if s["tri"] == Triage.SUSPECT.value),
            "lus": sum(1 for s in all_s if s["decision"]),
            "a_risque": len(risk),
            "vus_par_ophtalmo": seen,
            "taux_suivi": round(seen / len(risk), 3) if risk else None,
            "reference_fes_2013": 0.69,
        }

    return app
