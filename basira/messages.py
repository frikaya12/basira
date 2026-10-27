"""Messages envoyés au patient après la décision de l'ophtalmologue.

Dans la v0.1, les messages sont seulement générés et stockés : aucun SMS
n'est envoyé. Le branchement d'un service SMS ou WhatsApp est prévu (v0.3).
"""

from __future__ import annotations

from datetime import date, timedelta
from enum import Enum

MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]

RDV_DELAI_JOURS = 7
CONTROLE_DELAI_JOURS = 365


class Decision(str, Enum):
    RDV = "rdv"  # rétinopathie : rendez-vous prioritaire
    NORMAL = "normal"  # pas de rétinopathie : contrôle dans 12 mois
    REFAIRE = "refaire"  # photo à refaire


def fmt_date(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def fmt_mois(d: date) -> str:
    return f"{MOIS[d.month - 1]} {d.year}"


def rdv_date(decision_day: date) -> date:
    return decision_day + timedelta(days=RDV_DELAI_JOURS)


def patient_message(decision: Decision, exam_day: date, rdv_day: date | None = None) -> str:
    decision = Decision(decision)
    if decision is Decision.RDV:
        if rdv_day is None:
            raise ValueError("Une date de rendez-vous est nécessaire.")
        return (
            f"Basira : bonjour, votre examen des yeux du {fmt_date(exam_day)} montre des signes à faire vérifier "
            f"par un ophtalmologue. Rendez-vous réservé le {fmt_date(rdv_day)} au cabinet d'ophtalmologie partenaire. "
            "Répondez OUI pour confirmer."
        )
    if decision is Decision.NORMAL:
        prochain = exam_day + timedelta(days=CONTROLE_DELAI_JOURS)
        return (
            f"Basira : bonjour, votre examen des yeux du {fmt_date(exam_day)} est normal. "
            f"Prochain contrôle conseillé : {fmt_mois(prochain)}. Nous vous enverrons un rappel."
        )
    return (
        f"Basira : bonjour, une de vos photos des yeux du {fmt_date(exam_day)} n'est pas assez nette. "
        "Repassez à votre pharmacie quand vous voulez pour la refaire, c'est gratuit."
    )
