from datetime import date

import pytest

from basira.messages import Decision, patient_message, rdv_date
from basira.triage import EyeLabel, Triage, combine, label_eye


def test_label_eye():
    assert label_eye(None, 0.5) is EyeLabel.SANS_AVIS
    assert label_eye(0.7, 0.5) is EyeLabel.SUSPECT
    assert label_eye(0.5, 0.5) is EyeLabel.SUSPECT
    assert label_eye(0.2, 0.5) is EyeLabel.NORMAL
    with pytest.raises(ValueError):
        label_eye(1.5, 0.5)


def test_un_seul_oeil_suspect_suffit():
    assert combine([EyeLabel.NORMAL, EyeLabel.SUSPECT]) is Triage.SUSPECT
    assert combine([EyeLabel.SANS_AVIS, EyeLabel.SUSPECT]) is Triage.SUSPECT


def test_sans_avis_donne_a_lire():
    assert combine([EyeLabel.NORMAL, EyeLabel.SANS_AVIS]) is Triage.A_LIRE
    assert combine([EyeLabel.NORMAL, EyeLabel.NORMAL]) is Triage.NORMAL


def test_message_rdv():
    exam = date(2026, 10, 28)
    rdv = rdv_date(exam)
    text = patient_message(Decision.RDV, exam, rdv)
    assert "28/10/2026" in text and "04/11/2026" in text


def test_message_normal_annonce_le_controle_dans_un_an():
    text = patient_message(Decision.NORMAL, date(2026, 10, 28))
    assert "octobre 2027" in text


def test_message_refaire():
    assert "refaire" in patient_message(Decision.REFAIRE, date(2026, 10, 28))


def test_rdv_sans_date_refuse():
    with pytest.raises(ValueError):
        patient_message(Decision.RDV, date(2026, 10, 28))
