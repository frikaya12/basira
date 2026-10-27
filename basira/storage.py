"""Stockage des dépistages dans une base SQLite locale.

Prototype : une base SQLite et un dossier de photos sur le disque. Avant tout
usage avec de vrais patients, il faudra un stockage chiffré, un contrôle des
accès et une déclaration à la CNDP (loi 09-08 sur les données personnelles).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS screenings (
    num INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    initiales TEXT NOT NULL,
    age INTEGER NOT NULL,
    duree_diabete INTEGER NOT NULL,
    type_diabete TEXT NOT NULL,
    telephone TEXT NOT NULL,
    photo_od TEXT NOT NULL,
    photo_og TEXT NOT NULL,
    qualite_od TEXT NOT NULL,
    qualite_og TEXT NOT NULL,
    modele TEXT NOT NULL,
    proba_od REAL,
    proba_og REAL,
    avis_od TEXT NOT NULL,
    avis_og TEXT NOT NULL,
    tri TEXT NOT NULL,
    decision TEXT,
    note TEXT,
    decided_at TEXT,
    rdv_date TEXT,
    message TEXT,
    vu_ophtalmo INTEGER NOT NULL DEFAULT 0
);
"""


def code_from_num(num: int) -> str:
    return f"BAS-{num:04d}"


def num_from_code(code: str) -> Optional[int]:
    if not code.startswith("BAS-"):
        return None
    try:
        return int(code[4:])
    except ValueError:
        return None


@dataclass
class NewScreening:
    initiales: str
    age: int
    duree_diabete: int
    type_diabete: str
    telephone: str
    photo_od: str
    photo_og: str
    qualite_od: dict
    qualite_og: dict
    modele: str
    proba_od: Optional[float]
    proba_og: Optional[float]
    avis_od: str
    avis_og: str
    tri: str


class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def insert(self, s: NewScreening, now: Optional[datetime] = None) -> str:
        now = now or datetime.now()
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO screenings (created_at, initiales, age, duree_diabete, type_diabete, telephone,
                   photo_od, photo_og, qualite_od, qualite_og, modele, proba_od, proba_og, avis_od, avis_og, tri)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now.isoformat(timespec="seconds"), s.initiales, s.age, s.duree_diabete, s.type_diabete, s.telephone,
                    s.photo_od, s.photo_og, json.dumps(s.qualite_od), json.dumps(s.qualite_og), s.modele,
                    s.proba_od, s.proba_og, s.avis_od, s.avis_og, s.tri,
                ),
            )
            return code_from_num(cur.lastrowid)

    def get(self, code: str) -> Optional[dict]:
        num = num_from_code(code)
        if num is None:
            return None
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM screenings WHERE num = ?", (num,)).fetchone()
        return _row_to_dict(row) if row else None

    def list_all(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM screenings ORDER BY num").fetchall()
        return [_row_to_dict(r) for r in rows]

    def decide(self, code: str, decision: str, note: str, decided_at: datetime, rdv: Optional[date], message: str) -> bool:
        """Enregistre la décision. Renvoie False si l'examen était déjà décidé."""
        num = num_from_code(code)
        with self._conn() as conn:
            cur = conn.execute(
                """UPDATE screenings SET decision = ?, note = ?, decided_at = ?, rdv_date = ?, message = ?
                   WHERE num = ? AND decision IS NULL""",
                (decision, note, decided_at.isoformat(timespec="seconds"), rdv.isoformat() if rdv else None, message, num),
            )
            return cur.rowcount == 1

    def set_seen(self, code: str, seen: bool) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE screenings SET vu_ophtalmo = ? WHERE num = ?", (1 if seen else 0, num_from_code(code)))


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["id"] = code_from_num(d.pop("num"))
    d["qualite_od"] = json.loads(d["qualite_od"])
    d["qualite_og"] = json.loads(d["qualite_og"])
    d["vu_ophtalmo"] = bool(d["vu_ophtalmo"])
    return d
