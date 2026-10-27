"""Images synthétiques de fond d'œil, pour les tests et les démonstrations.

Ces images sont dessinées par le programme : ce ne sont pas des photos de
patients. Elles imitent grossièrement une rétine (vaisseaux, papille, macula)
et, en option, des lésions (micro-hémorragies rouges, exsudats jaunes) ou un
flou de mise au point. Elles servent à tester le code sans données de santé.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

VARIANTS = ("saine", "lesions", "floue")
SIZE = 512
CENTER = (256, 256)
FIELD_RADIUS = 240


def _radial_gradient(size: int, center: tuple[float, float], radius: float, stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / radius
    dist = np.clip(dist, 0.0, 1.0)
    positions = [s[0] for s in stops]
    out = np.zeros((size, size, 3), dtype=np.float32)
    for channel in range(3):
        values = [s[1][channel] for s in stops]
        out[..., channel] = np.interp(dist, positions, values)
    return out


def _draw_vessel(img: np.ndarray, rng: np.random.Generator, x: float, y: float, angle: float, width: float, length: float, depth: int, color: tuple[int, int, int]) -> None:
    steps = 14
    seg = length / steps
    for i in range(steps):
        angle += (rng.random() - 0.5) * 0.35
        nx, ny = x + math.cos(angle) * seg, y + math.sin(angle) * seg
        thickness = max(1, int(round(width * (1 - i / steps * 0.6))))
        cv2.line(img, (int(x), int(y)), (int(nx), int(ny)), color, thickness, cv2.LINE_AA)
        x, y = nx, ny
        if depth > 0 and i > 3 and rng.random() < 0.16:
            branch = angle + (-1 if rng.random() < 0.5 else 1) * (0.5 + rng.random() * 0.5)
            _draw_vessel(img, rng, x, y, branch, width * 0.6, length * 0.55, depth - 1, color)


def make_fundus(variant: str = "saine", side: int = 1, seed: int = 0) -> np.ndarray:
    """Renvoie une image BGR 512 x 512 (uint8).

    variant : "saine", "lesions" ou "floue".
    side : 1 pour l'œil droit (papille à droite), -1 pour l'œil gauche.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Variante inconnue : {variant!r}. Choix : {VARIANTS}")
    rng = np.random.default_rng(seed)

    # Fond orangé de la rétine (couleurs en RGB, converties en BGR à la fin).
    rgb = _radial_gradient(SIZE, (256, 250), 250, [(0.0, (229, 138, 69)), (0.6, (196, 82, 31)), (1.0, (110, 42, 18))])

    # Macula : zone plus sombre au centre.
    mx, my = 256 - side * 30, 262
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    mac = np.clip(1 - np.sqrt((xx - mx) ** 2 + (yy - my) ** 2) / 46.0, 0, 1) * 0.75
    rgb = rgb * (1 - mac[..., None]) + np.array([110, 34, 12], dtype=np.float32) * mac[..., None]

    bgr = np.ascontiguousarray(rgb[..., ::-1]).astype(np.uint8)

    # Vaisseaux partant de la papille.
    dx, dy = 256 + side * 110, 248
    for v in range(9):
        base = math.pi if side > 0 else 0.0
        angle = base + (v / 8 - 0.5) * 3.4 + (rng.random() - 0.5) * 0.2
        color = (11, 20, 94) if v % 2 else (16, 30, 122)  # BGR
        _draw_vessel(bgr, rng, dx, dy, angle, 6 if v % 2 else 4.5, 210 + rng.random() * 60, 2, color)

    # Papille (disque optique) : tache claire.
    disc = np.clip(1 - np.sqrt((xx - dx) ** 2 + (yy - dy) ** 2) / 36.0, 0, 1)
    disc_alpha = np.clip(disc / 0.3, 0, 1)[..., None]
    disc_color = np.array([200, 240, 255], dtype=np.float32)  # BGR clair
    bgr = (bgr.astype(np.float32) * (1 - disc_alpha) + disc_color * disc_alpha).astype(np.uint8)

    if variant == "lesions":
        for _ in range(26):  # micro-hémorragies
            r, t = 40 + rng.random() * 120, rng.random() * 2 * math.pi
            cv2.circle(bgr, (int(mx + math.cos(t) * r), int(my + math.sin(t) * r)), int(2 + rng.random() * 4), (8, 12, 107), -1, cv2.LINE_AA)
        for _ in range(14):  # exsudats
            r, t = 30 + rng.random() * 80, rng.random() * 2 * math.pi
            cv2.circle(bgr, (int(mx + math.cos(t) * r), int(my + math.sin(t) * r)), int(3 + rng.random() * 4), (138, 224, 247), -1, cv2.LINE_AA)

    # Tout ce qui est hors du champ circulaire est noir.
    outside = (xx - CENTER[0]) ** 2 + (yy - CENTER[1]) ** 2 > FIELD_RADIUS**2
    bgr[outside] = (5, 5, 5)

    if variant == "floue":
        tiny = cv2.resize(bgr, (36, 36), interpolation=cv2.INTER_AREA)
        bgr = cv2.resize(tiny, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        bgr = (bgr.astype(np.float32) * 0.5).astype(np.uint8)

    return bgr


def encode_png(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Encodage PNG impossible.")
    return buf.tobytes()
