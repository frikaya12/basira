"""Contrôle qualité d'une photo du fond d'œil.

Avant d'envoyer une photo à l'IA ou à l'ophtalmologue, on vérifie qu'elle est
exploitable : assez nette, ni trop sombre ni trop claire, et que la rétine
occupe une part suffisante de l'image. Une photo refusée doit être reprise
tout de suite, pendant que le patient est encore à la pharmacie.

Méthode :
- l'image est réduite à 256 x 256 pixels et convertie en niveaux de gris ;
- le « champ » (la rétine) est la zone plus claire que le fond noir ;
- la luminosité est la moyenne des niveaux de gris au centre du champ ;
- la netteté est l'écart type du laplacien au centre du champ : une image
  floue n'a presque plus de contours, donc un laplacien presque nul.

Les seuils par défaut sont provisoires : ils doivent être recalibrés sur de
vraies photos prises avec le rétinographe portable choisi.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

ANALYSIS_SIZE = 256
FIELD_LUMINANCE = 14  # au-dessus de ce niveau de gris, le pixel appartient à la rétine
INNER_RADIUS_RATIO = 0.8  # on ignore le bord du champ, où le contraste avec le noir fausse la netteté


@dataclass(frozen=True)
class QualityThresholds:
    min_sharpness: float = 6.0
    min_brightness: float = 45.0
    max_brightness: float = 215.0
    min_field_fraction: float = 0.20


DEFAULT_THRESHOLDS = QualityThresholds()


@dataclass
class QualityReport:
    ok: bool
    sharpness: float
    brightness: float
    field_fraction: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def decode_image(data: bytes) -> np.ndarray:
    """Décode des octets (JPEG, PNG…) en image BGR. Lève ValueError si illisible."""
    if not data:
        raise ValueError("Fichier vide.")
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image illisible : envoyez un fichier JPEG ou PNG.")
    return image


def assess_quality(image_bgr: np.ndarray, thresholds: QualityThresholds = DEFAULT_THRESHOLDS) -> QualityReport:
    """Mesure la qualité d'une photo du fond d'œil (image BGR, uint8)."""
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Image couleur attendue (hauteur x largeur x 3).")

    n = ANALYSIS_SIZE
    small = cv2.resize(image_bgr, (n, n), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    field_mask = gray > FIELD_LUMINANCE
    area = int(field_mask.sum())
    field_fraction = area / float(n * n)
    if area < 50:
        return QualityReport(False, 0.0, 0.0, field_fraction, ["Image trop sombre ou vide"])

    ys, xs = np.nonzero(field_mask)
    cy, cx = ys.mean(), xs.mean()
    radius = np.sqrt(area / np.pi) * INNER_RADIUS_RATIO
    yy, xx = np.mgrid[0:n, 0:n]
    inner = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False

    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=1)
    sharpness = float(laplacian[inner].std())
    brightness = float(gray[inner].mean())

    reasons: list[str] = []
    if field_fraction < thresholds.min_field_fraction:
        reasons.append("Rétine trop petite dans l'image")
    if brightness < thresholds.min_brightness:
        reasons.append("Photo trop sombre")
    if brightness > thresholds.max_brightness:
        reasons.append("Photo trop claire")
    if sharpness < thresholds.min_sharpness:
        reasons.append("Photo floue")

    return QualityReport(
        ok=not reasons,
        sharpness=round(sharpness, 2),
        brightness=round(brightness, 1),
        field_fraction=round(field_fraction, 3),
        reasons=reasons,
    )
