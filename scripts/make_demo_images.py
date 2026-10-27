"""Écrit quelques images synthétiques de fond d'œil pour essayer l'API.

    python -m scripts.make_demo_images

Images créées dans data/demo/ : od_saine.png, og_saine.png, od_lesions.png,
og_lesions.png, og_floue.png. Ce sont des dessins, pas des photos de patients.
"""

from pathlib import Path

import cv2

from basira.synthetic import make_fundus

OUT = Path("data/demo")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, variant, side in [("od_saine", "saine", 1), ("og_saine", "saine", -1), ("od_lesions", "lesions", 1),
                                ("og_lesions", "lesions", -1), ("og_floue", "floue", -1)]:
        cv2.imwrite(str(OUT / f"{name}.png"), make_fundus(variant, side, seed=7))
    print(f"Images de démonstration écrites dans {OUT}/")


if __name__ == "__main__":
    main()
