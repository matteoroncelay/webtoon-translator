"""Lecture et écriture d'images.

Piège Windows : cv2.imread/cv2.imwrite échouent EN SILENCE (renvoient None/False)
si le chemin contient des accents ou des caractères non ASCII
(ex. C:\\Users\\matte\\Téléchargements). On passe donc par numpy, qui gère
correctement les chemins Unicode.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _natural_key(path: Path) -> list:
    """Tri "naturel" : page_2 avant page_10 (le tri alphabétique ferait l'inverse)."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def list_images(folder: str | Path) -> list[Path]:
    """Liste les images d'un dossier, dans l'ordre de lecture."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Dossier introuvable : {folder}")
    images = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(images, key=_natural_key)


def read_image(path: str | Path) -> np.ndarray:
    """Lit une image au format OpenCV (BGR). Lève une erreur claire si impossible."""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Impossible de lire l'image : {path}")
    return image


def write_image(path: str | Path, image: np.ndarray, quality: int = 95) -> None:
    """Écrit une image (le dossier parent est créé si besoin)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in {".jpg", ".jpeg"} else []
    ok, buffer = cv2.imencode(ext, image, params)
    if not ok:
        raise ValueError(f"Impossible d'encoder l'image : {path}")
    buffer.tofile(str(path))
