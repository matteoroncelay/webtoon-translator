"""Lecture du texte coréen des bulles (OCR).

Avant : l'OCR était fait DEUX fois (dans prepare_bubbles_json.py pour les
bulles à queue, puis dans run_ocr.py pour les autres), avec deux copies du code.
Maintenant : une seule passe, sur toutes les bulles qui n'ont pas encore de texte.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from ..models import Page


class OcrEngine(Protocol):
    def read(self, crop_bgr: np.ndarray) -> str: ...


class GoogleVisionOcr:
    """OCR Google Cloud Vision. Les identifiants sont lus depuis
    GOOGLE_APPLICATION_CREDENTIALS (voir .env.example)."""

    def __init__(self) -> None:
        from google.cloud import vision
        self._vision = vision
        self._client = vision.ImageAnnotatorClient()

    def read(self, crop_bgr: np.ndarray) -> str:
        ok, buffer = cv2.imencode(".png", crop_bgr)
        if not ok:
            return ""
        response = self._client.text_detection(image=self._vision.Image(content=buffer.tobytes()))
        if response.error.message:
            raise RuntimeError(f"Google Vision : {response.error.message}")
        annotations = response.text_annotations
        return annotations[0].description.strip() if annotations else ""


MIN_SIZE = 10  # px : en dessous, la bulle est trop petite pour contenir du texte lisible


def read_page_text(page_bgr: np.ndarray, page: Page, engine: OcrEngine) -> int:
    """Remplit `text_ko` des bulles vides. Renvoie le nombre de bulles lues."""
    height, width = page_bgr.shape[:2]
    count = 0
    for bubble in page.bubbles:
        if bubble.text_ko:
            continue
        b = bubble.bbox
        x1, y1 = max(0, int(b.x1)), max(0, int(b.y1))
        x2, y2 = min(width, int(b.x2)), min(height, int(b.y2))
        if x2 - x1 < MIN_SIZE or y2 - y1 < MIN_SIZE:
            continue
        # Les retours à la ligne de l'OCR suivent la mise en page, pas le sens : on les retire
        bubble.text_ko = " ".join(engine.read(page_bgr[y1:y2, x1:x2]).split())
        count += bool(bubble.text_ko)
    return count
