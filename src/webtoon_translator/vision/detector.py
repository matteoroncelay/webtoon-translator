"""Détection YOLO (bulles et personnages) sur des pages découpées en panneaux.

Avant : deux scripts quasi identiques (detect_perso.py, detect_bulles_json.py).
Maintenant : une seule classe YoloDetector, et deux petites fonctions qui
convertissent ses résultats en Bubble / CharacterDetection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config, GutterConfig
from ..models import BBox, Bubble, BubbleKind, CharacterDetection
from .panels import find_gutters, panel_ranges


@dataclass(frozen=True)
class RawDetection:
    class_id: int
    bbox: BBox
    confidence: float


class YoloDetector:
    """Enveloppe un modèle YOLO. Le modèle n'est chargé qu'au premier usage
    (l'ancien tail_direction.py le chargeait dès l'import, même sans s'en servir)."""

    def __init__(self, model_path: str | Path, imgsz: int, iou: float, model: Any = None):
        self.model_path = Path(model_path)
        self.imgsz = imgsz
        self.iou = iou
        self._model = model  # injectable pour les tests

    @property
    def model(self) -> Any:
        if self._model is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"Modèle introuvable : {self.model_path}")
            from ultralytics import YOLO  # import tardif : lourd (PyTorch)
            self._model = YOLO(str(self.model_path))
        return self._model

    def detect(self, image: np.ndarray, conf: float, gutters: GutterConfig) -> list[RawDetection]:
        """Détecte sur toute la page, panneau par panneau, en coordonnées PAGE."""
        height, width = image.shape[:2]
        detections: list[RawDetection] = []

        for y1, y2 in panel_ranges(height, find_gutters(image, gutters), gutters):
            results = self.model(image[y1:y2], imgsz=self.imgsz, conf=conf,
                                 iou=self.iou, verbose=False)
            boxes = getattr(results[0], "boxes", None) if results else None
            if boxes is None:
                continue
            for box in boxes:
                bx1, by1, bx2, by2 = (float(v) for v in box.xyxy[0].tolist())
                # Recadrage dans le panneau, puis passage au repère de la page
                bx1, bx2 = max(0.0, bx1), min(float(width), bx2)
                by1, by2 = max(0.0, by1), min(float(y2 - y1), by2)
                if bx2 <= bx1 or by2 <= by1:
                    continue
                detections.append(RawDetection(
                    class_id=int(box.cls[0]),
                    bbox=BBox(bx1, by1, bx2, by2).shifted(dy=y1),
                    confidence=float(box.conf[0]),
                ))
        return detections


def detect_bubbles(image: np.ndarray, detector: YoloDetector, config: Config) -> list[Bubble]:
    raw = detector.detect(image, config.detection.bubble_conf, config.gutters)
    bubbles = []
    for det in raw:
        try:
            kind = BubbleKind(det.class_id)
        except ValueError:
            continue  # classe inconnue : on l'ignore plutôt que de planter
        bubbles.append(Bubble(id=0, bbox=det.bbox, kind=kind, confidence=det.confidence))
    return bubbles


def detect_characters(image: np.ndarray, detector: YoloDetector,
                      config: Config) -> list[CharacterDetection]:
    raw = detector.detect(image, config.detection.character_conf, config.gutters)
    return [CharacterDetection(d.class_id, d.bbox, d.confidence) for d in raw]
