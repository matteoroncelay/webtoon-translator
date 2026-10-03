"""Analyse de la queue des bulles : position de la pointe et direction.

Principe (inchangé par rapport à ton tail_direction.py) :
  1. obtenir le contour de la bulle (segmentation YOLO, sinon binarisation) ;
  2. la pointe = le point du contour où l'angle est le plus aigu ;
  3. la base = les deux points de part et d'autre les plus proches du centre ;
  4. direction = vecteur base -> pointe.

Corrections apportées (cause probable des "contours ratés") :
  A. On recadre avec une MARGE (crop_padding) : la boîte YOLO de la bulle
     n'englobe souvent pas toute la queue, qui était donc coupée. Si la pointe
     touche encore le bord du recadrage, on recommence avec une marge doublée.
  B. On RÉÉCHANTILLONNE le contour à pas constant (resample_step px) : l'ancien
     `delta = 20` comptait des POINTS, or le contour YOLO-SEG a peu de points et
     le contour OpenCV en a des centaines -> l'angle mesuré n'avait pas le même sens.
  C. On choisit le contour qui CONTIENT le centre de la bulle (et non le plus
     grand, qui pouvait être une bulle voisine incluse dans le recadrage).
  D. La pointe doit être un sommet CONVEXE : l'ancien code pouvait prendre
     le creux à la jonction bulle/queue, qui est aussi un angle aigu.
  E. Le secours par binarisation essaie aussi l'image inversée (bulles sombres).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from ..config import TailConfig
from ..models import BBox


@dataclass(frozen=True)
class TailInfo:
    """Résultat en coordonnées PAGE."""

    tip: tuple[float, float]
    base_center: tuple[float, float]
    angle_deg: float        # direction base -> pointe, 0° = vers la droite, 90° = vers le bas
    tip_angle_deg: float    # angle au sommet de la pointe (plus petit = plus pointu)
    contour: np.ndarray     # contour utilisé (pour le débogage visuel)


class Segmenter(Protocol):
    def polygons(self, crop_bgr: np.ndarray) -> list[np.ndarray]: ...


class YoloSegmenter:
    """Modèle YOLO de segmentation des bulles (chargé au premier usage)."""

    def __init__(self, model_path: str | Path, cfg: TailConfig, model: Any = None):
        self.model_path = Path(model_path)
        self.cfg = cfg
        self._model = model

    def polygons(self, crop_bgr: np.ndarray) -> list[np.ndarray]:
        if self._model is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"Modèle introuvable : {self.model_path}")
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path))
        results = self._model(crop_bgr, imgsz=self.cfg.seg_imgsz, conf=self.cfg.seg_conf,
                              iou=self.cfg.seg_iou, verbose=False)
        masks = getattr(results[0], "masks", None) if results else None
        if masks is None or masks.xy is None:
            return []
        return [np.asarray(p, dtype=np.float32) for p in masks.xy if p is not None and len(p) >= 3]


# --------------------------------------------------------------------------- #
# Choix du contour
# --------------------------------------------------------------------------- #

def _contains(contour: np.ndarray, point: tuple[float, float]) -> bool:
    return cv2.pointPolygonTest(contour.reshape(-1, 1, 2).astype(np.float32),
                                (float(point[0]), float(point[1])), False) >= 0


def _area(contour: np.ndarray) -> float:
    return float(cv2.contourArea(contour.reshape(-1, 1, 2).astype(np.float32)))


def _pick_contour(candidates: list[np.ndarray], center: tuple[float, float],
                  max_area: float) -> np.ndarray | None:
    """Garde le plus grand contour qui contient le centre de la bulle (correction C)."""
    valid = [c for c in candidates
             if len(c) >= 10 and _area(c) < max_area and _contains(c, center)]
    return max(valid, key=_area) if valid else None


def _fallback_contours(crop_bgr: np.ndarray) -> list[np.ndarray]:
    """Binarisation (Otsu), normale ET inversée (correction E)."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours = []
    for img in (binary, cv2.bitwise_not(binary)):
        found, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours += [c.reshape(-1, 2).astype(np.float32) for c in found]
    return contours


# --------------------------------------------------------------------------- #
# Géométrie du contour
# --------------------------------------------------------------------------- #

def resample_contour(points: np.ndarray, step: float) -> np.ndarray:
    """Rééchantillonne un contour fermé à intervalles réguliers de `step` px (correction B)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    closed = np.vstack([pts, pts[:1]])
    seg_len = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total <= 0:
        return pts
    samples = np.arange(0.0, total, step)
    x = np.interp(samples, cum, closed[:, 0])
    y = np.interp(samples, cum, closed[:, 1])
    return np.stack([x, y], axis=1)


def find_tail_tip(points: np.ndarray, window: int, max_angle: float) -> tuple[int, float] | None:
    """Indice du sommet convexe le plus aigu, et son angle (correction D)."""
    n = len(points)
    if n < 2 * window + 1 or window < 1:
        return None
    prev_pts = np.roll(points, window, axis=0)
    next_pts = np.roll(points, -window, axis=0)
    v1, v2 = prev_pts - points, next_pts - points
    norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    cos = np.einsum("ij,ij->i", v1, v2) / np.maximum(norms, 1e-9)
    angles = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    contour = points.astype(np.float32)
    for i in np.argsort(angles):
        if angles[i] > max_angle:
            return None
        midpoint = (prev_pts[i] + next_pts[i]) / 2.0
        if _contains(contour, (midpoint[0], midpoint[1])):  # sommet convexe
            return int(i), float(angles[i])
    return None


def _find_base(points: np.ndarray, tip: int, center: np.ndarray, search: int) -> np.ndarray:
    """Base de la queue : de chaque côté de la pointe, le point le plus proche du centre."""
    n = len(points)

    def closest(direction: int) -> np.ndarray:
        idx = [(tip + direction * k) % n for k in range(1, search + 1)]
        dists = np.linalg.norm(points[idx] - center, axis=1)
        return points[idx[int(np.argmin(dists))]]

    return (closest(-1) + closest(+1)) / 2.0


# --------------------------------------------------------------------------- #
# Fonction principale
# --------------------------------------------------------------------------- #

MAX_PADDING_ATTEMPTS = 3


def analyze_tail(page_bgr: np.ndarray, bubble_bbox: BBox, cfg: TailConfig,
                 segmenter: Segmenter | None = None) -> TailInfo | None:
    """Analyse la queue d'une bulle. Renvoie None si aucune queue fiable n'est trouvée.

    Si la pointe trouvée touche le bord du recadrage, c'est que la queue a été
    coupée (le "coin" du recadrage ressemble à une pointe) : on recommence avec
    une marge doublée (correction A, version adaptative).
    """
    padding = cfg.crop_padding
    info = None
    for _ in range(MAX_PADDING_ATTEMPTS):
        info, touches_border = _analyze_with_padding(page_bgr, bubble_bbox, cfg, segmenter, padding)
        if info is None or not touches_border or padding == 0:
            break
        padding *= 2
    return info


def _analyze_with_padding(page_bgr: np.ndarray, bubble_bbox: BBox, cfg: TailConfig,
                          segmenter: Segmenter | None,
                          padding: float) -> tuple[TailInfo | None, bool]:
    h, w = page_bgr.shape[:2]
    crop_box = bubble_bbox.padded(padding, w, h)
    x0, y0 = int(crop_box.x1), int(crop_box.y1)
    crop = page_bgr[y0:int(crop_box.y2), x0:int(crop_box.x2)]
    if crop.size == 0:
        return None, False

    cx, cy = bubble_bbox.center
    center_local = (cx - x0, cy - y0)
    max_area = 0.95 * crop.shape[0] * crop.shape[1]  # exclut le "contour" du cadre entier

    contour = None
    if segmenter is not None:
        contour = _pick_contour(segmenter.polygons(crop), center_local, max_area)
    if contour is None:
        contour = _pick_contour(_fallback_contours(crop), center_local, max_area)
    if contour is None:
        return None, False

    points = resample_contour(contour, cfg.resample_step)
    window = max(1, round(cfg.tip_window / cfg.resample_step))
    found = find_tail_tip(points, window, cfg.max_tip_angle)
    if found is None:
        return None, False
    tip_idx, tip_angle = found

    moments = cv2.moments(points.reshape(-1, 1, 2).astype(np.float32))
    if abs(moments["m00"]) < 1e-6:
        return None, False
    centroid = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])

    search = max(1, round(cfg.base_search_length / cfg.resample_step))
    base = _find_base(points, tip_idx, centroid, search)
    tip = points[tip_idx]
    angle = (math.degrees(math.atan2(tip[1] - base[1], tip[0] - base[0])) + 360.0) % 360.0

    # La pointe colle-t-elle au bord du recadrage (hors bord réel de la page) ?
    ch, cw = crop.shape[:2]
    margin = 2.0
    touches = ((tip[0] <= margin and x0 > 0) or (tip[1] <= margin and y0 > 0)
               or (tip[0] >= cw - 1 - margin and x0 + cw < w)
               or (tip[1] >= ch - 1 - margin and y0 + ch < h))

    offset = np.array([x0, y0], dtype=np.float64)
    return TailInfo(
        tip=tuple(float(v) for v in tip + offset),
        base_center=tuple(float(v) for v in base + offset),
        angle_deg=float(angle),
        tip_angle_deg=tip_angle,
        contour=points + offset,
    ), touches
