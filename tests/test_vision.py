"""Tests de la partie vision, sur des images SYNTHÉTIQUES (pas besoin des modèles)."""

import math
from dataclasses import replace

import cv2
import numpy as np
import pytest

from webtoon_translator.config import load_config
from webtoon_translator.models import BBox, BubbleKind
from webtoon_translator.vision import (
    YoloDetector, analyze_tail, detect_bubbles, find_gutters, panel_ranges,
)
from webtoon_translator.vision.tail import resample_contour

CONFIG = load_config()


# --------------------------------------------------------------------------- #
# Panneaux
# --------------------------------------------------------------------------- #

def test_gutters_found_in_white_bands():
    img = np.full((1000, 200, 3), 120, dtype=np.uint8)  # fond gris = contenu
    img[400:440] = 255   # gouttière blanche de 40 px, centre 420
    img[700:710] = 255   # trop fine (< 20 px) : ignorée
    assert find_gutters(img, CONFIG.gutters) == [420]


def test_panels_cover_whole_page():
    ranges = panel_ranges(3000, [700, 1600, 2300], CONFIG.gutters)
    assert ranges[0][0] == 0 and ranges[-1][1] == 3000
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))  # pas de trou
    assert ranges[0] == (0, 700)  # coupe sur la gouttière la plus proche de 750


# --------------------------------------------------------------------------- #
# Détecteur (avec un faux modèle YOLO)
# --------------------------------------------------------------------------- #

class _FakeTensor(list):
    def tolist(self):
        return list(self)


class _FakeBox:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = [_FakeTensor(xyxy)]
        self.cls = [cls]
        self.conf = [conf]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    """Renvoie une bulle à (10,10)-(50,50) dans chaque panneau."""

    def __call__(self, panel, **kwargs):
        return [_FakeResult([_FakeBox([10, 10, 50, 50], 0, 0.9)])]


def test_detections_are_reprojected_to_page_coordinates():
    img = np.full((1600, 200, 3), 120, dtype=np.uint8)
    img[780:820] = 255  # une gouttière vers 800 -> 2 panneaux
    detector = YoloDetector("inutile.pt", imgsz=640, iou=0.45, model=_FakeModel())
    bubbles = detect_bubbles(img, detector, CONFIG)
    assert [b.bbox.y1 for b in bubbles] == [10, 810]  # 2e panneau décalé de 800
    assert all(b.kind is BubbleKind.WITH_TAIL for b in bubbles)


# --------------------------------------------------------------------------- #
# Queue des bulles
# --------------------------------------------------------------------------- #

TIP = (260, 330)


def _bubble_with_tail() -> tuple[np.ndarray, BBox]:
    """Bulle blanche (ellipse) avec une queue qui sort LARGEMENT de sa boîte."""
    img = np.zeros((450, 400, 3), dtype=np.uint8)
    cv2.ellipse(img, (200, 150), (90, 55), 0, 0, 360, (255, 255, 255), -1)
    tail = np.array([[175, 190], [225, 190], TIP], dtype=np.int32)
    cv2.fillPoly(img, [tail], (255, 255, 255))
    bbox = BBox(110, 95, 290, 205)  # boîte de l'ellipse seule, comme un YOLO trop serré
    return img, bbox


def test_resample_has_constant_step():
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
    pts = resample_contour(square, step=4.0)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert len(pts) == 100 and np.allclose(steps, 4.0)


def test_tail_found_thanks_to_padding():
    img, bbox = _bubble_with_tail()
    info = analyze_tail(img, bbox, CONFIG.tail)
    assert info is not None
    assert math.dist(info.tip, TIP) < 10
    expected = math.degrees(math.atan2(TIP[1] - 190, TIP[0] - 200))  # base -> pointe
    assert abs(info.angle_deg - expected) < 15


def test_without_padding_the_tail_is_lost():
    """Reproduit le bug de l'ancien code : recadrage sur la boîte YOLO seule."""
    img, bbox = _bubble_with_tail()
    old_cfg = replace(CONFIG.tail, crop_padding=0)
    info = analyze_tail(img, bbox, old_cfg)
    assert info is None or math.dist(info.tip, TIP) > 50


def test_dark_bubble_uses_inverted_fallback():
    img, bbox = _bubble_with_tail()
    info = analyze_tail(255 - img, bbox, CONFIG.tail)  # bulle noire sur fond blanc
    assert info is not None and math.dist(info.tip, TIP) < 10


class _FakeSegmenter:
    """Renvoie une grande bulle voisine ET la bonne bulle : il faut choisir la bonne."""

    def __init__(self, good: np.ndarray):
        self.good = good

    def polygons(self, crop):
        h, w = crop.shape[:2]
        neighbour = np.array([[0, 0], [w * 0.9, 0], [w * 0.9, 20], [0, 20]], dtype=np.float32)
        neighbour = resample_contour(neighbour, 2).astype(np.float32)
        return [neighbour, self.good]


def test_segmenter_picks_contour_containing_bubble_center():
    img, bbox = _bubble_with_tail()
    # contour exact de la bulle, en coordonnées du recadrage
    pad = CONFIG.tail.crop_padding
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    good = contours[0].reshape(-1, 2).astype(np.float32) - [bbox.x1 - pad, bbox.y1 - pad]
    info = analyze_tail(img, bbox, CONFIG.tail, segmenter=_FakeSegmenter(good))
    assert info is not None and math.dist(info.tip, TIP) < 10
