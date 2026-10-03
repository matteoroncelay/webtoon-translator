"""Découpage des pages très hautes en panneaux, en se calant sur les gouttières.

Pourquoi : YOLO travaille en 640 px. Une page de webtoon fait souvent
800 x 10 000 px : redimensionnée d'un bloc, les bulles deviendraient minuscules.
On découpe donc en panneaux d'environ `target_panel_height`, en coupant de
préférence dans une gouttière (bande blanche ou noire) pour ne pas couper une bulle.

Avant : ce code était copié à l'identique dans detect_perso.py et detect_bulles_json.py.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import GutterConfig


def find_gutters(image_bgr: np.ndarray, cfg: GutterConfig) -> list[int]:
    """Renvoie l'ordonnée du centre de chaque gouttière horizontale."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    white_ratio = (gray >= cfg.white_threshold).mean(axis=1)
    black_ratio = (gray <= cfg.black_threshold).mean(axis=1)
    is_gutter = (white_ratio >= cfg.min_uniform_ratio) | (black_ratio >= cfg.min_uniform_ratio)

    # Repère les débuts/fins de chaque suite de lignes "gouttière" (vectorisé)
    edges = np.diff(np.concatenate([[0], is_gutter.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    return [int(s + (e - s) // 2)
            for s, e in zip(starts, ends)
            if e - s >= cfg.gutter_min_height]


def _next_cut(gutters: list[int], start: int, height: int, cfg: GutterConfig) -> int:
    """Choisit la prochaine coupe : la gouttière la plus proche de la hauteur visée."""
    if height - start <= cfg.panel_min_height:
        return height
    min_cut = start + cfg.panel_min_height
    candidates = [g for g in gutters if g >= min_cut]
    if not candidates:
        return height
    target = min(start + cfg.target_panel_height, height)
    return min(candidates, key=lambda g: abs(g - target))


def panel_ranges(height: int, gutters: list[int], cfg: GutterConfig) -> list[tuple[int, int]]:
    """Découpe [0, height] en intervalles (y_debut, y_fin) qui couvrent TOUTE la page.

    Différence avec l'ancien code : il ignorait les panneaux trop petits,
    ce qui pouvait faire perdre des bulles. Ici, rien n'est jamais ignoré.
    """
    ranges = []
    y = 0
    while y < height:
        cut = _next_cut(gutters, y, height, cfg)
        if cut <= y:
            cut = height
        ranges.append((y, cut))
        y = cut
    return ranges
