"""Tests de la logique de décision, sur des scènes construites à la main.

Aucun modèle n'est nécessaire : on fabrique les détections et la queue,
puis on vérifie que chaque règle prend la bonne décision.
"""

from pathlib import Path

import numpy as np
import pytest

from webtoon_translator.config import PROJECT_ROOT, load_config
from webtoon_translator.io.report import evaluate_report, write_report
from webtoon_translator.models import (
    BBox, Bubble, BubbleKind, BubbleType, Chapter, CharacterDetection, Page, SpeakerSource,
)
from webtoon_translator.reasoning import (
    assign_speaker, attribute_page, compute_presence, load_characters, update_presence,
)
from webtoon_translator.vision.tail import TailInfo

CONFIG = load_config()
A, B, C = 0, 1, 2  # classes YOLO de trois personnages


def char(class_id: int, cx: float, cy: float, conf: float = 0.9) -> CharacterDetection:
    """Personnage de 100 x 200 px centré en (cx, cy)."""
    return CharacterDetection(class_id, BBox(cx - 50, cy - 100, cx + 50, cy + 100), conf)


def tail(tip=(100.0, 100.0), angle=0.0) -> TailInfo:
    return TailInfo(tip=tip, base_center=tip, angle_deg=angle, tip_angle_deg=40.0,
                    contour=np.zeros((0, 2)))


# --------------------------------------------------------------------------- #
# Présence
# --------------------------------------------------------------------------- #

def test_presence_fades_one_level_per_page():
    cfg = CONFIG.presence
    scores = update_presence({}, [char(A, 0, 0, conf=0.9)], cfg)
    assert scores == {A: 3}
    scores = update_presence(scores, [], cfg)
    assert scores == {A: 2}            # toujours "présent" juste après
    scores = update_presence(scores, [], cfg)
    assert scores == {A: 1}
    assert update_presence(scores, [], cfg) == {}  # disparu


def test_weak_detection_does_not_lower_a_visual_character():
    cfg = CONFIG.presence
    scores = update_presence({A: 3}, [char(A, 0, 0, conf=0.5)], cfg)
    assert scores == {A: 3}
    assert update_presence({}, [char(B, 0, 0, conf=0.5)], cfg) == {B: 1}


def test_compute_presence_follows_reading_order():
    pages = [Page("1.jpg", 10, 10, characters=[char(A, 0, 0)]), Page("2.jpg", 10, 10)]
    compute_presence(pages, CONFIG.presence)
    assert pages[0].presence == {A: 3} and pages[1].presence == {A: 2}


# --------------------------------------------------------------------------- #
# Règle 1 : géométrie
# --------------------------------------------------------------------------- #

def test_geometry_picks_character_in_front_of_the_tail():
    chars = [char(A, 400, 100), char(B, 100, 400)]   # A à droite, B en dessous
    decision = assign_speaker(tail(angle=0), chars, {}, CONFIG.speaker)  # queue vers la droite
    assert decision.class_id == A and decision.source is SpeakerSource.GEOMETRY


def test_geometry_ignores_characters_behind_or_too_far():
    chars = [char(A, -200, 100), char(B, 2000, 100)]  # A derrière, B à 1900 px
    decision = assign_speaker(tail(angle=0), chars, {}, CONFIG.speaker)
    assert decision.source is SpeakerSource.NONE


def test_geometry_prefers_the_centre_of_the_cone():
    # A légèrement plus près mais à 20° de l'axe ; B un peu plus loin mais dans l'axe
    a_x = 100 + 300 * np.cos(np.radians(20))
    a_y = 100 + 300 * np.sin(np.radians(20))
    chars = [char(A, a_x, a_y), char(B, 450, 100)]
    decision = assign_speaker(tail(angle=0), chars, {}, CONFIG.speaker)
    assert decision.class_id == B  # l'angle pèse 2 fois plus que la distance


# --------------------------------------------------------------------------- #
# Règles 2 et 3 : logique de présence
# --------------------------------------------------------------------------- #

def test_single_character_in_scene():
    decision = assign_speaker(None, [], {A: 2}, CONFIG.speaker)
    assert decision.class_id == A and decision.source is SpeakerSource.SINGLE_PRESENT


def test_elimination_points_to_the_undetected_character():
    # A est détecté juste à gauche de la bulle, mais la queue pointe vers la droite :
    # la géométrie l'écarte, donc la bulle vise B, présent mais non détecté.
    chars = [char(A, -150, 100)]
    decision = assign_speaker(tail(angle=0), chars, {A: 3, B: 2}, CONFIG.speaker)
    assert decision.class_id == B and decision.source is SpeakerSource.ELIMINATION


def test_no_decision_when_both_are_detected_outside_the_cone():
    chars = [char(A, -150, 100), char(B, 100, -300)]
    decision = assign_speaker(tail(angle=0), chars, {A: 3, B: 3}, CONFIG.speaker)
    assert decision.source is SpeakerSource.NONE


def test_no_decision_with_three_characters_and_no_geometry():
    decision = assign_speaker(None, [], {A: 3, B: 2, C: 2}, CONFIG.speaker)
    assert decision.class_id is None


# --------------------------------------------------------------------------- #
# Page complète + fiches + export CSV
# --------------------------------------------------------------------------- #

def test_attribute_page_and_report(tmp_path: Path):
    registry = load_characters(PROJECT_ROOT / "config" / "characters.example.json")
    page = Page("p.jpg", 800, 1200, characters=[char(A, 400, 100)], presence={A: 3}, bubbles=[
        Bubble(1, BBox(50, 50, 150, 150), BubbleKind.WITH_TAIL),
        Bubble(2, BBox(50, 500, 150, 600), BubbleKind.NO_TAIL),
        Bubble(3, BBox(50, 900, 150, 950), BubbleKind.NARRATION),
    ])
    attribute_page(page, {1: tail(angle=0)}, registry, CONFIG.speaker)

    first, second, third = page.bubbles
    assert first.speaker == "minjun" and first.speaker_source is SpeakerSource.GEOMETRY
    assert second.type is BubbleType.UNKNOWN          # laissé au LLM
    assert third.type is BubbleType.NARRATION

    report = tmp_path / "report.csv"
    write_report(Chapter("demo", pages=[page]), report)
    lines = report.read_text(encoding="utf-8-sig").splitlines()
    assert len(lines) == 4  # en-tête + 3 bulles
    # Simule l'annotation manuelle : la 1re bulle est juste
    lines[1] = lines[1] + "1"
    report.write_text("\n".join(lines), encoding="utf-8-sig")
    assert evaluate_report(report)["geometry"] == (1, 1)


def test_duplicate_yolo_class_is_rejected(tmp_path: Path):
    bad = tmp_path / "chars.json"
    bad.write_text('{"characters": {"x": {"yolo_class": 0}, "y": {"yolo_class": 0}}}',
                   encoding="utf-8")
    with pytest.raises(ValueError, match="Classe YOLO 0"):
        load_characters(bad)
