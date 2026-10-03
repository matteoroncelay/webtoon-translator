from pathlib import Path

import numpy as np
import pytest

from webtoon_translator.io import list_images, load_chapter, read_image, save_chapter, write_image
from webtoon_translator.models import (
    BBox, Bubble, BubbleKind, BubbleType, Chapter, CharacterDetection, Page, SpeakerSource,
)


def test_bbox_basics():
    box = BBox(10, 20, 50, 100)
    assert box.width == 40 and box.height == 80
    assert box.center == (30, 60)
    assert box.shifted(dy=500) == BBox(10, 520, 50, 600)


def test_bbox_padding_stays_inside_image():
    box = BBox(5, 5, 95, 95)
    assert box.padded(20, width=100, height=100) == BBox(0, 0, 100, 100)


def test_invalid_bbox_rejected():
    with pytest.raises(ValueError):
        BBox(50, 0, 10, 10)


def test_reading_order_and_renumbering():
    page = Page("p.jpg", 800, 2000, bubbles=[
        Bubble(id=7, bbox=BBox(0, 900, 10, 950), kind=BubbleKind.NO_TAIL),
        Bubble(id=3, bbox=BBox(0, 100, 10, 150), kind=BubbleKind.WITH_TAIL),
    ])
    page.sort_bubbles_reading_order()
    assert [b.id for b in page.bubbles] == [1, 2]
    assert page.bubbles[0].bbox.y1 == 100


def test_chapter_json_roundtrip(tmp_path: Path):
    chapter = Chapter("demo", story_context="Début.", pages=[
        Page("page_001.jpg", 800, 1200,
             bubbles=[Bubble(1, BBox(1, 2, 3, 4), BubbleKind.WITH_TAIL, 0.9,
                             BubbleType.SPEECH, "안녕", "Salut", "minjun",
                             SpeakerSource.GEOMETRY, "seoyeon")],
             characters=[CharacterDetection(0, BBox(10, 10, 60, 90), 0.88)],
             presence={0: 3, 1: 1}),
    ])
    path = tmp_path / "out" / "chapter.json"
    save_chapter(chapter, path)
    assert load_chapter(path) == chapter


def test_images_natural_order_and_unicode_path(tmp_path: Path):
    folder = tmp_path / "Téléchargements"  # chemin avec accents
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    for name in ["page_10.jpg", "page_2.jpg", "page_1.png"]:
        write_image(folder / name, img)
    (folder / "notes.txt").write_text("pas une image")  # doit être ignoré
    assert [p.name for p in list_images(folder)] == ["page_1.png", "page_2.jpg", "page_10.jpg"]
    assert read_image(folder / "page_2.jpg").shape == (10, 10, 3)
