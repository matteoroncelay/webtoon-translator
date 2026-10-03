"""Test de bout en bout du pipeline, avec de faux composants (ni modèle, ni réseau)."""

import json
from pathlib import Path

import cv2
import numpy as np

from webtoon_translator.config import PROJECT_ROOT, load_config
from webtoon_translator.io import load_chapter, read_image, write_image
from webtoon_translator.models import BBox, Bubble, BubbleKind, BubbleType, SpeakerSource
from webtoon_translator.pipeline import Components, run_pipeline
from webtoon_translator.reasoning import load_characters
from webtoon_translator.render import render_page, write_reader
from webtoon_translator.text import GeminiClient
from webtoon_translator.vision import YoloDetector

CONFIG = load_config()


# --------------------------------------------------------------------------- #
# Rendu
# --------------------------------------------------------------------------- #

def test_render_erases_and_writes_inside_the_bubble():
    page = np.full((400, 400, 3), 255, dtype=np.uint8)
    cv2.putText(page, "KOREAN", (120, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    bubble = Bubble(1, BBox(100, 150, 300, 250), BubbleKind.WITH_TAIL,
                    text_fr="Une phrase assez longue pour devoir passer à la ligne")
    out = render_page(page, [bubble], "arial.ttf", CONFIG.render)
    assert out.shape == page.shape
    outside = np.ones(page.shape[:2], bool)
    outside[150:250, 100:300] = False
    assert (out[outside] == 255).all()          # rien n'est dessiné hors de la bulle
    assert (out[150:250, 100:300] < 128).any()  # du texte a bien été écrit dedans


def test_reader_html_uses_relative_paths(tmp_path: Path):
    imgs = [tmp_path / "pages" / "p1.jpg", tmp_path / "pages" / "p2.jpg"]
    write_reader(imgs, tmp_path / "lecture.html", "Mon <chapitre>")
    html = (tmp_path / "lecture.html").read_text(encoding="utf-8")
    assert 'src="pages/p1.jpg"' in html and "Mon &lt;chapitre&gt;" in html


# --------------------------------------------------------------------------- #
# Pipeline complet
# --------------------------------------------------------------------------- #

class _Box:
    def __init__(self, xyxy, cls):
        self.xyxy = [type("T", (list,), {"tolist": lambda s: list(s)})(xyxy)]
        self.cls, self.conf = [cls], [0.95]


class _Model:
    def __init__(self, boxes):
        self.boxes = boxes

    def __call__(self, panel, **kw):
        return [type("R", (), {"boxes": [_Box(*b) for b in self.boxes]})()]


class _Ocr:
    def read(self, crop):
        return "안녕"


class _Llm:
    """Faux transport Gemini : répond selon le type de requête."""

    def __init__(self):
        self.calls = 0

    def __call__(self, key, model, contents, cfg):
        self.calls += 1
        prompt = contents[0]
        ids = [line.split('"')[3] for line in prompt.splitlines() if '"id":' in line]
        if "You classify" in prompt:
            return json.dumps({"bubbles": [{"id": i, "type": "thought"} for i in ids]})
        if "WHO speaks" in prompt:
            return json.dumps({"bubbles": [{"id": i, "addressee": "seoyeon"} for i in ids]})
        return json.dumps({"translations": [{"id": i, "text_fr": "Salut"} for i in ids],
                           "page_summary": "Une page."})


def _components(llm) -> Components:
    bubbles = _Model([([60, 40, 260, 140], 0), ([60, 300, 260, 380], 1)])  # avec / sans queue
    characters = _Model([([500, 20, 600, 220], 0)])  # "minjun" à droite de la 1re bulle
    return Components(
        bubble_detector=YoloDetector("x", 640, 0.45, model=bubbles),
        character_detector=YoloDetector("x", 640, 0.45, model=characters),
        segmenter=None,
        ocr=_Ocr(),
        llm=GeminiClient(CONFIG.gemini, "cle", transport=llm, sleep=lambda s: None),
        registry=load_characters(PROJECT_ROOT / "config" / "characters.example.json"),
    )


def _demo_page() -> np.ndarray:
    img = np.full((500, 700, 3), 255, dtype=np.uint8)
    # Bulle à queue blanche sur fond gris, queue pointant vers la droite (vers minjun)
    img[:] = 200
    cv2.ellipse(img, (160, 90), (95, 45), 0, 0, 360, (255, 255, 255), -1)
    cv2.fillPoly(img, [np.array([[240, 75], [240, 105], [330, 95]], np.int32)], (255, 255, 255))
    cv2.rectangle(img, (60, 300), (260, 380), (255, 255, 255), -1)
    return img


def test_full_pipeline_and_resume(tmp_path: Path):
    write_image(tmp_path / "in" / "page_001.png", _demo_page())
    out = tmp_path / "out"
    llm = _Llm()

    chapter = run_pipeline(tmp_path / "in", out, CONFIG, _components(llm))
    first, second = chapter.pages[0].bubbles
    assert first.speaker == "minjun" and first.speaker_source is SpeakerSource.GEOMETRY
    assert second.type is BubbleType.THOUGHT
    assert first.text_fr == "Salut" and first.addressee == "seoyeon"
    for name in ("chapter.json", "lecture.html", "rapport.csv", "pages/page_001.png"):
        assert (out / name).is_file()
    assert read_image(out / "pages" / "page_001.png").shape == (500, 700, 3)

    # Reprise : tout est déjà fait, donc aucun nouvel appel au LLM
    calls = llm.calls
    run_pipeline(tmp_path / "in", out, CONFIG, _components(llm), resume=True)
    assert llm.calls == calls
    assert load_chapter(out / "chapter.json").steps_done == [
        "vision", "ocr", "classify", "context", "translate"]
