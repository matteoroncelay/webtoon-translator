"""Tests de la partie texte, avec un FAUX Gemini et un FAUX OCR (aucun appel réseau)."""

import json

import numpy as np
import pytest

from webtoon_translator.config import PROJECT_ROOT, load_config
from webtoon_translator.models import (
    BBox, Bubble, BubbleKind, BubbleType, Chapter, Page, SpeakerSource,
)
from webtoon_translator.reasoning import load_characters
from webtoon_translator.text import (
    GeminiClient, GeminiError, analyze_context, batched, classify_tailless, parse_json,
    read_page_text, translate_page,
)

CFG = load_config().gemini
REGISTRY = load_characters(PROJECT_ROOT / "config" / "characters.example.json")


class FakeTransport:
    """Rejoue une liste de réponses : chaîne JSON, None (réponse vide) ou exception."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, key, model, contents, cfg):
        self.calls.append((key, contents))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def client(transport, fallback=None):
    return GeminiClient(CFG, "cle-principale", fallback, transport=transport, sleep=lambda s: None)


# --------------------------------------------------------------------------- #
# Client Gemini
# --------------------------------------------------------------------------- #

def test_parse_json_strips_markdown_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_retries_on_quota_then_succeeds():
    transport = FakeTransport(RuntimeError("429 RESOURCE_EXHAUSTED"), '{"ok": true}')
    assert client(transport).generate_json(["x"]) == {"ok": True}
    assert len(transport.calls) == 2


def test_switches_to_fallback_key_when_quota_persists():
    errors = [RuntimeError("429 quota")] * CFG.max_retries
    transport = FakeTransport(*errors, '{"ok": 1}')
    assert client(transport, fallback="cle-secours").generate_json(["x"]) == {"ok": 1}
    assert transport.calls[-1][0] == "cle-secours"


def test_empty_answer_is_a_clear_error():
    with pytest.raises(GeminiError, match="vide"):
        client(FakeTransport(None)).generate_json(["x"])


def test_non_transient_error_is_not_retried():
    transport = FakeTransport(ValueError("400 INVALID_ARGUMENT"))
    with pytest.raises(GeminiError):
        client(transport).generate_json(["x"])
    assert len(transport.calls) == 1


def test_batched():
    assert list(batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


# --------------------------------------------------------------------------- #
# Étapes LLM
# --------------------------------------------------------------------------- #

def make_chapter() -> Chapter:
    page = Page("p1.jpg", 800, 1200, presence={0: 3, 1: 2}, bubbles=[
        Bubble(1, BBox(0, 0, 100, 100), BubbleKind.WITH_TAIL, type=BubbleType.SPEECH,
               text_ko="안녕", speaker="minjun", speaker_source=SpeakerSource.GEOMETRY),
        Bubble(2, BBox(0, 200, 100, 300), BubbleKind.NO_TAIL, text_ko="왜 그랬지..."),
        Bubble(3, BBox(0, 400, 100, 450), BubbleKind.NARRATION, type=BubbleType.NARRATION,
               text_ko="다음 날"),
    ])
    return Chapter("demo", pages=[page])


def test_classification_sets_type_of_tailless_bubbles_only():
    chapter = make_chapter()
    answer = json.dumps({"bubbles": [{"id": "p1.jpg#2", "type": "thought"}]})
    assert classify_tailless(chapter, client(FakeTransport(answer)), CFG) == 1
    assert chapter.pages[0].bubbles[1].type is BubbleType.THOUGHT


def test_context_never_overrides_geometry():
    chapter = make_chapter()
    chapter.pages[0].bubbles[1].type = BubbleType.THOUGHT
    answer = json.dumps({"bubbles": [
        {"id": "p1.jpg#1", "speaker": "seoyeon", "addressee": "seoyeon"},  # tente d'écraser
        {"id": "p1.jpg#2", "speaker": "seoyeon", "addressee": "self"},
    ]})
    transport = FakeTransport(answer)
    analyze_context(chapter, client(transport), REGISTRY, CFG)
    first, second, _ = chapter.pages[0].bubbles
    assert first.speaker == "minjun" and first.addressee == "seoyeon"     # géométrie conservée
    assert second.speaker == "seoyeon" and second.speaker_source is SpeakerSource.LLM
    assert second.addressee == "self"
    assert "p1.jpg#3" not in transport.calls[0][1][0]  # la narration n'est pas envoyée


def test_context_rejects_invented_names():
    chapter = make_chapter()
    answer = json.dumps({"bubbles": [{"id": "p1.jpg#2", "speaker": "personnage_invente"}]})
    analyze_context(chapter, client(FakeTransport(answer)), REGISTRY, CFG)
    assert chapter.pages[0].bubbles[1].speaker is None


def test_translation_updates_text_and_story():
    chapter = make_chapter()
    answer = json.dumps({"translations": [{"id": "p1.jpg#1", "text_fr": "Salut !"},
                                          {"id": "p1.jpg#3", "text_fr": "Le lendemain"}],
                         "page_summary": "Minjun salue Seoyeon."})
    done = translate_page(chapter, chapter.pages[0], client(FakeTransport(answer)), REGISTRY, CFG)
    assert done == 2
    assert chapter.pages[0].bubbles[0].text_fr == "Salut !"
    assert chapter.story_context == "Minjun salue Seoyeon."


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #

class FakeOcr:
    def read(self, crop):
        return "안녕\n하세요"


def test_ocr_fills_only_empty_bubbles_and_joins_lines():
    chapter = make_chapter()
    page = chapter.pages[0]
    page.bubbles[1].text_ko = ""
    image = np.zeros((1200, 800, 3), dtype=np.uint8)
    assert read_page_text(image, page, FakeOcr()) == 1
    assert page.bubbles[0].text_ko == "안녕"            # déjà lu : inchangé
    assert page.bubbles[1].text_ko == "안녕 하세요"
