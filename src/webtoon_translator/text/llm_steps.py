"""Les trois étapes qui utilisent le LLM : classification, contexte, traduction.

Chaque étape :
  1. sélectionne les bulles concernées ;
  2. les envoie PAR LOTS (l'ancien code envoyait tout le chapitre d'un coup,
     d'où des réponses tronquées sur les longs chapitres) ;
  3. ne modifie que les bulles dont l'identifiant est bien revenu dans la réponse.
"""

from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np

from ..config import GeminiConfig
from ..models import (
    Bubble, BubbleKind, BubbleType, Chapter, Page, SpeakerSource,
)
from ..reasoning.characters import CharacterRegistry
from ..reasoning.presence import in_scene
from . import prompts
from .gemini import GeminiClient, batched


def bubble_uid(page: Page, bubble: Bubble) -> str:
    return f"{page.file_name}#{bubble.id}"


def _dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1)


def _results_by_id(response: Any) -> dict[str, dict]:
    items = response.get("bubbles", []) if isinstance(response, dict) else []
    return {str(item["id"]): item for item in items if isinstance(item, dict) and "id" in item}


# --------------------------------------------------------------------------- #
# 1. Classification des bulles sans queue
# --------------------------------------------------------------------------- #

TYPE_FROM_LLM = {"thought": BubbleType.THOUGHT, "speech_offscreen": BubbleType.SPEECH_OFFSCREEN}


def classify_tailless(chapter: Chapter, client: GeminiClient, cfg: GeminiConfig) -> int:
    """Pensée ou parole hors champ ? Renvoie le nombre de bulles classées."""
    targets = [(p, b) for p in chapter.pages for b in p.bubbles
               if b.kind is BubbleKind.NO_TAIL and b.text_ko]
    done = 0
    for batch in batched(targets, cfg.batch_size):
        payload = [{"id": bubble_uid(p, b), "text_ko": b.text_ko} for p, b in batch]
        results = _results_by_id(client.generate_json([prompts.CLASSIFY.format(bubbles=_dump(payload))]))
        for page, bubble in batch:
            kind = TYPE_FROM_LLM.get(results.get(bubble_uid(page, bubble), {}).get("type"))
            if kind is not None:
                bubble.type = kind
                done += 1
    return done


# --------------------------------------------------------------------------- #
# 2. Contexte : qui parle, à qui
# --------------------------------------------------------------------------- #

def analyze_context(chapter: Chapter, client: GeminiClient, registry: CharacterRegistry,
                    cfg: GeminiConfig, previous_lines: int = 6) -> int:
    """Complète `speaker` (si inconnu) et `addressee`. Renvoie le nombre de bulles mises à jour."""
    sequence = [(p, b) for p in chapter.pages for b in p.bubbles
                if b.type is not BubbleType.NARRATION and b.text_ko]
    profiles = _dump(registry.to_prompt_dict())
    known_names = set(registry.profiles)
    history: list[str] = []
    updated = 0

    for batch in batched(sequence, cfg.batch_size):
        payload = [{
            "id": bubble_uid(p, b),
            "text_ko": b.text_ko,
            "type": b.type.value,
            "known_speaker": b.speaker,
            "possible_speakers": registry.names_of(in_scene(p.presence)),
        } for p, b in batch]
        prompt = prompts.CONTEXT.format(profiles=profiles, bubbles=_dump(payload),
                                        previous="\n".join(history[-previous_lines:]) or "(start)")
        results = _results_by_id(client.generate_json([prompt]))

        for page, bubble in batch:
            item = results.get(bubble_uid(page, bubble))
            if item is None:
                continue
            speaker, addressee = item.get("speaker"), item.get("addressee")
            # La géométrie a priorité : le LLM ne complète que les locuteurs inconnus
            if bubble.speaker is None and speaker in known_names:
                bubble.speaker = speaker
                bubble.speaker_source = SpeakerSource.LLM
            if addressee in known_names or addressee == "self":
                bubble.addressee = addressee
            updated += 1
            history.append(f"{bubble.speaker or '?'} -> {bubble.addressee or '?'} : {bubble.text_ko}")
    return updated


# --------------------------------------------------------------------------- #
# 3. Traduction, page par page, avec mémoire de l'histoire
# --------------------------------------------------------------------------- #

def _encode_page(page_bgr: np.ndarray, max_height: int = 3000) -> Any:
    """Image JPEG pour le LLM (réduite si la page est très haute)."""
    from google.genai import types  # import tardif : seulement si on envoie l'image

    h, w = page_bgr.shape[:2]
    if h > max_height:
        page_bgr = cv2.resize(page_bgr, (int(w * max_height / h), max_height))
    ok, buffer = cv2.imencode(".jpg", page_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Impossible d'encoder l'image de la page")
    return types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg")


def translate_page(chapter: Chapter, page: Page, client: GeminiClient,
                   registry: CharacterRegistry, cfg: GeminiConfig,
                   page_bgr: np.ndarray | None = None) -> int:
    """Traduit une page et met à jour le résumé de l'histoire. Renvoie le nb de bulles traduites."""
    bubbles = [b for b in page.bubbles if b.text_ko]
    if not bubbles:
        return 0
    payload = [{"id": bubble_uid(page, b), "text_ko": b.text_ko, "type": b.type.value,
                "speaker": b.speaker, "addressee": b.addressee} for b in bubbles]
    prompt = prompts.TRANSLATE.format(story=chapter.story_context or "(start of the chapter)",
                                      profiles=_dump(registry.to_prompt_dict()),
                                      bubbles=_dump(payload))
    contents: list[Any] = [prompt]
    if cfg.send_page_image and page_bgr is not None:
        contents.append(_encode_page(page_bgr))

    response = client.generate_json(contents)
    translations = {str(t.get("id")): t.get("text_fr", "")
                    for t in response.get("translations", []) if isinstance(t, dict)}
    done = 0
    for bubble in bubbles:
        text = translations.get(bubble_uid(page, bubble))
        if text:
            bubble.text_fr = text.strip()
            done += 1

    summary = (response.get("page_summary") or "").strip()
    if summary:
        story = f"{chapter.story_context}\n{summary}".strip()
        # On garde la fin du résumé : ce sont les événements récents qui comptent le plus
        chapter.story_context = story[-cfg.context_max_chars:]
    return done
