"""Applique les règles d'attribution à toutes les bulles d'une page.

Remplace prepare_bubbles_json.py + merge_data.py : il n'y a plus de
"fusion" à faire, car toutes les bulles (les 3 classes) vivent dès le départ
dans le même objet Page.
"""

from __future__ import annotations

from ..config import SpeakerConfig
from ..models import BubbleKind, BubbleType, Page
from ..vision.tail import TailInfo
from .characters import CharacterRegistry
from .speaker import assign_speaker


def attribute_page(page: Page, tails: dict[int, TailInfo | None],
                   registry: CharacterRegistry, cfg: SpeakerConfig) -> None:
    """`tails` : résultat de l'analyse de queue pour chaque bulle (clé = bubble.id)."""
    for bubble in page.bubbles:
        if bubble.kind is BubbleKind.NARRATION:
            bubble.type = BubbleType.NARRATION
            continue
        if bubble.kind is BubbleKind.NO_TAIL:
            continue  # pensée ou parole hors champ : décidé par le LLM (étape 5)

        decision = assign_speaker(tails.get(bubble.id), page.characters, page.presence, cfg)
        bubble.type = BubbleType.SPEECH
        bubble.speaker_source = decision.source
        bubble.speaker = (registry.name_of(decision.class_id)
                          if decision.class_id is not None else None)
