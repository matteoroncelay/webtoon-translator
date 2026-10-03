"""Suivi de la présence des personnages d'une page à l'autre.

Idée (ta logique d'origine, inchangée) : un détecteur rate souvent un personnage
d'une case à l'autre (dos tourné, coupé par le cadre...). Plutôt que de le faire
disparaître dès qu'il n'est plus détecté, chaque personnage a un score de 0 à 3 :

    3 = VISUEL     : détecté avec une forte confiance sur cette page
    2 = PRÉSENT    : vu récemment, sans doute encore dans la scène
    1 = POTENTIEL  : détection faible, ou vu il y a longtemps
    0 = ABSENT

À chaque page : détection forte -> 3 ; détection faible -> au moins 1 ;
personnage non détecté -> score - 1 (il "s'efface" progressivement).

Retiré par rapport à l'ancien code : CONFUSION_ALIAS (la correction par matrice
de confusion), qui était prévue mais vide, donc jamais utilisée.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..config import PresenceConfig
from ..models import CharacterDetection, Page

VISUAL, PRESENT, POTENTIAL, ABSENT = 3, 2, 1, 0


def update_presence(scores: dict[int, int], detections: Iterable[CharacterDetection],
                    cfg: PresenceConfig) -> dict[int, int]:
    """Nouveaux scores après une page (ne modifie pas `scores`)."""
    new_scores = dict(scores)
    seen: set[int] = set()

    for det in detections:
        seen.add(det.class_id)
        if det.confidence >= cfg.strong_conf:
            new_scores[det.class_id] = VISUAL
        else:
            new_scores[det.class_id] = max(new_scores.get(det.class_id, ABSENT), POTENTIAL)

    for class_id in new_scores:
        if class_id not in seen:
            new_scores[class_id] = max(new_scores[class_id] - 1, ABSENT)

    return {c: s for c, s in new_scores.items() if s > ABSENT}


def compute_presence(pages: list[Page], cfg: PresenceConfig) -> None:
    """Remplit `page.presence` pour chaque page, dans l'ordre de lecture."""
    scores: dict[int, int] = {}
    for page in pages:
        scores = update_presence(scores, page.characters, cfg)
        page.presence = dict(scores)


def in_scene(presence: dict[int, int]) -> set[int]:
    """Personnages considérés dans la scène : visuels ou présents (score >= 2)."""
    return {c for c, s in presence.items() if s >= PRESENT}
