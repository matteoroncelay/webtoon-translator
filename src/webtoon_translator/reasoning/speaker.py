"""Attribution du locuteur d'une bulle à queue : décision hiérarchique.

  1. GÉOMÉTRIE   : cône partant de la pointe de la queue, dans sa direction.
                   Parmi les personnages dans le cône, on garde celui qui minimise
                   score = w_dist * distance/dist_max + w_angle * écart_angle/angle_max
  2. SEUL PRÉSENT: s'il n'y a qu'un personnage dans la scène, c'est lui.
  3. ÉLIMINATION : 2 personnages dans la scène (A, B), seul A est détecté, A est
                   proche de la bulle mais HORS du cône -> la bulle vise B,
                   que le détecteur n'a pas vu.
  4. Sinon       : inconnu (le LLM tranchera à l'étape 5).

C'est ta fonction assign_speaker_intuitif, réécrite sans le code dupliqué
après le `return`, et avec une différence : la règle 2 s'applique aussi
quand la queue n'a pas pu être analysée.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import SpeakerConfig
from ..models import CharacterDetection, SpeakerSource
from ..vision.tail import TailInfo
from .presence import in_scene


@dataclass(frozen=True)
class SpeakerDecision:
    class_id: int | None
    source: SpeakerSource


UNKNOWN = SpeakerDecision(None, SpeakerSource.NONE)


def cone_candidate(tip: tuple[float, float], angle_deg: float,
                   characters: list[CharacterDetection],
                   cfg: SpeakerConfig) -> int | None:
    """Règle 1 : le personnage le mieux placé dans le cône, ou None."""
    theta = math.radians(angle_deg)
    direction = (math.cos(theta), math.sin(theta))
    best_class, best_score = None, math.inf

    for char in characters:
        cx, cy = char.bbox.center
        vx, vy = cx - tip[0], cy - tip[1]
        dist = max(math.hypot(vx, vy), 1e-3)
        if dist > cfg.max_tip_distance:
            continue
        dot = vx * direction[0] + vy * direction[1]
        if dot <= 0:                       # derrière la queue
            continue
        error = math.degrees(math.acos(max(-1.0, min(1.0, dot / dist))))
        if error > cfg.max_angle_error:    # hors du cône
            continue
        score = (cfg.weight_distance * dist / cfg.max_tip_distance
                 + cfg.weight_angle * error / cfg.max_angle_error)
        if score < best_score:
            best_class, best_score = char.class_id, score
    return best_class


def _closest_distance(class_id: int, tip: tuple[float, float],
                      characters: list[CharacterDetection]) -> float | None:
    dists = [c.bbox.distance_to_point(*tip) for c in characters if c.class_id == class_id]
    return min(dists) if dists else None


def assign_speaker(tail: TailInfo | None, characters: list[CharacterDetection],
                   presence: dict[int, int], cfg: SpeakerConfig) -> SpeakerDecision:
    # 1. Géométrie
    if tail is not None and characters:
        found = cone_candidate(tail.tip, tail.angle_deg, characters, cfg)
        if found is not None:
            return SpeakerDecision(found, SpeakerSource.GEOMETRY)

    scene = in_scene(presence)

    # 2. Un seul personnage dans la scène
    if len(scene) == 1:
        return SpeakerDecision(next(iter(scene)), SpeakerSource.SINGLE_PRESENT)

    # 3. Déduction par élimination (nécessite la pointe de la queue)
    if tail is not None and len(scene) == 2:
        detected = {c.class_id for c in characters} & scene
        if len(detected) == 1:
            a = next(iter(detected))
            b = next(iter(scene - {a}))
            dist_a = _closest_distance(a, tail.tip, characters)
            if dist_a is not None and dist_a <= cfg.near_distance:
                return SpeakerDecision(b, SpeakerSource.ELIMINATION)

    # 4. Inconnu
    return UNKNOWN
