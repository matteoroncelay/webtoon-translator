"""Structures de données partagées par tout le pipeline.

Avant : chaque script avait son propre format JSON (parfois "pages", parfois
"images", boîtes en "x1/y1/x2/y2" OU en "bbox": [...], états en chaînes libres).
Maintenant : un seul modèle, utilisé partout, sérialisé d'une seule façon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum


# --------------------------------------------------------------------------- #
# Géométrie
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BBox:
    """Boîte englobante en coordonnées page (pixels). x1,y1 = coin haut-gauche."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"BBox invalide : {self}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def shifted(self, dx: float = 0.0, dy: float = 0.0) -> BBox:
        """Translate la boîte (ex. : coordonnées panneau -> coordonnées page)."""
        return BBox(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def padded(self, margin: float, width: float, height: float) -> BBox:
        """Agrandit la boîte de `margin` px de chaque côté, sans sortir de l'image."""
        return BBox(
            max(0.0, self.x1 - margin),
            max(0.0, self.y1 - margin),
            min(width, self.x2 + margin),
            min(height, self.y2 + margin),
        )

    def distance_to_point(self, x: float, y: float) -> float:
        """Distance entre le centre de la boîte et un point."""
        cx, cy = self.center
        return math.hypot(cx - x, cy - y)

    def to_list(self) -> list[float]:
        return [round(self.x1, 2), round(self.y1, 2), round(self.x2, 2), round(self.y2, 2)]

    @classmethod
    def from_list(cls, values: list[float]) -> BBox:
        x1, y1, x2, y2 = (float(v) for v in values)
        return cls(x1, y1, x2, y2)


# --------------------------------------------------------------------------- #
# Bulles et personnages
# --------------------------------------------------------------------------- #

class BubbleKind(IntEnum):
    """Classes du modèle YOLO de détection de bulles."""

    WITH_TAIL = 0   # bulle avec queue : locuteur trouvable par la géométrie
    NO_TAIL = 1     # bulle sans queue : pensée ou parole hors champ (-> LLM)
    NARRATION = 2   # cartouche de narration : pas de locuteur


class BubbleType(str, Enum):
    """Nature du texte, déterminée au fil du pipeline."""

    UNKNOWN = "unknown"
    SPEECH = "speech"                    # parole d'un personnage visible
    SPEECH_OFFSCREEN = "speech_offscreen"
    THOUGHT = "thought"
    NARRATION = "narration"


class SpeakerSource(str, Enum):
    """Comment le locuteur a été trouvé (utile pour évaluer chaque méthode)."""

    NONE = "none"
    GEOMETRY = "geometry"        # cône depuis la pointe de la queue
    SINGLE_PRESENT = "single"    # un seul personnage présent dans la scène
    ELIMINATION = "elimination"  # déduction A/B
    LLM = "llm"                  # Gemini


@dataclass
class CharacterDetection:
    """Un personnage détecté par YOLO sur une page."""

    class_id: int
    bbox: BBox
    confidence: float

    def to_dict(self) -> dict:
        return {"class_id": self.class_id, "bbox": self.bbox.to_list(),
                "confidence": round(self.confidence, 4)}

    @classmethod
    def from_dict(cls, d: dict) -> CharacterDetection:
        return cls(int(d["class_id"]), BBox.from_list(d["bbox"]), float(d["confidence"]))


@dataclass
class Bubble:
    """Une bulle de texte et tout ce que le pipeline apprend sur elle."""

    id: int
    bbox: BBox
    kind: BubbleKind
    confidence: float = 1.0
    type: BubbleType = BubbleType.UNKNOWN
    text_ko: str = ""
    text_fr: str = ""
    speaker: str | None = None        # nom du personnage (clé des fiches)
    speaker_source: SpeakerSource = SpeakerSource.NONE
    addressee: str | None = None      # à qui il parle ("self" pour une pensée)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bbox": self.bbox.to_list(),
            "kind": int(self.kind),
            "confidence": round(self.confidence, 4),
            "type": self.type.value,
            "text_ko": self.text_ko,
            "text_fr": self.text_fr,
            "speaker": self.speaker,
            "speaker_source": self.speaker_source.value,
            "addressee": self.addressee,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Bubble:
        return cls(
            id=int(d["id"]),
            bbox=BBox.from_list(d["bbox"]),
            kind=BubbleKind(int(d["kind"])),
            confidence=float(d.get("confidence", 1.0)),
            type=BubbleType(d.get("type", "unknown")),
            text_ko=d.get("text_ko", ""),
            text_fr=d.get("text_fr", ""),
            speaker=d.get("speaker"),
            speaker_source=SpeakerSource(d.get("speaker_source", "none")),
            addressee=d.get("addressee"),
        )


# --------------------------------------------------------------------------- #
# Pages et chapitre
# --------------------------------------------------------------------------- #

@dataclass
class Page:
    """Une image du chapitre, avec ses détections."""

    file_name: str
    width: int
    height: int
    bubbles: list[Bubble] = field(default_factory=list)
    characters: list[CharacterDetection] = field(default_factory=list)
    # Présence calculée à l'étape 4 : {class_id: score 0..3}
    presence: dict[int, int] = field(default_factory=dict)

    def sort_bubbles_reading_order(self) -> None:
        """Trie les bulles de haut en bas puis de gauche à droite, et renumérote."""
        self.bubbles.sort(key=lambda b: (b.bbox.y1, b.bbox.x1))
        for i, bubble in enumerate(self.bubbles, start=1):
            bubble.id = i

    def to_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "width": self.width,
            "height": self.height,
            "bubbles": [b.to_dict() for b in self.bubbles],
            "characters": [c.to_dict() for c in self.characters],
            "presence": {str(k): v for k, v in self.presence.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> Page:
        return cls(
            file_name=d["file_name"],
            width=int(d["width"]),
            height=int(d["height"]),
            bubbles=[Bubble.from_dict(b) for b in d.get("bubbles", [])],
            characters=[CharacterDetection.from_dict(c) for c in d.get("characters", [])],
            presence={int(k): int(v) for k, v in d.get("presence", {}).items()},
        )


@dataclass
class Chapter:
    """Un chapitre complet : c'est l'objet qui traverse tout le pipeline."""

    name: str
    pages: list[Page] = field(default_factory=list)
    # Résumé de l'histoire, mis à jour page après page par la traduction
    story_context: str = ""

    def to_dict(self) -> dict:
        return {
            "format_version": 1,
            "name": self.name,
            "story_context": self.story_context,
            "pages": [p.to_dict() for p in self.pages],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Chapter:
        version = d.get("format_version")
        if version != 1:
            raise ValueError(f"Version de format non prise en charge : {version}")
        return cls(
            name=d["name"],
            pages=[Page.from_dict(p) for p in d.get("pages", [])],
            story_context=d.get("story_context", ""),
        )
