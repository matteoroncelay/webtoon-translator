"""Fiches personnages : relie les classes YOLO aux noms et profils.

Avant : le tableau classe YOLO -> nom était codé en dur, en double,
dans analyze_context_gemini.py et translator.py. Maintenant : un fichier JSON
fourni par l'utilisateur (voir config/characters.example.json).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CharacterProfile:
    name: str
    yolo_class: int | None
    role: str = ""
    tone: tuple[str, ...] = ()
    relations: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CharacterRegistry:
    profiles: dict[str, CharacterProfile]

    def name_of(self, class_id: int) -> str | None:
        """Nom du personnage correspondant à une classe YOLO (None si inconnue)."""
        for profile in self.profiles.values():
            if profile.yolo_class == class_id:
                return profile.name
        return None

    def names_of(self, class_ids) -> list[str]:
        return [n for n in (self.name_of(c) for c in sorted(class_ids)) if n is not None]

    def to_prompt_dict(self) -> dict:
        """Version simplifiée envoyée au LLM (sans les numéros de classe)."""
        return {p.name: {"role": p.role, "tone": list(p.tone), "relations": p.relations}
                for p in self.profiles.values()}


def load_characters(path: str | Path) -> CharacterRegistry:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Fiches personnages introuvables : {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    raw = data.get("characters")
    if not isinstance(raw, dict):
        raise ValueError(f"{path} doit contenir une clé 'characters' (voir characters.example.json)")

    profiles: dict[str, CharacterProfile] = {}
    seen_classes: dict[int, str] = {}
    for name, info in raw.items():
        yolo_class = info.get("yolo_class")
        if yolo_class is not None:
            yolo_class = int(yolo_class)
            if yolo_class in seen_classes:
                raise ValueError(f"Classe YOLO {yolo_class} attribuée à "
                                 f"'{seen_classes[yolo_class]}' et à '{name}'")
            seen_classes[yolo_class] = name
        profiles[name] = CharacterProfile(
            name=name,
            yolo_class=yolo_class,
            role=info.get("role", ""),
            tone=tuple(info.get("tone", [])),
            relations=dict(info.get("relations", {})),
        )
    return CharacterRegistry(profiles)
