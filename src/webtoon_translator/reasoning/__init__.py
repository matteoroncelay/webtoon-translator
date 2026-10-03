"""Raisonnement : présence des personnages, attribution des locuteurs."""

from .attribution import attribute_page
from .characters import CharacterProfile, CharacterRegistry, load_characters
from .presence import compute_presence, in_scene, update_presence
from .speaker import SpeakerDecision, assign_speaker, cone_candidate

__all__ = [
    "attribute_page",
    "CharacterProfile", "CharacterRegistry", "load_characters",
    "compute_presence", "in_scene", "update_presence",
    "SpeakerDecision", "assign_speaker", "cone_candidate",
]
