"""Sauvegarde et chargement de l'état d'un chapitre.

Avant : 9 fichiers JSON intermédiaires différents (detections_perso, presence,
bubbles_full_data, bubbles_merged, temp_complete_text...).
Maintenant : un seul fichier `chapter.json`, enrichi à chaque étape.
Écrit de façon atomique : en cas de plantage, l'ancien fichier reste intact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..models import Chapter


def save_chapter(chapter: Chapter, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(chapter.to_dict(), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)  # remplacement atomique


def load_chapter(path: str | Path) -> Chapter:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Fichier de chapitre introuvable : {path}")
    with path.open(encoding="utf-8") as f:
        return Chapter.from_dict(json.load(f))
