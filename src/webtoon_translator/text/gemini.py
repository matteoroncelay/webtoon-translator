"""Client Gemini unique, utilisé par la classification, le contexte et la traduction.

Avant : trois implémentations différentes, avec deux SDK différents
(google.generativeai ET google.genai), chacune avec sa propre gestion d'erreurs.
Maintenant : une seule classe, qui gère :
  - les erreurs temporaires (429 quota, 503 surcharge) : nouvelle tentative avec
    un délai qui DOUBLE à chaque fois (backoff exponentiel) ;
  - la clé de secours : si la clé principale reste bloquée par le quota ;
  - les réponses en JSON : demandées explicitement au modèle (response_mime_type),
    puis nettoyées et validées ;
  - les réponses vides ou bloquées : erreur claire au lieu d'un plantage plus loin.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any, TypeVar

from ..config import GeminiConfig

T = TypeVar("T")

# Signature du "transport" : (clé, modèle, contenus, config) -> texte de la réponse.
# Injectable, pour tester toute la logique sans appeler la vraie API.
Transport = Callable[[str, str, list[Any], GeminiConfig], str | None]


class GeminiError(RuntimeError):
    """Échec définitif d'un appel Gemini."""


class TransientError(Exception):
    """Erreur temporaire : on peut réessayer."""


def _is_transient(error: Exception) -> bool:
    message = str(error).upper()
    return any(code in message for code in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE",
                                            "DEADLINE_EXCEEDED", "500 INTERNAL"))


def google_transport(api_key: str, model: str, contents: list[Any], cfg: GeminiConfig) -> str | None:
    """Transport réel, avec le SDK officiel google-genai."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=cfg.temperature,
            response_mime_type="application/json",
        ),
    )
    return response.text


def parse_json(text: str) -> Any:
    """Parse une réponse JSON, en tolérant les balises ```json ... ``` éventuelles."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        cleaned = cleaned.rsplit("```", 1)[0]
    return json.loads(cleaned)


class GeminiClient:
    def __init__(self, cfg: GeminiConfig, api_key: str | None,
                 fallback_key: str | None = None,
                 transport: Transport = google_transport,
                 sleep: Callable[[float], None] = time.sleep):
        if not api_key:
            raise GeminiError("GEMINI_API_KEY manquante (voir .env.example)")
        self.cfg = cfg
        self.keys = [k for k in (api_key, fallback_key) if k]
        self.transport = transport
        self.sleep = sleep

    def generate_json(self, contents: list[Any]) -> Any:
        """Envoie la requête et renvoie le JSON décodé. Lève GeminiError si échec."""
        last_error: Exception | None = None
        for key in self.keys:
            delay = self.cfg.base_delay
            for attempt in range(1, self.cfg.max_retries + 1):
                try:
                    text = self.transport(key, self.cfg.model, contents, self.cfg)
                except Exception as error:  # noqa: BLE001 - on trie juste après
                    if not _is_transient(error):
                        raise GeminiError(f"Erreur Gemini : {error}") from error
                    last_error = error
                    print(f"   [Gemini] erreur temporaire ({attempt}/{self.cfg.max_retries}), "
                          f"nouvel essai dans {delay:.0f} s...")
                    self.sleep(delay)
                    delay *= 2
                    continue

                if not text:
                    raise GeminiError("Réponse vide (probablement bloquée par les filtres "
                                      "de sécurité du modèle)")
                try:
                    return parse_json(text)
                except json.JSONDecodeError as error:
                    last_error = error  # réponse mal formée : on retente
                    print(f"   [Gemini] JSON invalide ({attempt}/{self.cfg.max_retries})")
            if len(self.keys) > 1:
                print("   [Gemini] passage à la clé de secours")
        raise GeminiError(f"Échec après toutes les tentatives : {last_error}")


def batched(items: Sequence[T], size: int) -> Iterator[list[T]]:
    """Découpe une liste en lots de `size` éléments maximum."""
    for start in range(0, len(items), max(1, size)):
        yield list(items[start:start + size])
