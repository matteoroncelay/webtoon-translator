"""Chargement de la configuration (YAML) et des secrets (.env).

Principe :
- config/default.yaml contient TOUTES les valeurs par défaut ;
- un fichier utilisateur optionnel ne contient que les clés à modifier ;
- les deux sont fusionnés, puis convertis en dataclasses typées.
Une clé inconnue (faute de frappe) déclenche une erreur au lieu d'être ignorée.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


class ConfigError(ValueError):
    """Erreur dans un fichier de configuration."""


# --------------------------------------------------------------------------- #
# Sections de configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PathsConfig:
    character_model: Path
    bubble_model: Path
    bubble_seg_model: Path
    characters_file: Path
    font: str


@dataclass(frozen=True)
class GutterConfig:
    target_panel_height: int
    panel_min_height: int
    gutter_min_height: int
    white_threshold: int
    black_threshold: int
    min_uniform_ratio: float


@dataclass(frozen=True)
class DetectionConfig:
    imgsz: int
    iou: float
    character_conf: float
    bubble_conf: float


@dataclass(frozen=True)
class TailConfig:
    seg_imgsz: int
    seg_conf: float
    seg_iou: float
    crop_padding: int
    resample_step: float
    tip_window: float
    max_tip_angle: float
    base_search_length: float


@dataclass(frozen=True)
class PresenceConfig:
    strong_conf: float


@dataclass(frozen=True)
class SpeakerConfig:
    max_tip_distance: float
    max_angle_error: float
    near_distance: float
    weight_distance: float
    weight_angle: float


@dataclass(frozen=True)
class GeminiConfig:
    model: str
    batch_size: int
    max_retries: int
    base_delay: float
    temperature: float
    send_page_image: bool
    context_max_chars: int


@dataclass(frozen=True)
class RenderConfig:
    font_size: int
    min_font_size: int
    outline_width: int
    wrap_ratio: float
    erase_original: bool


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    gutters: GutterConfig
    detection: DetectionConfig
    tail: TailConfig
    presence: PresenceConfig
    speaker: SpeakerConfig
    gemini: GeminiConfig
    render: RenderConfig


@dataclass(frozen=True)
class Secrets:
    gemini_api_key: str | None
    gemini_api_key_fallback: str | None
    google_credentials: str | None


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #

def _deep_merge(base: dict, override: dict) -> dict:
    """Fusionne récursivement `override` par-dessus `base` (sans les modifier)."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _build(cls: type, data: Any, where: str) -> Any:
    """Construit récursivement la dataclass `cls` à partir d'un dict."""
    if not is_dataclass(cls):
        return data
    if not isinstance(data, dict):
        raise ConfigError(f"'{where}' doit être une section (dictionnaire).")

    expected = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(expected)
    if unknown:
        raise ConfigError(f"Clé(s) inconnue(s) dans '{where}' : {sorted(unknown)}")
    missing = set(expected) - set(data)
    if missing:
        raise ConfigError(f"Clé(s) manquante(s) dans '{where}' : {sorted(missing)}")

    values = {}
    for name, f in expected.items():
        field_type = _SECTIONS.get(name) if cls is Config else None
        if field_type is not None:
            values[name] = _build(field_type, data[name], name)
        elif f.type in ("Path", Path):
            values[name] = _resolve_path(data[name])
        else:
            values[name] = data[name]
    return cls(**values)


def _resolve_path(value: str) -> Path:
    """Les chemins relatifs sont résolus depuis la racine du projet."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


_SECTIONS = {
    "paths": PathsConfig,
    "gutters": GutterConfig,
    "detection": DetectionConfig,
    "tail": TailConfig,
    "presence": PresenceConfig,
    "speaker": SpeakerConfig,
    "gemini": GeminiConfig,
    "render": RenderConfig,
}


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"Fichier de configuration introuvable : {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} doit contenir des sections YAML.")
    return data


def load_config(user_config: str | Path | None = None) -> Config:
    """Charge default.yaml, puis applique le fichier utilisateur s'il est fourni."""
    data = _read_yaml(DEFAULT_CONFIG)
    if user_config is not None:
        data = _deep_merge(data, _read_yaml(Path(user_config)))
    return _build(Config, data, "config")


def gcloud_default_credentials() -> Path | None:
    """Fichier créé par `gcloud auth application-default login`, s'il existe.

    Google Cloud (donc l'OCR Vision) l'utilise automatiquement quand
    GOOGLE_APPLICATION_CREDENTIALS n'est pas défini.
    """
    if os.name == "nt":
        base = Path(os.getenv("APPDATA", ""))
    else:
        base = Path(os.getenv("XDG_CONFIG_HOME") or Path.home() / ".config")
    path = base / "gcloud" / "application_default_credentials.json"
    return path if path.is_file() else None


def load_secrets(env_file: str | Path | None = None) -> Secrets:
    """Lit les clés API depuis les variables d'environnement (et le fichier .env)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file or PROJECT_ROOT / ".env")
    except ImportError:
        pass  # python-dotenv absent : on se contente des variables d'environnement

    def get(name: str) -> str | None:
        value = os.getenv(name, "").strip()
        return value or None

    return Secrets(
        gemini_api_key=get("GEMINI_API_KEY"),
        gemini_api_key_fallback=get("GEMINI_API_KEY_FALLBACK"),
        google_credentials=get("GOOGLE_APPLICATION_CREDENTIALS"),
    )
