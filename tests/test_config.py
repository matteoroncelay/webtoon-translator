from pathlib import Path

import pytest

from webtoon_translator.config import ConfigError, PROJECT_ROOT, load_config


def test_default_config_loads():
    config = load_config()
    assert config.speaker.max_angle_error == 22.5
    assert config.presence.strong_conf == 0.75
    # Les chemins relatifs sont résolus depuis la racine du projet
    assert config.paths.bubble_model == PROJECT_ROOT / "models" / "detect_bubbles.pt"


def test_user_config_overrides_only_given_keys(tmp_path: Path):
    user = tmp_path / "user.yaml"
    user.write_text("speaker:\n  max_angle_error: 30.0\n", encoding="utf-8")
    config = load_config(user)
    assert config.speaker.max_angle_error == 30.0
    assert config.speaker.weight_angle == 2.0  # valeur par défaut conservée


def test_unknown_key_is_rejected(tmp_path: Path):
    user = tmp_path / "typo.yaml"
    user.write_text("speaker:\n  max_angle_eror: 30.0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="inconnue"):
        load_config(user)
