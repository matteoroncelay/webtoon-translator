"""Point d'entrée en ligne de commande.

Pour l'instant (étape 1) : seulement `--check`, qui vérifie l'installation.
La commande de traduction sera ajoutée à l'étape 6.
"""

from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_config, load_secrets


def check(config_path: str | None) -> int:
    """Vérifie la config, les modèles et les clés. Renvoie le code de sortie."""
    try:
        config = load_config(config_path)
    except ConfigError as e:
        print(f"[ERREUR] {e}")
        return 1
    print("[OK] Configuration chargée.")

    problems = 0
    for label, path in [
        ("Modèle bulles", config.paths.bubble_model),
        ("Modèle segmentation", config.paths.bubble_seg_model),
        ("Modèle personnages", config.paths.character_model),
        ("Fiches personnages", config.paths.characters_file),
    ]:
        if path.is_file():
            print(f"[OK] {label} : {path}")
        else:
            print(f"[MANQUANT] {label} : {path}")
            problems += 1

    secrets = load_secrets()
    for label, value in [
        ("GEMINI_API_KEY", secrets.gemini_api_key),
        ("GOOGLE_APPLICATION_CREDENTIALS", secrets.google_credentials),
    ]:
        if value:
            print(f"[OK] {label} définie.")
        else:
            print(f"[MANQUANT] {label} (voir .env.example)")
            problems += 1

    print("\nTout est prêt." if problems == 0 else f"\n{problems} élément(s) à régler.")
    return 0 if problems == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="webtoon-translator",
        description="Traduction automatique de webtoons coréens vers le français.",
    )
    parser.add_argument("--config", help="Fichier YAML qui surcharge config/default.yaml")
    parser.add_argument("--check", action="store_true", help="Vérifier l'installation")
    args = parser.parse_args(argv)

    if args.check:
        return check(args.config)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
