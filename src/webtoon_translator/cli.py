"""Point d'entrée en ligne de commande.

--translate DOSSIER : traduit les images du dossier (résultat dans --out)
--resume            : reprend une traduction interrompue
--evaluate CSV      : précision de chaque règle d'attribution (rapport annoté à la main)
--check             : vérifie l'installation
--debug-vision      : dessine bulles et queues détectées sur des pages
--figure            : génère docs/attribution.png (schéma de l'attribution des locuteurs)
"""

from __future__ import annotations

import argparse
import sys

from .config import ConfigError, gcloud_default_credentials, load_config, load_secrets


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
    if secrets.gemini_api_key:
        print("[OK] GEMINI_API_KEY définie.")
    else:
        print("[MANQUANT] GEMINI_API_KEY (voir .env.example)")
        problems += 1

    adc = gcloud_default_credentials()
    if secrets.google_credentials:
        print("[OK] Google Vision : GOOGLE_APPLICATION_CREDENTIALS définie.")
    elif adc:
        print(f"[OK] Google Vision : connexion gcloud trouvée ({adc})")
    else:
        print("[MANQUANT] Google Vision : ni GOOGLE_APPLICATION_CREDENTIALS, "
              "ni connexion `gcloud auth application-default login`")
        problems += 1

    print("\nTout est prêt." if problems == 0 else f"\n{problems} élément(s) à régler.")
    return 0 if problems == 0 else 1


def translate(input_dir: str, out: str | None, config_path: str | None, resume: bool) -> int:
    from pathlib import Path

    from .pipeline import build_components, run_pipeline
    from .text import GeminiError
    output = Path(out or f"outputs/{Path(input_dir).name}")
    try:
        config = load_config(config_path)
        components = build_components(config, load_secrets())
        run_pipeline(input_dir, output, config, components, resume=resume)
    except (ConfigError, FileNotFoundError, ValueError, GeminiError) as e:
        print(f"[ERREUR] {e}")
        if (output / "chapter.json").is_file():
            print("Astuce : relance avec --resume pour reprendre là où ça s'est arrêté.")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="webtoon-translator",
        description="Traduction automatique de webtoons coréens vers le français.",
    )
    parser.add_argument("--config", help="Fichier YAML qui surcharge config/default.yaml")
    parser.add_argument("--check", action="store_true", help="Vérifier l'installation")
    parser.add_argument("--debug-vision", metavar="DOSSIER",
                        help="Annoter les bulles et queues détectées sur les images du dossier")
    parser.add_argument("--translate", metavar="DOSSIER", help="Traduire les images du dossier")
    parser.add_argument("--resume", action="store_true",
                        help="Reprendre une traduction interrompue (avec --translate)")
    parser.add_argument("--evaluate", metavar="CSV",
                        help="Calculer la précision à partir d'un rapport.csv annoté")
    parser.add_argument("--figure", action="store_true",
                        help="Générer le schéma de l'attribution des locuteurs (docs/attribution.png)")
    parser.add_argument("--out", help="Dossier de sortie (défaut : outputs/<nom du dossier>)")
    args = parser.parse_args(argv)

    if args.figure:
        from .figure import attribution_figure
        path = args.out or "docs/attribution.png"
        for d in attribution_figure(path, load_config(args.config)):
            print(f"  décision : classe {d.class_id} ({d.source.value})")
        print(f"Figure enregistrée : {path}")
        return 0

    if args.translate:
        return translate(args.translate, args.out, args.config, args.resume)

    if args.evaluate:
        from .io.report import evaluate_report
        results = evaluate_report(args.evaluate)
        if not results:
            print("Aucune ligne annotée (colonne 'correct' : 1 ou 0).")
            return 1
        for source, (ok, n) in sorted(results.items()):
            print(f"{source:12s} : {ok}/{n} corrects ({100 * ok / n:.0f} %)")
        return 0

    if args.check:
        return check(args.config)

    if args.debug_vision:
        from .debug import debug_vision  # import tardif : charge OpenCV et YOLO
        try:
            debug_vision(load_config(args.config), args.debug_vision, args.out or "outputs/debug")
        except (ConfigError, FileNotFoundError, ValueError) as e:
            print(f"[ERREUR] {e}")
            return 1
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
