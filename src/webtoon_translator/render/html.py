"""Page HTML de lecture en défilement continu, comme sur une application de webtoon."""

from __future__ import annotations

import html
from pathlib import Path

TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ margin: 0; background: #1a1a1a; color: #eee; font-family: system-ui, sans-serif; }}
  header, footer {{ text-align: center; padding: 20px; }}
  main {{ max-width: 800px; margin: 0 auto; }}
  img {{ display: block; width: 100%; height: auto; }}
</style>
</head>
<body>
<header><h1>{title}</h1></header>
<main>
{images}
</main>
<footer>Fin du chapitre — traduit automatiquement par webtoon-translator</footer>
</body>
</html>
"""


def write_reader(image_paths: list[Path], html_path: str | Path, title: str) -> None:
    """Écrit la page HTML. Les chemins des images sont relatifs au fichier HTML."""
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    tags = "\n".join(
        f'<img src="{html.escape(p.relative_to(html_path.parent).as_posix())}" '
        f'alt="{html.escape(p.name)}" loading="lazy">'
        for p in image_paths
    )
    html_path.write_text(TEMPLATE.format(title=html.escape(title), images=tags), encoding="utf-8")
