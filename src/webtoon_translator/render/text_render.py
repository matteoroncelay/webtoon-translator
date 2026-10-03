"""Écriture du texte traduit dans les bulles.

Améliorations par rapport à render_bubbles_on_pages.py :
  - le texte coréen est recouvert (option `erase_original`) au lieu d'écrire
    le français PAR-DESSUS ;
  - la taille de police s'adapte : on la réduit tant que le texte ne tient pas
    dans la bulle (avant : taille fixe, le texte débordait sur les longues phrases) ;
  - la police est cherchée aussi dans le dossier des polices de Windows.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import RenderConfig
from ..models import Bubble, BubbleType

WINDOWS_FONTS = Path("C:/Windows/Fonts")


@lru_cache(maxsize=64)
def load_font(font: str, size: int) -> ImageFont.ImageFont:
    for candidate in (Path(font), WINDOWS_FONTS / font):
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: float,
              draw: ImageDraw.ImageDraw) -> list[str]:
    """Coupe le texte en lignes qui tiennent dans `max_width` pixels."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def fit_text(text: str, box_w: float, box_h: float, font_name: str, cfg: RenderConfig,
             draw: ImageDraw.ImageDraw) -> tuple[ImageFont.ImageFont, str]:
    """Plus grande taille de police (<= font_size) pour laquelle le texte tient dans la boîte."""
    size = cfg.font_size
    while True:
        font = load_font(font_name, size)
        block = "\n".join(wrap_text(text, font, box_w, draw))
        x1, y1, x2, y2 = draw.multiline_textbbox((0, 0), block, font=font, align="center",
                                                 stroke_width=cfg.outline_width)
        if (x2 - x1 <= box_w and y2 - y1 <= box_h) or size <= cfg.min_font_size:
            return font, block
        size -= 2


def render_page(page_bgr: np.ndarray, bubbles: list[Bubble], font_name: str,
                cfg: RenderConfig) -> np.ndarray:
    """Renvoie une copie de la page avec le texte français dans les bulles."""
    image = Image.fromarray(page_bgr[:, :, ::-1].copy())  # BGR -> RGB
    draw = ImageDraw.Draw(image)

    for bubble in bubbles:
        text = bubble.text_fr.strip()
        if not text:
            continue
        b = bubble.bbox
        box_w, box_h = b.width * cfg.wrap_ratio, b.height * cfg.wrap_ratio

        if cfg.erase_original:
            # Rectangle arrondi légèrement en retrait, pour ne pas effacer le contour de la bulle
            inset_x, inset_y = b.width * 0.08, b.height * 0.08
            radius = int(min(b.width, b.height) * 0.3)
            draw.rounded_rectangle((b.x1 + inset_x, b.y1 + inset_y, b.x2 - inset_x, b.y2 - inset_y),
                                   radius=radius, fill=(255, 255, 255))

        font, block = fit_text(text, box_w, box_h, font_name, cfg, draw)
        x1, y1, x2, y2 = draw.multiline_textbbox((0, 0), block, font=font, align="center",
                                                 stroke_width=cfg.outline_width)
        cx, cy = b.center
        position = (cx - (x2 - x1) / 2 - x1, cy - (y2 - y1) / 2 - y1)

        italic_like = bubble.type is BubbleType.THOUGHT  # pas d'italique dispo : couleur adoucie
        fill = (70, 70, 70) if italic_like else (20, 20, 20)
        if cfg.erase_original:
            draw.multiline_text(position, block, font=font, fill=fill, align="center")
        else:  # ancien style : texte blanc à contour noir, par-dessus l'original
            draw.multiline_text(position, block, font=font, fill=(255, 255, 255), align="center",
                                stroke_width=cfg.outline_width, stroke_fill=(0, 0, 0))

    return np.asarray(image)[:, :, ::-1].copy()  # RGB -> BGR
