"""Rendu : écriture du texte traduit dans les bulles, page de lecture HTML."""

from .html import write_reader
from .text_render import render_page

__all__ = ["render_page", "write_reader"]
