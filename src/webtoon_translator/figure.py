"""Figure explicative de l'attribution des locuteurs (pour le README).

Les trois scènes sont dessinées, mais les DÉCISIONS affichées ne sont pas écrites
à la main : elles sont calculées par la vraie fonction `assign_speaker`.
La figure montre donc ce que fait réellement le code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import Config
from .models import BBox, CharacterDetection, SpeakerSource
from .reasoning.speaker import SpeakerDecision, assign_speaker
from .render.text_render import load_font
from .vision.tail import TailInfo

W, H = 600, 600
BUBBLE_C, BUBBLE_R = (210.0, 150.0), (120.0, 58.0)  # centre et demi-axes de la bulle
COLORS = {0: (214, 96, 77), 1: (67, 147, 195)}   # A = rouge, B = bleu
NAMES = {0: "A", 1: "B"}
SOURCE_LABEL = {
    SpeakerSource.GEOMETRY: "géométrie",
    SpeakerSource.SINGLE_PRESENT: "seul personnage présent",
    SpeakerSource.ELIMINATION: "élimination",
    SpeakerSource.NONE: "inconnu -> LLM",
}


@dataclass
class Scene:
    title: str
    explanation: str
    tail: TailInfo | None
    detected: list[CharacterDetection]      # ce que voit le détecteur
    hidden: dict[int, tuple[float, float]]  # personnages présents mais NON détectés
    presence: dict[int, int]


def _char(class_id: int, cx: float, cy: float) -> CharacterDetection:
    return CharacterDetection(class_id, BBox(cx - 45, cy - 85, cx + 45, cy + 85), 0.9)


def _tail_base(tip: tuple[float, float]) -> tuple[float, float]:
    """Point du bord de la bulle d'où part la queue (dans la direction de la pointe)."""
    (cx, cy), (a, b) = BUBBLE_C, BUBBLE_R
    t = math.atan2((tip[1] - cy) / b, (tip[0] - cx) / a)
    return cx + a * math.cos(t), cy + b * math.sin(t)


def _tail(tip: tuple[float, float]) -> TailInfo:
    """Queue dont la direction (base -> pointe) est cohérente avec le dessin."""
    bx, by = _tail_base(tip)
    angle = math.degrees(math.atan2(tip[1] - by, tip[0] - bx)) % 360
    return TailInfo(tip=tip, base_center=(bx, by), angle_deg=angle, tip_angle_deg=35.0,
                    contour=np.zeros((0, 2)))


def _on_ray(tail: TailInfo, dist: float, offset_deg: float = 0.0) -> tuple[float, float]:
    """Point à `dist` px de la pointe, dans la direction de la queue (+ décalage angulaire)."""
    a = math.radians(tail.angle_deg + offset_deg)
    return tail.tip[0] + dist * math.cos(a), tail.tip[1] + dist * math.sin(a)


def scenes() -> list[Scene]:
    t1 = _tail((300.0, 225.0))
    t3 = _tail((350.0, 175.0))
    return [
        Scene("1. Géométrie",
              "Le personnage dans le cône de la queue,\nle plus proche de l'axe, est choisi.",
              t1, [_char(0, *_on_ray(t1, 190)), _char(1, 110, 380)], {}, {0: 3, 1: 3}),
        Scene("2. Seul personnage présent",
              "Pas de queue exploitable, mais un seul\npersonnage dans la scène : c'est lui.",
              None, [_char(0, 300, 380)], {}, {0: 3, 1: 1}),
        Scene("3. Élimination",
              "A est détecté près de la bulle mais hors du\ncône : la bulle vise B, présent mais non détecté.",
              t3, [_char(0, 150, 380)], {1: _on_ray(t3, 140)}, {0: 3, 1: 2}),
    ]


def _dashed_line(draw, p, q, color, dash=9, width=3) -> None:
    length = math.dist(p, q)
    for start in np.arange(0, length, 2 * dash):
        a, b = start / length, min(start + dash, length) / length
        draw.line((p[0] + a * (q[0] - p[0]), p[1] + a * (q[1] - p[1]),
                   p[0] + b * (q[0] - p[0]), p[1] + b * (q[1] - p[1])), fill=color, width=width)


def _draw_person(draw: ImageDraw.ImageDraw, cx: float, cy: float, color, label: str,
                 dashed: bool = False) -> None:
    font = load_font("arial.ttf", 20)
    head = (cx - 26, cy - 85, cx + 26, cy - 33)
    body = (cx - 38, cy - 25, cx + 38, cy + 80)
    if dashed:  # silhouette en pointillés : personnage NON détecté
        for i in range(0, 360, 30):
            draw.arc(head, i, i + 15, fill=color, width=3)
        x1, y1, x2, y2 = body
        for p, q in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                     ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
            _dashed_line(draw, p, q, color)
    else:
        draw.ellipse(head, fill=color)
        draw.rounded_rectangle(body, radius=16, fill=color)
    draw.text((cx, cy + 92), label, font=font, fill=color, anchor="mt")


def _draw_scene(scene: Scene, decision: SpeakerDecision, config: Config) -> Image.Image:
    img = Image.new("RGB", (W, H), (255, 255, 255))
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(img)
    title_font = load_font("arial.ttf", 26)

    # Cône de la queue
    if scene.tail is not None:
        tx, ty = scene.tail.tip
        r = 330
        half = config.speaker.max_angle_error
        pts = [(tx, ty)] + [(tx + r * math.cos(math.radians(scene.tail.angle_deg + a)),
                             ty + r * math.sin(math.radians(scene.tail.angle_deg + a)))
                            for a in np.linspace(-half, half, 12)]
        od.polygon(pts, fill=(255, 200, 0, 70))
    img.paste(overlay, (0, 0), overlay)

    # Bulle : on dessine la forme en noir un peu plus grande, puis en blanc par-dessus,
    # ce qui donne un contour continu entre la bulle et sa queue.
    (cx, cy), (ra, rb) = BUBBLE_C, BUBBLE_R
    tail_poly = None
    if scene.tail is not None:
        tx, ty = scene.tail.tip
        bx, by = _tail_base(scene.tail.tip)
        ux, uy = (tx - bx), (ty - by)
        n = math.hypot(ux, uy)
        ux, uy = ux / n, uy / n
        px, py = -uy, ux
        inner = (bx - 4 * ux, by - 4 * uy)  # la base rentre dans la bulle
        tail_poly = [(inner[0] + 20 * px, inner[1] + 20 * py),
                     (inner[0] - 20 * px, inner[1] - 20 * py), (tx, ty)]
    for color, grow in (("black", 3), ("white", 0)):
        draw.ellipse((cx - ra - grow, cy - rb - grow, cx + ra + grow, cy + rb + grow), fill=color)
        if tail_poly:
            gx, gy = sum(p[0] for p in tail_poly) / 3, sum(p[1] for p in tail_poly) / 3
            k = 1 + grow / 25
            draw.polygon([(gx + k * (x - gx), gy + k * (y - gy)) for x, y in tail_poly], fill=color)
    if scene.tail is not None:
        tx, ty = scene.tail.tip
        draw.ellipse((tx - 7, ty - 7, tx + 7, ty + 7), fill=(220, 0, 0))
    draw.text((cx, cy), "안녕하세요", font=load_font("malgun.ttf", 26), fill="black", anchor="mm")

    # Personnages
    for char in scene.detected:
        x, y = char.bbox.center
        _draw_person(draw, x, y, COLORS[char.class_id], f"{NAMES[char.class_id]} (détecté)")
    for class_id, (x, y) in scene.hidden.items():
        _draw_person(draw, x, y, COLORS[class_id], f"{NAMES[class_id]} (non détecté)", dashed=True)

    # Textes
    draw.text((20, 18), scene.title, font=title_font, fill="black")
    who = NAMES.get(decision.class_id, "?")
    draw.text((20, H - 58), f"Locuteur : {who}  ({SOURCE_LABEL[decision.source]})",
              font=title_font, fill=COLORS.get(decision.class_id, (0, 0, 0)))
    return img


def attribution_figure(path: str | Path, config: Config) -> list[SpeakerDecision]:
    """Dessine les trois scènes côte à côte et renvoie les décisions calculées."""
    panels, decisions = [], []
    caption_font = load_font("arial.ttf", 18)
    for scene in scenes():
        decision = assign_speaker(scene.tail, scene.detected, scene.presence, config.speaker)
        decisions.append(decision)
        panel = _draw_scene(scene, decision, config)
        framed = Image.new("RGB", (W, H + 70), (255, 255, 255))
        framed.paste(panel, (0, 0))
        ImageDraw.Draw(framed).multiline_text((20, H + 8), scene.explanation,
                                              font=caption_font, fill=(80, 80, 80), spacing=4)
        panels.append(framed)

    gap = 20
    figure = Image.new("RGB", (len(panels) * W + (len(panels) - 1) * gap, H + 70), (230, 230, 230))
    for i, panel in enumerate(panels):
        figure.paste(panel, (i * (W + gap), 0))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.save(path)
    return decisions
