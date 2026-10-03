"""Outils de débogage visuel : dessine ce que voit le pipeline sur les pages."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .io import list_images, read_image, write_image
from .models import BubbleKind
from .vision import YoloDetector, YoloSegmenter, analyze_tail, detect_bubbles

COLORS = {  # BGR
    BubbleKind.WITH_TAIL: (0, 180, 0),     # vert
    BubbleKind.NO_TAIL: (200, 120, 0),     # bleu
    BubbleKind.NARRATION: (128, 128, 128), # gris
}
TIP_COLOR = (0, 0, 255)          # rouge
CONTOUR_COLOR = (0, 200, 255)    # orange


def debug_vision(config: Config, input_dir: str | Path, output_dir: str | Path) -> None:
    """Détecte les bulles et analyse les queues, puis enregistre des images annotées.

    Légende : boîte verte = bulle avec queue, bleue = sans queue, grise = narration ;
    contour orange = contour utilisé ; point rouge + flèche = pointe et direction.
    """
    detector = YoloDetector(config.paths.bubble_model, config.detection.imgsz,
                            config.detection.iou)
    segmenter = YoloSegmenter(config.paths.bubble_seg_model, config.tail)
    output_dir = Path(output_dir)

    total_tail_bubbles = total_found = 0
    for path in list_images(input_dir):
        image = read_image(path)
        canvas = image.copy()
        bubbles = detect_bubbles(image, detector, config)
        found = 0
        tail_bubbles = [b for b in bubbles if b.kind is BubbleKind.WITH_TAIL]

        for bubble in bubbles:
            b = bubble.bbox
            cv2.rectangle(canvas, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)),
                          COLORS[bubble.kind], 3)
            if bubble.kind is not BubbleKind.WITH_TAIL:
                continue
            info = analyze_tail(image, b, config.tail, segmenter)
            if info is None:
                cv2.putText(canvas, "queue ?", (int(b.x1), int(b.y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, TIP_COLOR, 2)
                continue
            found += 1
            cv2.polylines(canvas, [info.contour.astype(np.int32)], True, CONTOUR_COLOR, 2)
            tip = (int(info.tip[0]), int(info.tip[1]))
            cv2.circle(canvas, tip, 8, TIP_COLOR, -1)
            rad = math.radians(info.angle_deg)
            end = (int(tip[0] + 120 * math.cos(rad)), int(tip[1] + 120 * math.sin(rad)))
            cv2.arrowedLine(canvas, tip, end, TIP_COLOR, 3, tipLength=0.25)

        write_image(output_dir / path.name, canvas)
        total_tail_bubbles += len(tail_bubbles)
        total_found += found
        print(f"{path.name} : {len(bubbles)} bulles, "
              f"queue trouvée pour {found}/{len(tail_bubbles)} bulles à queue")

    print(f"\nTotal : queue trouvée pour {total_found}/{total_tail_bubbles} bulles à queue.")
    print(f"Images annotées dans : {output_dir}")
