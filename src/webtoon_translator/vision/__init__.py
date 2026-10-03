"""Vision : découpage en panneaux, détection YOLO, analyse de la queue des bulles."""

from .detector import YoloDetector, detect_bubbles, detect_characters
from .panels import find_gutters, panel_ranges
from .tail import TailInfo, YoloSegmenter, analyze_tail

__all__ = [
    "YoloDetector", "detect_bubbles", "detect_characters",
    "find_gutters", "panel_ranges",
    "TailInfo", "YoloSegmenter", "analyze_tail",
]
