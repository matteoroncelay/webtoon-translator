"""Export des résultats bulle par bulle, pour les évaluer à la main.

Le CSV contient une colonne `correct` vide : ouvre-le dans Excel/LibreOffice,
mets 1 (juste) ou 0 (faux) pour chaque bulle, puis lance `evaluate_report`
pour obtenir la précision de chaque méthode d'attribution.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from ..models import Chapter

COLUMNS = ["page", "bubble_id", "kind", "type", "speaker", "speaker_source",
           "x1", "y1", "x2", "y2", "correct"]


def write_report(chapter: Chapter, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig : Excel sous Windows lit correctement les accents
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(COLUMNS)
        for page in chapter.pages:
            for b in page.bubbles:
                writer.writerow([page.file_name, b.id, b.kind.name, b.type.value,
                                 b.speaker or "", b.speaker_source.value,
                                 *[round(v) for v in b.bbox.to_list()], ""])


def evaluate_report(path: str | Path) -> dict[str, tuple[int, int]]:
    """Précision par méthode : {source: (nb_corrects, nb_annotés)}."""
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            value = (row.get("correct") or "").strip()
            if value not in {"0", "1"}:
                continue
            totals[row["speaker_source"]][0] += int(value)
            totals[row["speaker_source"]][1] += 1
    return {source: (ok, n) for source, (ok, n) in totals.items()}
