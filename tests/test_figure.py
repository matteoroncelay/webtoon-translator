from webtoon_translator.config import load_config
from webtoon_translator.figure import attribution_figure
from webtoon_translator.models import SpeakerSource


def test_figure_shows_the_three_rules(tmp_path):
    """Les décisions affichées sur la figure sont celles du vrai code."""
    decisions = attribution_figure(tmp_path / "fig.png", load_config())
    assert [(d.class_id, d.source) for d in decisions] == [
        (0, SpeakerSource.GEOMETRY),
        (0, SpeakerSource.SINGLE_PRESENT),
        (1, SpeakerSource.ELIMINATION),
    ]
    assert (tmp_path / "fig.png").stat().st_size > 0
