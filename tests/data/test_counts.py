"""Fast tests for the shared speaker-count sweep (design §4.6)."""

from __future__ import annotations

from satasr.core.models import PlacedUtterance
from satasr.data.counts import speaker_counts


def test_counts_distinct_speakers_and_peak_overlap() -> None:
    utterances = [
        PlacedUtterance("a", 0.0, 2.0, "hi"),
        PlacedUtterance("b", 1.0, 3.0, "there"),
        PlacedUtterance("a", 4.0, 5.0, "again"),
    ]

    counts = speaker_counts(utterances)

    assert counts.total == 2
    assert counts.max_simultaneous == 2


def test_no_overlap_when_speakers_take_turns() -> None:
    utterances = [
        PlacedUtterance("a", 0.0, 1.0, "hi"),
        PlacedUtterance("b", 1.0, 2.0, "there"),
    ]

    counts = speaker_counts(utterances)

    assert counts.total == 2
    assert counts.max_simultaneous == 1
