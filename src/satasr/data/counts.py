"""Speaker-count bookkeeping shared by every real-corpus loader (design §4.6).

The two counts the design insists on — distinct identities and peak
simultaneous overlap — are computed from a recording's utterance timeline the
same way regardless of source corpus, so this lives once here instead of being
reimplemented per loader (rule 4, single source of truth).
"""

from __future__ import annotations

from collections.abc import Iterable

from satasr.core.models import PlacedUtterance, SpeakerCounts


def speaker_counts(utterances: Iterable[PlacedUtterance]) -> SpeakerCounts:
    """Distinct speakers and peak overlap across a recording's utterances."""
    ordered = list(utterances)
    total = len({utterance.speaker_id for utterance in ordered})
    intervals = [(u.start_s, u.end_s) for u in ordered]
    return SpeakerCounts(total=total, max_simultaneous=_peak_overlap(intervals))


def _peak_overlap(intervals: list[tuple[float, float]]) -> int:
    """Sweep-line max concurrently-active intervals. A clip ending exactly as
    another starts is not counted as overlap (ends processed before starts)."""
    events: list[tuple[float, int]] = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda event: (event[0], event[1]))

    peak = active = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    return peak
