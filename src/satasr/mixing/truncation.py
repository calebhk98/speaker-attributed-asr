"""Cut an utterance short at a word boundary when a speaker is interrupted.

Implements design §4.7 steps 4-5: given word timestamps on the *clean* clip and
a cut time, keep whole words before the cut, drop any word the cut lands inside,
and mark a mid-word cut so the transcript shows an interrupted utterance.
"""

from __future__ import annotations

from dataclasses import dataclass

from satasr.core.models import Word

# Tolerance so a cut sitting exactly on a boundary counts the word as kept.
_EPS = 1e-6


@dataclass(frozen=True)
class TruncationResult:
    """Outcome of truncating one utterance."""

    kept_words: tuple[Word, ...]
    text: str
    truncated: bool  # were any words dropped?
    mid_word: bool  # did the cut land inside a word?


def truncate_words(
    words: tuple[Word, ...], cut_s: float, marker: str
) -> TruncationResult:
    """Keep words ending at/before ``cut_s``; drop the rest; mark mid-word cuts."""
    kept = tuple(w for w in words if w.end_s <= cut_s + _EPS)
    dropped = len(kept) < len(words)
    mid_word = any(w.start_s < cut_s < w.end_s for w in words)

    parts = [w.text for w in kept]
    if mid_word:
        parts.append(marker)

    return TruncationResult(
        kept_words=kept,
        text=" ".join(parts),
        truncated=dropped,
        mid_word=mid_word,
    )
