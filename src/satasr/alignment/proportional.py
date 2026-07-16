"""Reference aligner: proportional word-timing, no ML required.

This is the offline stand-in used with the sine reference engine and in CI —
it never downloads weights and runs in microseconds. Real pipelines prefer
CTC-segmentation for synthetic (TTS) audio, or classical MFA for real
recordings (design §4.7); see ``ctc_segmentation.py`` for that path. This
aligner instead splits text into words and gives each one a slice of the clip
duration proportional to its character length, so timestamps are contiguous,
monotonic, and always span the whole buffer.
"""

from __future__ import annotations

from satasr.alignment.registry import ALIGNERS
from satasr.core.audio import AudioBuffer
from satasr.core.models import Word


@ALIGNERS.register("proportional")
class ProportionalAligner:
    """Splits text into words and places them back-to-back across the audio."""

    def align(self, audio: AudioBuffer, text: str) -> tuple[Word, ...]:
        tokens = text.split()
        if not tokens:
            raise ValueError("cannot align empty text")

        weights = [len(token) for token in tokens]
        total_weight = sum(weights)
        words = self._place(tokens, weights, total_weight, audio.duration_s)
        return tuple(words)

    @staticmethod
    def _place(
        tokens: list[str], weights: list[int], total_weight: int, duration_s: float
    ) -> list[Word]:
        """Walk tokens left to right, giving each a duration_s * weight share."""
        words: list[Word] = []
        cursor = 0.0
        for token, weight in zip(tokens, weights, strict=True):
            end = cursor + duration_s * weight / total_weight
            words.append(Word(token, cursor, end))
            cursor = end
        return _snap_last_end(words, duration_s)


def _snap_last_end(words: list[Word], duration_s: float) -> list[Word]:
    """Correct float drift so the last word ends exactly at the clip length."""
    last = words[-1]
    words[-1] = Word(last.text, last.start_s, duration_s)
    return words
