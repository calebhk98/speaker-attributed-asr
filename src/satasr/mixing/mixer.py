"""Render a schedule of clips into one mixed clip plus its serialized transcript.

:class:`OverlapMixer` is the concrete :class:`~satasr.core.interfaces.Mixer`.
It delegates timing to :class:`~satasr.mixing.overlap.OverlapScheduler` and cut
bookkeeping to :func:`~satasr.mixing.truncation.truncate_words`, then sums the
audio and reports the two speaker counts the design tracks (§4.6).
"""

from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np

from satasr.core.audio import AudioBuffer
from satasr.core.config import MixingConfig
from satasr.core.models import (
    MixedClip,
    PlacedUtterance,
    SpeakerClip,
    SpeakerCounts,
    Word,
)
from satasr.mixing.overlap import OverlapScheduler, Placement
from satasr.mixing.truncation import truncate_words


class OverlapMixer:
    """Combine single-speaker clips into an overlapping multi-speaker example."""

    def __init__(self, config: MixingConfig, rng: random.Random) -> None:
        self._config = config
        self._rng = rng
        self._scheduler = OverlapScheduler(config)

    def mix(self, clips: Iterable[SpeakerClip], counts: SpeakerCounts) -> MixedClip:
        ordered = list(clips)
        if not ordered:
            raise ValueError("cannot mix zero clips")

        durations = [clip.audio.duration_s for clip in ordered]
        placements = self._scheduler.schedule(
            durations, counts.max_simultaneous, self._rng
        )
        audio = self._render(ordered, placements)
        utterances = tuple(self._to_utterance(ordered[p.index], p) for p in placements)
        return MixedClip(audio, utterances, self._realized_counts(ordered, placements))

    def _render(
        self, clips: list[SpeakerClip], placements: list[Placement]
    ) -> AudioBuffer:
        sample_rate = clips[0].audio.sample_rate
        total = self._total_samples(clips, placements, sample_rate)
        canvas = np.zeros(total, dtype=np.float32)
        for placement in placements:
            self._add_segment(canvas, clips[placement.index], placement, sample_rate)
        np.clip(canvas, -1.0, 1.0, out=canvas)  # keep within valid audio range
        return AudioBuffer(canvas, sample_rate)

    def _add_segment(
        self,
        canvas: np.ndarray,
        clip: SpeakerClip,
        placement: Placement,
        sample_rate: int,
    ) -> None:
        kept = self._kept_samples(clip, placement, sample_rate)
        start = round(placement.offset_s * sample_rate)
        segment = clip.audio.samples[:kept]
        canvas[start : start + segment.shape[0]] += segment

    def _to_utterance(self, clip: SpeakerClip, placement: Placement) -> PlacedUtterance:
        kept_s = self._kept_seconds(clip, placement)
        end_s = placement.offset_s + kept_s
        if not clip.words:
            truncated = placement.cut_at_s is not None
            return PlacedUtterance(
                clip.speaker_id, placement.offset_s, end_s, clip.text, (), truncated
            )

        cut_s = placement.cut_at_s if placement.cut_at_s is not None else kept_s + 1.0
        result = truncate_words(clip.words, cut_s, self._config.truncation_marker)
        shifted = tuple(
            self._shift(word, placement.offset_s) for word in result.kept_words
        )
        return PlacedUtterance(
            clip.speaker_id,
            placement.offset_s,
            end_s,
            result.text,
            shifted,
            result.truncated,
        )

    def _realized_counts(
        self, clips: list[SpeakerClip], placements: list[Placement]
    ) -> SpeakerCounts:
        total = len({clip.speaker_id for clip in clips})
        intervals = [
            (p.offset_s, p.offset_s + self._kept_seconds(clips[p.index], p))
            for p in placements
        ]
        return SpeakerCounts(total=total, max_simultaneous=_peak_overlap(intervals))

    def _kept_seconds(self, clip: SpeakerClip, placement: Placement) -> float:
        if placement.cut_at_s is None:
            return clip.audio.duration_s
        return min(clip.audio.duration_s, placement.cut_at_s)

    def _kept_samples(self, clip: SpeakerClip, placement: Placement, rate: int) -> int:
        return round(self._kept_seconds(clip, placement) * rate)

    def _total_samples(
        self, clips: list[SpeakerClip], placements: list[Placement], rate: int
    ) -> int:
        ends = [
            round(p.offset_s * rate) + self._kept_samples(clips[p.index], p, rate)
            for p in placements
        ]
        return max(ends)

    @staticmethod
    def _shift(word: Word, offset_s: float) -> Word:
        return Word(word.text, word.start_s + offset_s, word.end_s + offset_s)


def _peak_overlap(intervals: list[tuple[float, float]]) -> int:
    """Most intervals active at any instant. A clip ending as another starts
    does not count as overlap (ends are processed before starts)."""
    events: list[tuple[float, int]] = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda event: (event[0], event[1]))

    peak = 0
    active = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    return peak
