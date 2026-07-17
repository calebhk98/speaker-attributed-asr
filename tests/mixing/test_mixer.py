"""Tests for the overlap mixer that produces training examples (§4.6)."""

from __future__ import annotations

import random

import numpy as np

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.config import MixingConfig
from satasr.core.models import SpeakerClip, SpeakerCounts, Word
from satasr.mixing.mixer import OverlapMixer


def _clip(speaker: str, duration_s: float, level: float = 0.5) -> SpeakerClip:
    n = round(duration_s * SAMPLE_RATE)
    audio = AudioBuffer(np.full(n, level, dtype=np.float32))
    # One word per 0.5s so truncation has boundaries to land on.
    words = tuple(
        Word(f"{speaker}{i}", i * 0.5, (i + 1) * 0.5)
        for i in range(max(1, int(duration_s / 0.5)))
    )
    return SpeakerClip(
        audio=audio,
        text=" ".join(w.text for w in words),
        speaker_id=speaker,
        words=words,
    )


def _mixer(seed: int) -> OverlapMixer:
    return OverlapMixer(MixingConfig(), random.Random(seed))


def test_mix_reports_total_speakers() -> None:
    clips = [_clip("A", 2.0), _clip("B", 2.0), _clip("A", 1.5)]
    mixed = _mixer(1).mix(clips, SpeakerCounts(total=2, max_simultaneous=2))
    assert mixed.counts.total == 2  # two distinct identities, three clips


def test_mix_never_exceeds_target_overlap() -> None:
    clips = [_clip(s, 2.0) for s in "ABCD"]
    target = SpeakerCounts(total=4, max_simultaneous=2)
    mixed = _mixer(3).mix(clips, target)
    assert mixed.counts.max_simultaneous <= 2


def test_single_clip_has_no_overlap() -> None:
    mixed = _mixer(0).mix([_clip("A", 1.0)], SpeakerCounts(total=1, max_simultaneous=1))
    assert mixed.counts.max_simultaneous == 1
    assert len(mixed.utterances) == 1
    assert not mixed.utterances[0].truncated


def test_audio_is_valid_and_covers_all_utterances() -> None:
    clips = [_clip(s, 2.0) for s in "ABC"]
    mixed = _mixer(5).mix(clips, SpeakerCounts(total=3, max_simultaneous=3))
    assert mixed.audio.samples.dtype == np.float32
    assert float(np.abs(mixed.audio.samples).max()) <= 1.0
    latest_end = max(u.end_s for u in mixed.utterances)
    assert mixed.audio.duration_s >= latest_end - 1e-6


def test_overlap_actually_happens_with_high_target() -> None:
    # With four 2s clips and a target of 3, a seeded run should overlap someone.
    clips = [_clip(s, 2.0) for s in "ABCD"]
    mixed = _mixer(11).mix(clips, SpeakerCounts(total=4, max_simultaneous=3))
    assert mixed.counts.max_simultaneous >= 2
