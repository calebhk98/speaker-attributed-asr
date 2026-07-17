"""Tests for the domain data model."""

from __future__ import annotations

import numpy as np

from satasr.core.audio import AudioBuffer
from satasr.core.models import (
    MixedClip,
    PlacedUtterance,
    SpeakerClip,
    SpeakerCounts,
    Word,
)


def _clip() -> SpeakerClip:
    audio = AudioBuffer(np.zeros(1000, dtype=np.float32))
    return SpeakerClip(audio=audio, text="hi", speaker_id="A")


def test_speaker_clip_alignment_flag() -> None:
    assert not _clip().is_aligned
    aligned = SpeakerClip(
        audio=_clip().audio,
        text="hi",
        speaker_id="A",
        words=(Word("hi", 0.0, 0.5),),
    )
    assert aligned.is_aligned


def test_with_audio_preserves_labels() -> None:
    original = MixedClip(
        audio=AudioBuffer(np.zeros(10, dtype=np.float32)),
        utterances=(PlacedUtterance("A", 0.0, 1.0, "hi"),),
        counts=SpeakerCounts(total=1, max_simultaneous=1),
    )
    louder = AudioBuffer(np.ones(10, dtype=np.float32))
    updated = original.with_audio(louder)

    assert updated.utterances == original.utterances
    assert updated.counts == original.counts
    assert updated.audio is louder
