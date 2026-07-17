"""Tests for the dependency-free ProportionalAligner."""

from __future__ import annotations

import numpy as np
import pytest

from satasr.alignment.registry import ALIGNERS
from satasr.core.audio import SAMPLE_RATE, AudioBuffer

TEXT = "the quick brown fox jumps"


def _audio(duration_s: float = 2.0) -> AudioBuffer:
    samples = np.zeros(round(duration_s * SAMPLE_RATE), dtype=np.float32)
    return AudioBuffer(samples, SAMPLE_RATE)


def test_alignment_is_deterministic() -> None:
    aligner = ALIGNERS.create("proportional")
    first = aligner.align(_audio(), TEXT)
    second = aligner.align(_audio(), TEXT)
    assert first == second


def test_word_count_matches_token_count() -> None:
    aligner = ALIGNERS.create("proportional")
    words = aligner.align(_audio(), TEXT)
    assert [w.text for w in words] == TEXT.split()


def test_covers_roughly_the_full_duration() -> None:
    aligner = ALIGNERS.create("proportional")
    audio = _audio(3.0)
    words = aligner.align(audio, TEXT)
    assert words[0].start_s == pytest.approx(0.0)
    assert words[-1].end_s == pytest.approx(audio.duration_s)


def test_words_are_monotonic_and_non_overlapping() -> None:
    aligner = ALIGNERS.create("proportional")
    words = aligner.align(_audio(), TEXT)
    for earlier, later in zip(words, words[1:], strict=False):
        assert earlier.end_s <= later.start_s
        assert earlier.start_s < earlier.end_s


def test_longer_words_get_longer_spans() -> None:
    aligner = ALIGNERS.create("proportional")
    words = aligner.align(_audio(), "a bb ccc")
    durations = [w.end_s - w.start_s for w in words]
    assert durations[0] < durations[1] < durations[2]


def test_empty_text_is_rejected() -> None:
    aligner = ALIGNERS.create("proportional")
    with pytest.raises(ValueError, match="empty"):
        aligner.align(_audio(), "   ")


def test_proportional_is_registered() -> None:
    assert "proportional" in ALIGNERS.available()
