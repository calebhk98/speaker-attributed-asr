"""Tests for the AudioBuffer value object."""

from __future__ import annotations

import numpy as np
import pytest

from satasr.core.audio import SAMPLE_RATE, AudioBuffer


def test_duration_matches_sample_count() -> None:
    buffer = AudioBuffer(np.zeros(SAMPLE_RATE, dtype=np.float32))
    assert buffer.duration_s == pytest.approx(1.0)
    assert buffer.num_samples == SAMPLE_RATE


def test_silence_has_requested_length() -> None:
    buffer = AudioBuffer.silence(2.0)
    assert buffer.duration_s == pytest.approx(2.0)
    assert not buffer.samples.any()


def test_rejects_non_mono_audio() -> None:
    with pytest.raises(ValueError, match="mono"):
        AudioBuffer(np.zeros((2, 10), dtype=np.float32))


def test_rejects_wrong_dtype() -> None:
    with pytest.raises(ValueError, match="float32"):
        AudioBuffer(np.zeros(10, dtype=np.float64))


def test_rejects_bad_sample_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        AudioBuffer(np.zeros(10, dtype=np.float32), sample_rate=0)
