"""Tests for GainAugmenter: RMS moves the expected direction and stays in range."""

from __future__ import annotations

import numpy as np
import pytest

from satasr.augment.gain import GainAugmenter
from satasr.core.audio import SAMPLE_RATE, AudioBuffer


def _tone(amplitude: float = 0.2, duration_s: float = 0.1) -> AudioBuffer:
    times = np.arange(round(SAMPLE_RATE * duration_s), dtype=np.float32) / SAMPLE_RATE
    wave = amplitude * np.sin(2.0 * np.pi * 220.0 * times)
    return AudioBuffer(wave.astype(np.float32), SAMPLE_RATE)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def test_positive_gain_raises_rms() -> None:
    audio = _tone()
    louder = GainAugmenter(gain_db=6.0).apply(audio)
    assert _rms(louder.samples) > _rms(audio.samples)


def test_negative_gain_lowers_rms() -> None:
    audio = _tone()
    quieter = GainAugmenter(gain_db=-6.0).apply(audio)
    assert _rms(quieter.samples) < _rms(audio.samples)


def test_large_gain_clips_to_valid_range() -> None:
    audio = _tone(amplitude=0.9)
    loud = GainAugmenter(gain_db=24.0).apply(audio)
    assert loud.samples.max() <= 1.0
    assert loud.samples.min() >= -1.0


def test_output_is_float32_mono() -> None:
    audio = _tone()
    result = GainAugmenter(gain_db=3.0).apply(audio)
    assert result.samples.dtype == np.float32
    assert result.samples.ndim == 1
    assert result.sample_rate == audio.sample_rate


def test_zero_db_is_approximately_identity() -> None:
    audio = _tone()
    result = GainAugmenter(gain_db=0.0).apply(audio)
    assert result.samples == pytest.approx(audio.samples, abs=1e-6)
