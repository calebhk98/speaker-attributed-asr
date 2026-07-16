"""Tests for ReverbAugmenter (synthetic RIR) and the file-based RIR stub."""

from __future__ import annotations

import numpy as np
import pytest

from satasr.augment.reverb import ReverbAugmenter, RirFileReverbAugmenter
from satasr.core.audio import SAMPLE_RATE, AudioBuffer


def _tone(amplitude: float = 0.4, duration_s: float = 0.2) -> AudioBuffer:
    times = np.arange(round(SAMPLE_RATE * duration_s), dtype=np.float32) / SAMPLE_RATE
    wave = amplitude * np.sin(2.0 * np.pi * 220.0 * times)
    return AudioBuffer(wave.astype(np.float32), SAMPLE_RATE)


def test_preserves_length() -> None:
    audio = _tone()
    result = ReverbAugmenter().apply(audio)
    assert result.num_samples == audio.num_samples


def test_changes_the_signal() -> None:
    audio = _tone()
    result = ReverbAugmenter().apply(audio)
    assert not np.array_equal(result.samples, audio.samples)


def test_zero_wet_level_is_approximately_identity() -> None:
    audio = _tone()
    result = ReverbAugmenter(wet_level=0.0).apply(audio)
    assert result.samples == pytest.approx(audio.samples, abs=1e-6)


def test_longer_decay_changes_the_tail_more() -> None:
    audio = _tone(duration_s=0.5)
    short = ReverbAugmenter(decay_s=0.05).apply(audio)
    long = ReverbAugmenter(decay_s=0.4).apply(audio)
    assert not np.array_equal(short.samples, long.samples)


def test_rejects_non_positive_decay() -> None:
    with pytest.raises(ValueError, match="decay_s"):
        ReverbAugmenter(decay_s=0.0)


def test_rejects_out_of_range_wet_level() -> None:
    with pytest.raises(ValueError, match="wet_level"):
        ReverbAugmenter(wet_level=1.5)


def test_output_is_float32_mono_in_range() -> None:
    audio = _tone()
    result = ReverbAugmenter().apply(audio)
    assert result.samples.dtype == np.float32
    assert result.samples.ndim == 1
    assert result.samples.max() <= 1.0
    assert result.samples.min() >= -1.0


def test_rir_file_stub_raises_not_implemented() -> None:
    augmenter = RirFileReverbAugmenter()
    with pytest.raises(NotImplementedError):
        augmenter.apply(_tone())
