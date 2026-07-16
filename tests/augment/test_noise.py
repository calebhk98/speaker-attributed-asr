"""Tests for WhiteNoiseAugmenter (SNR accuracy, determinism) and the MUSAN stub."""

from __future__ import annotations

import random

import numpy as np
import pytest

from satasr.augment.noise import MusanNoiseAugmenter, WhiteNoiseAugmenter
from satasr.core.audio import SAMPLE_RATE, AudioBuffer

_SNR_TOLERANCE_DB = 1.0


def _tone(amplitude: float = 0.3, duration_s: float = 1.0) -> AudioBuffer:
    times = np.arange(round(SAMPLE_RATE * duration_s), dtype=np.float32) / SAMPLE_RATE
    wave = amplitude * np.sin(2.0 * np.pi * 220.0 * times)
    return AudioBuffer(wave.astype(np.float32), SAMPLE_RATE)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def test_achieves_approximately_the_requested_snr() -> None:
    audio = _tone()
    target_snr_db = 10.0
    augmenter = WhiteNoiseAugmenter(snr_db=target_snr_db, rng=random.Random(1))

    noisy = augmenter.apply(audio)
    noise_only = noisy.samples - audio.samples
    measured_snr_db = 20.0 * np.log10(_rms(audio.samples) / _rms(noise_only))

    assert measured_snr_db == pytest.approx(target_snr_db, abs=_SNR_TOLERANCE_DB)


def test_deterministic_given_a_seeded_rng() -> None:
    audio = _tone()
    first = WhiteNoiseAugmenter(snr_db=5.0, rng=random.Random(42)).apply(audio)
    second = WhiteNoiseAugmenter(snr_db=5.0, rng=random.Random(42)).apply(audio)
    assert np.array_equal(first.samples, second.samples)


def test_different_seeds_give_different_noise() -> None:
    audio = _tone()
    first = WhiteNoiseAugmenter(snr_db=5.0, rng=random.Random(1)).apply(audio)
    second = WhiteNoiseAugmenter(snr_db=5.0, rng=random.Random(2)).apply(audio)
    assert not np.array_equal(first.samples, second.samples)


def test_output_is_float32_mono_in_range() -> None:
    audio = _tone()
    result = WhiteNoiseAugmenter(snr_db=0.0, rng=random.Random(0)).apply(audio)
    assert result.samples.dtype == np.float32
    assert result.samples.ndim == 1
    assert result.samples.max() <= 1.0
    assert result.samples.min() >= -1.0


def test_musan_stub_raises_not_implemented() -> None:
    augmenter = MusanNoiseAugmenter(snr_db=10.0)
    with pytest.raises(NotImplementedError, match="MUSAN"):
        augmenter.apply(_tone())
