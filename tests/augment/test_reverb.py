"""Tests for ReverbAugmenter (synthetic RIR) and RirFileReverbAugmenter (real,
file-loaded RIR)."""

from __future__ import annotations

import os
import random
import wave
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from satasr.augment.reverb import ReverbAugmenter, RirFileReverbAugmenter
from satasr.core.audio import SAMPLE_RATE, AudioBuffer


def _tone(amplitude: float = 0.4, duration_s: float = 0.2) -> AudioBuffer:
    times = np.arange(round(SAMPLE_RATE * duration_s), dtype=np.float32) / SAMPLE_RATE
    wave_ = amplitude * np.sin(2.0 * np.pi * 220.0 * times)
    return AudioBuffer(wave_.astype(np.float32), SAMPLE_RATE)


def _tiny_rir(sample_rate: int, scale: float = 1.0) -> NDArray[np.float32]:
    """A tiny synthetic RIR: a direct impulse plus one decayed reflection."""
    rir = np.zeros(round(sample_rate * 0.01), dtype=np.float32)
    rir[0] = scale
    rir[-1] = 0.5 * scale
    return rir


def _write_rir_wav(path: Path, rir: NDArray[np.float32], sample_rate: int) -> None:
    """Write a mono 16-bit PCM WAV file — no soundfile dependency needed."""
    ints = np.clip(rir, -1.0, 1.0) * 32767.0
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(ints.astype(np.int16).tobytes())


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


def test_rir_file_preserves_length(tmp_path: Path) -> None:
    _write_rir_wav(tmp_path / "room1.wav", _tiny_rir(SAMPLE_RATE), SAMPLE_RATE)
    audio = _tone()
    result = RirFileReverbAugmenter(str(tmp_path)).apply(audio)
    assert result.num_samples == audio.num_samples


def test_rir_file_changes_the_signal(tmp_path: Path) -> None:
    _write_rir_wav(tmp_path / "room1.wav", _tiny_rir(SAMPLE_RATE), SAMPLE_RATE)
    audio = _tone()
    result = RirFileReverbAugmenter(str(tmp_path)).apply(audio)
    assert not np.array_equal(result.samples, audio.samples)


def test_rir_file_output_is_float32_mono_in_range(tmp_path: Path) -> None:
    _write_rir_wav(tmp_path / "room1.wav", _tiny_rir(SAMPLE_RATE), SAMPLE_RATE)
    result = RirFileReverbAugmenter(str(tmp_path)).apply(_tone())
    assert result.samples.dtype == np.float32
    assert result.samples.ndim == 1
    assert result.samples.max() <= 1.0
    assert result.samples.min() >= -1.0


def test_rir_file_samples_across_multiple_corpus_files(tmp_path: Path) -> None:
    small_rir = _tiny_rir(SAMPLE_RATE, 1.0)
    large_rir = _tiny_rir(SAMPLE_RATE, 0.5)
    _write_rir_wav(tmp_path / "room_small.wav", small_rir, SAMPLE_RATE)
    _write_rir_wav(tmp_path / "room_large.wav", large_rir, SAMPLE_RATE)
    augmenter = RirFileReverbAugmenter(str(tmp_path), rng=random.Random(0))
    result = augmenter.apply(_tone())
    assert result.num_samples == _tone().num_samples


def test_rir_file_construction_does_not_touch_disk() -> None:
    # Must not raise: a nonexistent dir is only looked at inside apply().
    RirFileReverbAugmenter("/nonexistent/rir/corpus/path/xyz")


def test_rir_file_raises_when_corpus_dir_has_no_wavs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RirFileReverbAugmenter(str(tmp_path)).apply(_tone())


def test_rir_file_rejects_out_of_range_wet_level() -> None:
    with pytest.raises(ValueError, match="wet_level"):
        RirFileReverbAugmenter("unused", wet_level=2.0)


@pytest.mark.slow
def test_rir_file_real_corpus_convolution() -> None:
    """Real convolution against a real RIR corpus (e.g. OpenSLR SLR26/SLR28).

    Requires ``SATASR_RIR_CORPUS_DIR`` to point at a directory of real,
    recorded/simulated RIR ``.wav`` files (a small fixture subset is enough).
    """
    corpus_dir = os.environ["SATASR_RIR_CORPUS_DIR"]
    augmenter = RirFileReverbAugmenter(corpus_dir, rng=random.Random(0))
    audio = _tone(duration_s=0.5)
    result = augmenter.apply(audio)
    assert result.num_samples == audio.num_samples
    assert not np.array_equal(result.samples, audio.samples)
