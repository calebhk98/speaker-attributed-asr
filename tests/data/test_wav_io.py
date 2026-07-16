"""Fast tests for the stdlib-only WAV reader (design §4.8)."""

from __future__ import annotations

import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from satasr.data.wav_io import load_wav_mono


def test_reads_mono_pcm16_wav_into_float32_buffer(
    tmp_path: Path, write_wav: Callable[..., None]
) -> None:
    path = tmp_path / "clip.wav"
    write_wav(path, duration_s=0.5)

    audio = load_wav_mono(path)

    assert audio.sample_rate == 16_000
    assert audio.samples.dtype == np.float32
    assert audio.samples.ndim == 1
    assert abs(audio.duration_s - 0.5) < 0.01


def test_downmixes_stereo_to_mono(
    tmp_path: Path, write_wav: Callable[..., None]
) -> None:
    path = tmp_path / "stereo.wav"
    write_wav(path, duration_s=0.2, channels=2)

    audio = load_wav_mono(path)

    assert audio.samples.ndim == 1
    assert abs(audio.duration_s - 0.2) < 0.01


def test_rejects_non_16_bit_pcm(tmp_path: Path) -> None:
    path = tmp_path / "bad.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(16_000)
        handle.writeframes(bytes([128] * 100))

    with pytest.raises(ValueError, match="16-bit"):
        load_wav_mono(path)
