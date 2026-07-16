"""Minimal WAV reading shared by all real-corpus loaders (design §4.8).

Real corpora ship plain PCM WAV files. The project's only runtime dependency
is numpy (see pyproject.toml), so this reads audio with the standard-library
``wave`` module rather than pulling in soundfile/librosa — good enough for the
16-bit PCM that LibriSpeechMix, LibriCSS and AMI exports commonly use.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from satasr.core.audio import AudioBuffer

_PCM16_FULL_SCALE = 32768.0
_PCM16_WIDTH_BYTES = 2


def load_wav_mono(path: Path) -> AudioBuffer:
    """Read a PCM WAV file, downmixing to mono float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as handle:
        return _read(handle)


def _read(handle: wave.Wave_read) -> AudioBuffer:
    _require_16_bit(handle.getsampwidth())
    channels = handle.getnchannels()
    raw = handle.readframes(handle.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / _PCM16_FULL_SCALE
    mono = _downmix(samples, channels)
    return AudioBuffer(mono, handle.getframerate())


def _downmix(samples: NDArray[np.float32], channels: int) -> NDArray[np.float32]:
    if channels == 1:
        return samples
    averaged: NDArray[np.float32] = (
        samples.reshape(-1, channels).mean(axis=1).astype(np.float32)
    )
    return averaged


def _require_16_bit(sample_width_bytes: int) -> None:
    if sample_width_bytes != _PCM16_WIDTH_BYTES:
        raise ValueError(
            f"expected 16-bit PCM WAV, got {sample_width_bytes * 8}-bit samples"
        )
