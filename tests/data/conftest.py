"""Shared fixtures for the ``tests/data`` fast-test fixtures (design §4.8).

Real corpora ship plain PCM WAV. ``write_wav`` produces tiny ones with the
standard-library ``wave`` module so no audio files are ever committed.
"""

from __future__ import annotations

import math
import struct
import wave
from collections.abc import Callable
from pathlib import Path

import pytest

_AMPLITUDE = 3000  # comfortably within signed 16-bit range


@pytest.fixture
def write_wav() -> Callable[..., None]:
    """Write a short mono (or multi-channel) 16-bit PCM WAV tone."""

    def _write(
        path: Path,
        duration_s: float,
        *,
        tone_hz: float = 200.0,
        sample_rate: int = 16_000,
        channels: int = 1,
    ) -> None:
        num_samples = round(duration_s * sample_rate)
        frame = bytearray()
        for i in range(num_samples):
            value = int(_AMPLITUDE * math.sin(2 * math.pi * tone_hz * i / sample_rate))
            frame += struct.pack("<h", value) * channels
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(bytes(frame))

    return _write
