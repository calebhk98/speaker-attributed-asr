"""Reference engine: a deterministic sine-tone "voice".

This is the template real engines follow, and a dependency-free stand-in that
lets the whole pipeline (mixing, alignment, training glue) be exercised in tests
and CI without downloading multi-gigabyte weights. It is NOT a real TTS model.
"""

from __future__ import annotations

import numpy as np

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

_SECONDS_PER_CHAR = 0.06  # rough spoken pace, enough to give clips real length
_MIN_DURATION_S = 0.3
_AMPLITUDE = 0.3
_BASE_HZ = 110.0  # preset name shifts the pitch so voices are distinguishable


@TTS_ENGINES.register("sine")
class SineTTSEngine(BaseTTSEngine):
    """A preset-voiced sine generator — fast, deterministic, offline."""

    name = "sine"
    supports_cloning = False

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        duration_s = max(_MIN_DURATION_S, len(text) * _SECONDS_PER_CHAR)
        num_samples = round(duration_s * SAMPLE_RATE)
        times = np.arange(num_samples, dtype=np.float32) / SAMPLE_RATE
        frequency = self._preset_frequency(voice.preset)
        wave = _AMPLITUDE * np.sin(2.0 * np.pi * frequency * times)
        return AudioBuffer(wave.astype(np.float32), SAMPLE_RATE)

    @staticmethod
    def _preset_frequency(preset: str | None) -> float:
        """Map a preset name to a stable pitch so each voice sounds distinct."""
        offset = 0 if preset is None else sum(ord(ch) for ch in preset) % 200
        return _BASE_HZ + offset
