"""Bark (Suno) TTS engine — autoregressive codec-token synthesis.

Bark generates speech by autoregressively predicting audio codec tokens
conditioned on text and a "history prompt" (a bundled preset embedding for one
of Bark's built-in speakers). It has no voice-cloning API of its own, so this
engine is preset-only: callers select a voice via ``VoiceReference.preset``
naming one of Bark's shipped speaker prompts (e.g. ``"v2/en_speaker_6"``).

The ``bark`` package and its torch weights are a heavy, GPU-hungry optional
dependency that is not installed in this environment, so the import is kept
strictly inside :meth:`BarkTTSEngine._render` — module import (and therefore
registry discovery) never touches it.
"""

from __future__ import annotations

import numpy as np

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

#: Bark's native codec sample rate, per the model card / generate_audio docs.
_BARK_NATIVE_SAMPLE_RATE = 24_000
#: Fallback preset used only if a caller passes an empty string (guard clause;
#: BaseTTSEngine already rejects ``preset is None`` before we get here).
_DEFAULT_HISTORY_PROMPT = "v2/en_speaker_6"


@TTS_ENGINES.register("bark")
class BarkTTSEngine(BaseTTSEngine):
    """Suno Bark: preset-voice, autoregressive codec-token TTS."""

    name = "bark"
    supports_cloning = False

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Synthesize ``text`` with Bark's preset ``voice.preset`` history prompt.

        Returns 16 kHz mono float32 audio, resampling down from Bark's native
        24 kHz codec output.
        """
        from bark import generate_audio, preload_models  # type: ignore

        preload_models()
        history_prompt = voice.preset or _DEFAULT_HISTORY_PROMPT
        raw = generate_audio(text, history_prompt=history_prompt)
        return self._to_16k(raw)

    @staticmethod
    def _to_16k(raw: object) -> AudioBuffer:
        """Downmix/resample Bark's raw waveform to the project's 16 kHz mono.

        Uses plain numpy linear interpolation rather than scipy so this helper
        (and its fast test) needs no extra dependency beyond the project's
        existing numpy requirement.
        """
        samples = np.asarray(raw, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=-1).astype(np.float32)
        resampled = _linear_resample(samples, _BARK_NATIVE_SAMPLE_RATE, SAMPLE_RATE)
        clipped = np.clip(resampled, -1.0, 1.0).astype(np.float32)
        return AudioBuffer(clipped, SAMPLE_RATE)


def _linear_resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a 1-D float32 array from ``src_rate`` to ``dst_rate`` Hz."""
    if samples.size == 0:
        return samples
    duration_s = samples.shape[0] / src_rate
    dst_count = max(1, round(duration_s * dst_rate))
    src_times = np.arange(samples.shape[0], dtype=np.float64) / src_rate
    dst_times = np.arange(dst_count, dtype=np.float64) / dst_rate
    interpolated: np.ndarray = np.interp(dst_times, src_times, samples)
    return interpolated.astype(np.float32)
