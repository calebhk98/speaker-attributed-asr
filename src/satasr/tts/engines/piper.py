"""Piper TTS engine — CPU-fast, non-autoregressive VITS-derived synthesis.

Piper (rhasspy/piper) is a flow-based, non-autoregressive text-to-speech
engine distributed as small per-voice ONNX models. It has no voice-cloning
capability of its own: each voice is a separate preset model identified by
name (e.g. "en_US-lessac-medium"), so this engine is preset-only and honors
``VoiceReference.preset`` as that voice name (design §4.4). Being
non-autoregressive and CPU-cheap, it is a good fit for bulk CPU generation
(design §3).

The ``piper-tts`` package and its per-voice ONNX weights are a heavy optional
dependency not installed in this environment, so the import is kept strictly
inside :meth:`PiperTTSEngine._render` — module import (and therefore registry
discovery) never touches it.
"""

from __future__ import annotations

import numpy as np

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

#: Fallback preset used only if a caller passes an empty string (guard
#: clause; BaseTTSEngine already rejects ``preset is None`` before we get
#: here). Names one of Piper's standard shipped voice models.
_DEFAULT_VOICE = "en_US-lessac-medium"
#: 16-bit PCM full-scale divisor used to convert Piper's raw int16 samples
#: into the project's float32 [-1, 1] convention.
_INT16_FULL_SCALE = 32768.0


@TTS_ENGINES.register("piper")
class PiperTTSEngine(BaseTTSEngine):
    """Piper: preset-voice, non-autoregressive ONNX TTS."""

    name = "piper"
    supports_cloning = False

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Synthesize ``text`` with Piper's preset ``voice.preset`` voice.

        Loads the named Piper voice model, synthesizes raw 16-bit PCM at
        that voice's native sample rate (varies per voice, e.g. 16 kHz or
        22.05 kHz), and resamples down to the project's 16 kHz mono float32
        convention.
        """
        from piper import PiperVoice  # type: ignore

        model_name = voice.preset or _DEFAULT_VOICE
        piper_voice = PiperVoice.load(model_name)
        pcm_chunks = list(piper_voice.synthesize_stream_raw(text))
        native_rate = int(piper_voice.config.sample_rate)
        return self._to_16k(pcm_chunks, native_rate)

    @staticmethod
    def _to_16k(pcm_chunks: list[bytes], native_rate: int) -> AudioBuffer:
        """Convert Piper's raw int16 PCM chunks to 16 kHz mono float32.

        Uses plain numpy linear interpolation rather than scipy so this
        helper (and its fast test) needs no extra dependency beyond the
        project's existing numpy requirement.
        """
        raw = b"".join(pcm_chunks)
        int16_samples = np.frombuffer(raw, dtype=np.int16)
        samples = int16_samples.astype(np.float32) / _INT16_FULL_SCALE
        resampled = _linear_resample(samples, native_rate, SAMPLE_RATE)
        clipped = np.clip(resampled, -1.0, 1.0).astype(np.float32)
        return AudioBuffer(clipped, SAMPLE_RATE)


def _linear_resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a 1-D float32 array from ``src_rate`` to ``dst_rate`` Hz."""
    if samples.size == 0 or src_rate == dst_rate:
        return samples
    duration_s = samples.shape[0] / src_rate
    dst_count = max(1, round(duration_s * dst_rate))
    src_times = np.arange(samples.shape[0], dtype=np.float64) / src_rate
    dst_times = np.arange(dst_count, dtype=np.float64) / dst_rate
    interpolated: np.ndarray = np.interp(dst_times, src_times, samples)
    return interpolated.astype(np.float32)
