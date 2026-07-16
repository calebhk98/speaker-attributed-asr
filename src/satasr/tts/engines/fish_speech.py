"""Fish Speech (Fish Audio S1/S2) TTS engine — autoregressive codec-token
voice cloning.

Fish Speech is Fish Audio's open TTS stack: a text-to-semantic-token
autoregressive transformer paired with a VQ-GAN codec decoder/vocoder. Voice
cloning is in-context — a short reference clip conditions the semantic model
to imitate the reference speaker's timbre — so this engine has no built-in
presets and is registered with ``supports_cloning = True``, requiring
``VoiceReference.reference_audio`` (design §4.4).

The ``fish-speech`` package, its checkpoints, and CUDA weights are a heavy,
GPU-hungry optional dependency not installed in this environment (no GPU, no
weights downloaded here), so every import of it is kept strictly inside
:meth:`FishSpeechTTSEngine._render` — module import (and therefore registry
discovery) never touches it.
"""

from __future__ import annotations

import numpy as np

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

#: Fish Speech's VQ-GAN decoder emits audio at this native rate (model card /
#: ``fish_speech.utils.spectrogram`` defaults).
_FISH_SPEECH_NATIVE_SAMPLE_RATE = 44_100


@TTS_ENGINES.register("fish_speech")
class FishSpeechTTSEngine(BaseTTSEngine):
    """Fish Audio's Fish Speech (S1/S2): cloning, autoregressive codec-token TTS."""

    name = "fish_speech"
    supports_cloning = True

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Clone ``voice.reference_audio``'s timbre and speak ``text``.

        Uses ``fish_speech.inference_engine.TTSInferenceEngine`` (the packaged
        S1/S2 inference API) to build an in-context voice prompt from the
        reference clip, then generates the target utterance. Output is
        resampled from the model's native 44.1 kHz to the project's 16 kHz
        mono convention.
        """
        from fish_speech.inference_engine import TTSInferenceEngine  # type: ignore

        reference = self._require_reference_audio(voice)
        engine = TTSInferenceEngine.from_pretrained()
        raw = engine.generate(
            text=text,
            reference_audio=reference.samples,
            reference_sample_rate=reference.sample_rate,
        )
        return self._to_16k(raw)

    @staticmethod
    def _require_reference_audio(voice: VoiceReference) -> AudioBuffer:
        """Narrow ``voice.reference_audio`` to non-``None``.

        ``BaseTTSEngine._check_voice`` already enforces this before
        ``_render`` runs; this guard just keeps the type checker and the
        error message local to this engine.
        """
        if voice.reference_audio is None:
            raise ValueError("fish_speech clones voices; reference_audio is required")
        return voice.reference_audio

    @staticmethod
    def _to_16k(raw: object) -> AudioBuffer:
        """Downmix/resample Fish Speech's raw waveform to the project's 16 kHz
        mono convention.

        Uses plain numpy linear interpolation rather than scipy so this helper
        (and its fast test) needs no extra dependency beyond the project's
        existing numpy requirement.
        """
        samples = np.asarray(raw, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=-1).astype(np.float32)
        resampled = _linear_resample(
            samples, _FISH_SPEECH_NATIVE_SAMPLE_RATE, SAMPLE_RATE
        )
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
