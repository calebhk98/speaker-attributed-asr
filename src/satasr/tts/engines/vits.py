"""VITS TTS engine — original academic flow-based end-to-end multi-speaker TTS.

VITS (Kim et al., 2021, "Conditional Variational Autoencoder with Adversarial
Learning for End-to-End Text-to-Speech") is a flow-based, non-autoregressive
end-to-end model trained on a fixed set of speakers baked into the checkpoint.
It has no reference-audio cloning API — voices are selected by a multi-speaker
embedding id — so this engine is preset-only: callers select a voice via
``VoiceReference.preset`` naming one of the checkpoint's built-in speaker ids
(e.g. the VCTK VITS checkpoint's ``"p225"``).

This engine loads the pretrained multi-speaker VITS checkpoint distributed via
the Coqui ``TTS`` package (MIT licensed community wrapper around the academic
VITS architecture; verify license/model terms before production use). The
``TTS`` package and its torch weights are a heavy, GPU-hungry optional
dependency not installed in this environment, so the import is kept strictly
inside :meth:`VitsTTSEngine._render` — module import (and therefore registry
discovery) never touches it.
"""

from __future__ import annotations

import numpy as np

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

#: Pretrained multi-speaker VITS checkpoint (VCTK), per the Coqui TTS model zoo.
_MODEL_NAME = "tts_models/en/vctk/vits"
#: VITS/VCTK's native output sample rate, per the model card.
_VITS_NATIVE_SAMPLE_RATE = 22_050
#: Fallback speaker id used only if a caller passes an empty string (guard
#: clause; BaseTTSEngine already rejects ``preset is None`` before we get here).
_DEFAULT_SPEAKER_ID = "p225"


@TTS_ENGINES.register("vits")
class VitsTTSEngine(BaseTTSEngine):
    """Original VITS: preset multi-speaker, flow-based end-to-end TTS."""

    name = "vits"
    supports_cloning = False

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Synthesize ``text`` with VITS's preset ``voice.preset`` speaker id.

        Returns 16 kHz mono float32 audio, resampled down from VITS/VCTK's
        native 22.05 kHz output.
        """
        from TTS.api import TTS as CoquiTTS  # type: ignore

        model = CoquiTTS(_MODEL_NAME)
        speaker_id = voice.preset or _DEFAULT_SPEAKER_ID
        raw = model.tts(text=text, speaker=speaker_id)
        return self._to_16k(raw)

    @staticmethod
    def _to_16k(raw: object) -> AudioBuffer:
        """Downmix/resample VITS's raw waveform to the project's 16 kHz mono.

        Uses plain numpy linear interpolation rather than scipy so this
        helper (and its fast test) needs no extra dependency beyond the
        project's existing numpy requirement.
        """
        samples = np.asarray(raw, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=-1).astype(np.float32)
        resampled = _linear_resample(samples, _VITS_NATIVE_SAMPLE_RATE, SAMPLE_RATE)
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
