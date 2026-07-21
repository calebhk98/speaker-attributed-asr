"""Dia2 (Nari Labs) TTS engine — streaming dialogue codec-token synthesis.

Dia2 is Nari Labs' successor to Dia: a streaming, dialogue-native text-to-speech
model published as ``nari-labs/Dia2-1B`` / ``nari-labs/Dia2-2B`` on the Hugging
Face Hub (Apache 2.0). Like Dia it renders a whole conversation from a transcript
whose turns are prefixed with speaker tags such as ``"[S1]"`` / ``"[S2]"``, and
it decodes audio through the Kyutai Mimi codec (~24 kHz), not Dia's 44.1 kHz DAC.

This pipeline calls TTS engines one sentence and one speaker at a time (see
``BaseTTSEngine.synthesize``), so this engine renders a single-turn utterance
per call, exactly like the ``dia`` engine. Dia2 can also condition on prefix
speaker audio for cloning, but — mirroring the ``dia`` engine and issue #16 —
it is registered preset-only (``supports_cloning = False``):
``VoiceReference.preset`` supplies the speaker tag and no reference audio is
used. Both ``dia`` (v1) and this ``dia2`` engine stay registered independently.

ASSUMPTIONS (unverified here — no GPU/weights in this environment; from the
``nari-labs/Dia2-2B`` model card's "Programmatic Usage"):
- The optional dependency installs as the ``dia2`` package, exposing
  ``dia2.Dia2`` plus ``GenerationConfig`` / ``SamplingConfig``.
- ``Dia2.from_repo(checkpoint, device=, dtype=)`` loads the model and
  ``model.generate(text, config=...)`` returns a ``GenerationResult`` whose
  ``.waveform`` holds a 1-D float waveform tensor.
- The smaller 1B checkpoint is used as the default (cheaper for bulk
  generation); the 2B checkpoint exists for higher quality.

The ``dia2`` package and its torch weights are a heavy, GPU-hungry optional
dependency not installed here, so every import lives strictly inside
:meth:`Dia2TTSEngine._render` — module import (and registry discovery) never
touches it.
"""

from __future__ import annotations

import numpy as np

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

#: Dia2's native codec sample rate (Kyutai Mimi), per the model card.
_DIA2_NATIVE_SAMPLE_RATE = 24_000
#: Published checkpoint used for inference (smaller of the two; see docstring).
_MODEL_ID = "nari-labs/Dia2-1B"
#: Speaker tag used when a caller supplies no distinguishing preset content.
_DEFAULT_SPEAKER_TAG = "S1"


@TTS_ENGINES.register("dia2")
class Dia2TTSEngine(BaseTTSEngine):
    """Nari Labs Dia2: preset-tagged, streaming dialogue codec-token TTS."""

    name = "dia2"
    supports_cloning = False
    #: Dia2 checkpoint this engine loads (see ``_MODEL_ID`` above).
    model_ids: tuple[str, ...] = (_MODEL_ID,)

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Synthesize one speaker-tagged utterance with Dia2.

        Returns 16 kHz mono float32 audio, resampled down from Dia2's native
        24 kHz Mimi codec output.
        """
        from dia2 import Dia2, GenerationConfig  # type: ignore

        model = Dia2.from_repo(_MODEL_ID, device="cuda", dtype="bfloat16")
        prompt = self._build_prompt(text, voice)
        result = model.generate(prompt, config=GenerationConfig(), verbose=False)
        return self._to_16k(result.waveform)

    @staticmethod
    def _build_prompt(text: str, voice: VoiceReference) -> str:
        """Prefix ``text`` with the speaker tag Dia2 expects for one turn.

        ``voice.preset`` names the tag (e.g. ``"S1"``); bare names are wrapped
        in brackets, and an already-bracketed preset is used as-is.
        """
        tag = voice.preset or _DEFAULT_SPEAKER_TAG
        bracketed = tag if tag.startswith("[") else f"[{tag}]"
        return f"{bracketed} {text}"

    @staticmethod
    def _to_16k(raw: object) -> AudioBuffer:
        """Downmix/resample Dia2's raw waveform to the project's 16 kHz mono.

        Uses plain numpy linear interpolation rather than scipy so this helper
        (and its fast test) needs no extra dependency beyond numpy.
        """
        samples = np.asarray(raw, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=-1).astype(np.float32)
        resampled = _linear_resample(samples, _DIA2_NATIVE_SAMPLE_RATE, SAMPLE_RATE)
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
