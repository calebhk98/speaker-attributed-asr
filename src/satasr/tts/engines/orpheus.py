"""Orpheus (Canopy Labs) TTS engine — Llama-backbone codec-token synthesis.

Orpheus is an autoregressive, Llama-3B-backbone speech-language model from
Canopy Labs that predicts SNAC audio-codec tokens from text. Per issue #6 this
engine always clones the caller-supplied ``VoiceReference.reference_audio``
clip rather than picking one of Orpheus's named preset voices ("tara",
"leah", ...), so ``supports_cloning`` is ``True``.

Upstream API note: the public ``orpheus_tts.OrpheusModel.generate_speech``
entry point (see ``canopyai/Orpheus-TTS``) takes a plain-text ``prompt`` and a
preset ``voice`` name; Canopy Labs' own README says zero-shot cloning from raw
reference audio "hasn't been explicitly trained on" and is not a stabilized,
documented kwarg (there is an open upstream issue asking for exactly this).
:meth:`_build_clone_prompt` below is therefore a best-effort mapping of our
``reference_audio`` onto that still-evolving convention, not a verified call —
see the accompanying issue comment for the full list of assumptions.

The ``orpheus-speech`` package (imported as ``orpheus_tts``) and its vLLM/
Llama-3B weights are a heavy, GPU-hungry optional dependency that is not
installed in this environment, so every import lives strictly inside
:meth:`OrpheusTTSEngine._render` — module import (and therefore registry
discovery) never touches it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

#: Orpheus/SNAC's native codec sample rate, per the model card and the
#: reference streaming examples (``wave`` writer configured with 24000 Hz).
_ORPHEUS_NATIVE_SAMPLE_RATE = 24_000
#: Published fine-tuned checkpoint used for inference (model card default).
_MODEL_NAME = "canopylabs/orpheus-tts-0.1-finetune-prod"


def _resample_linear(
    samples: NDArray[np.float32], src_rate: int, dst_rate: int
) -> NDArray[np.float32]:
    """Resample ``samples`` from ``src_rate`` to ``dst_rate`` via interpolation.

    Plain-numpy linear interpolation (no ``scipy`` dependency); adequate here
    since the result is immediately clipped/packaged, not used for spectral
    analysis.
    """
    if src_rate == dst_rate or samples.size == 0:
        return samples
    duration_s = samples.shape[0] / src_rate
    dst_count = round(duration_s * dst_rate)
    src_times = np.arange(samples.shape[0], dtype=np.float64) / src_rate
    dst_times = np.arange(dst_count, dtype=np.float64) / dst_rate
    return np.interp(dst_times, src_times, samples).astype(np.float32)


@TTS_ENGINES.register("orpheus")
class OrpheusTTSEngine(BaseTTSEngine):
    """Canopy Labs Orpheus: cloned-voice, autoregressive codec-token TTS."""

    name = "orpheus"
    supports_cloning = True
    #: Orpheus checkpoint this engine loads (see ``_MODEL_NAME`` above).
    model_ids: tuple[str, ...] = (_MODEL_NAME,)

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Clone ``voice.reference_audio`` and speak ``text`` with Orpheus.

        Returns 16 kHz mono float32 audio, resampled down from Orpheus's
        native 24 kHz SNAC-decoder output.
        """
        from orpheus_tts import OrpheusModel  # type: ignore

        model = OrpheusModel(model_name=_MODEL_NAME)
        prompt = self._build_clone_prompt(text, voice)
        chunks = model.generate_speech(prompt=prompt, voice=None)
        return self._to_16k(chunks)

    @staticmethod
    def _build_clone_prompt(text: str, voice: VoiceReference) -> object:
        """Condition generation on the reference clip (zero-shot cloning).

        ASSUMPTION (unverified — see module docstring): this delegates to a
        package-level helper that turns raw reference audio into the
        text/audio "prompt" turn Orpheus's zero-shot cloning convention
        expects, then appends ``text`` as the turn to synthesize.
        """
        from orpheus_tts.audio import encode_reference_prompt  # type: ignore

        reference = voice.reference_audio
        assert reference is not None  # enforced by BaseTTSEngine._check_voice
        return encode_reference_prompt(
            reference_samples=reference.samples,
            reference_sample_rate=reference.sample_rate,
            target_text=text,
        )

    @staticmethod
    def _to_16k(chunks: object) -> AudioBuffer:
        """Concatenate streamed PCM16 chunks and resample to 16 kHz mono.

        Uses plain-numpy linear resampling rather than ``scipy`` so this
        helper — exercised directly by the fast contract test — needs no
        dependency beyond the project's existing ``numpy`` requirement.
        """
        raw = b"".join(chunks)  # type: ignore[arg-type]
        pcm16 = np.frombuffer(raw, dtype=np.int16)
        samples = pcm16.astype(np.float32) / 32768.0
        resampled = _resample_linear(samples, _ORPHEUS_NATIVE_SAMPLE_RATE, SAMPLE_RATE)
        clipped = np.clip(resampled, -1.0, 1.0).astype(np.float32)
        return AudioBuffer(clipped, SAMPLE_RATE)
