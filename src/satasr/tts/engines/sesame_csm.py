"""Sesame CSM (Conversational Speech Model) TTS engine — context-conditioned
voice cloning.

CSM (Sesame AI Labs, Apache-2.0 — verify license terms before production use)
is a dialogue-native model: a Llama-backbone generator over a Mimi neural
audio codec that conditions on prior conversation ``Segment``s (speaker,
text, audio) rather than a single embedding. There is no built-in voice
bank, so cloning is the only mode: a voice is "continued" by handing the
reference clip to the model as one-segment context and asking it to
generate the next turn in the same voice (design §4.4).

The ``csm`` package (Sesame AI Labs' reference implementation), its Llama
backbone weights, and the Mimi codec are a heavy optional dependency not
installed in this environment and require a GPU, so every third-party
import stays strictly inside the render path — module import (and
therefore registry discovery) never touches them.
"""

from __future__ import annotations

import numpy as np

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

#: CSM has no reference transcript for the cloned clip, so the context
#: segment is given this placeholder text; only its audio/speaker matter for
#: voice conditioning.
_CONTEXT_PLACEHOLDER_TEXT = "..."
#: Single synthetic speaker id used for both the context segment and the
#: generated turn, since VoiceReference tracks one voice at a time.
_SPEAKER_ID = 0
#: Generous cap so long sentences are not truncated mid-word.
_MAX_AUDIO_LENGTH_MS = 90_000.0


@TTS_ENGINES.register("sesame_csm")
class SesameCsmTTSEngine(BaseTTSEngine):
    """Sesame CSM: context-conditioned dialogue TTS with voice cloning."""

    name = "sesame_csm"
    supports_cloning = True

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Clone ``voice.reference_audio`` and speak ``text`` with CSM.

        ``voice.reference_audio`` is guaranteed non-None here —
        ``BaseTTSEngine._check_voice`` enforces the cloning contract before
        ``_render`` is ever called. Returns 16 kHz mono float32 audio,
        resampled down from CSM's native 24 kHz Mimi-codec output.
        """
        from csm.generator import load_csm_1b  # type: ignore
        from torch import cuda  # type: ignore

        device = "cuda" if cuda.is_available() else "cpu"
        generator = load_csm_1b(device)
        context = [self._context_segment(voice, generator.sample_rate)]
        wav = generator.generate(
            text=text,
            speaker=_SPEAKER_ID,
            context=context,
            max_audio_length_ms=_MAX_AUDIO_LENGTH_MS,
        )
        return self._to_16k(wav, generator.sample_rate)

    @staticmethod
    def _context_segment(voice: VoiceReference, native_rate: int) -> object:
        """Wrap the reference clip as the one-turn context CSM clones from."""
        # torch/csm.generator were already imported (with type:ignore) in
        # _render, in the same file — mypy only flags a missing module's
        # first import per file, so re-importing here needs no ignore.
        import torch
        from csm.generator import Segment

        assert voice.reference_audio is not None  # enforced by base class
        samples = SesameCsmTTSEngine._resample(
            voice.reference_audio.samples,
            voice.reference_audio.sample_rate,
            native_rate,
        )
        return Segment(
            speaker=_SPEAKER_ID,
            text=_CONTEXT_PLACEHOLDER_TEXT,
            audio=torch.from_numpy(samples),
        )

    @staticmethod
    def _resample(samples: object, from_rate: int, to_rate: int) -> np.ndarray:
        """Resample raw float32 samples between two sample rates.

        Uses plain numpy linear interpolation rather than scipy so this
        helper (and its fast tests) need no extra dependency beyond the
        project's existing numpy requirement.
        """
        array = np.asarray(samples, dtype=np.float32)
        if from_rate == to_rate or array.size == 0:
            return array
        duration_s = array.shape[0] / from_rate
        dst_count = max(1, round(duration_s * to_rate))
        src_times = np.arange(array.shape[0], dtype=np.float64) / from_rate
        dst_times = np.arange(dst_count, dtype=np.float64) / to_rate
        interpolated: np.ndarray = np.interp(dst_times, src_times, array)
        return interpolated.astype(np.float32)

    @staticmethod
    def _to_16k(wav: object, native_rate: int) -> AudioBuffer:
        """Downmix/resample CSM's raw waveform to the project's 16 kHz mono."""
        samples = np.asarray(wav, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=-1).astype(np.float32)
        resampled = SesameCsmTTSEngine._resample(samples, native_rate, SAMPLE_RATE)
        clipped = np.clip(resampled, -1.0, 1.0).astype(np.float32)
        return AudioBuffer(clipped, SAMPLE_RATE)
