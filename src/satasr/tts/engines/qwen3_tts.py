"""Qwen3-TTS (Alibaba) TTS engine — autoregressive codec-token synthesis.

Qwen3-TTS predicts speech codec tokens autoregressively, conditioned on text
and a short reference clip. Unlike preset engines (e.g. Bark, Kokoro), its
"Base" checkpoint performs 3-second voice cloning: given ``ref_audio`` (and
ideally its transcript ``ref_text``), it reproduces that speaker's voice for
new text (design §4.4). This engine is therefore cloning-only — callers must
set ``VoiceReference.reference_audio``.

Assumptions, since the ``qwen-tts`` package and model weights are not
installed/downloadable in this environment (verify before relying on them):
  * PyPI package ``qwen-tts`` exposes ``Qwen3TTSModel.from_pretrained(...)``
    and a ``generate_voice_clone(text=, ref_audio=, ref_text=)`` method that
    returns a ``(waveform, sample_rate)`` tuple (per the QwenLM/Qwen3-TTS
    README and examples).
  * ``ref_audio`` accepts a ``(numpy_array, sample_rate)`` tuple, so our
    already-decoded ``AudioBuffer`` can be passed straight through.
  * The default checkpoint is the smallest cloning-capable Base model,
    ``Qwen/Qwen3-TTS-12Hz-1.7B-Base``; larger/other checkpoints may exist.
  * ``VoiceReference`` has no transcript field, so ``ref_text`` is passed as
    ``None`` and relies on the model's own reference-audio transcription
    fallback — this is unverified without real weights.
  * License: Qwen/Tongyi terms (reported as Apache 2.0 for the open weights)
    — verify before production use, per the issue.

The heavy ``qwen_tts``/``torch`` import is kept strictly inside
:meth:`Qwen3TTSEngine._render` so module import (and registry discovery)
never touches it.
"""

from __future__ import annotations

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

#: Smallest published checkpoint that supports voice cloning (see module docstring).
_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


@TTS_ENGINES.register("qwen3_tts")
class Qwen3TTSEngine(BaseTTSEngine):
    """Alibaba Qwen3-TTS: cloning-only, autoregressive codec-token TTS."""

    name = "qwen3_tts"
    supports_cloning = True

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Clone ``voice.reference_audio`` and speak ``text`` with it.

        Returns 16 kHz mono float32 audio, resampled/downmixed from whatever
        rate the model natively produces.
        """
        from qwen_tts import Qwen3TTSModel  # type: ignore

        model = Qwen3TTSModel.from_pretrained(_MODEL_ID)
        ref_audio = voice.reference_audio
        assert ref_audio is not None  # enforced by BaseTTSEngine._check_voice

        waveform, native_rate = model.generate_voice_clone(
            text=text,
            ref_audio=(ref_audio.samples, ref_audio.sample_rate),
            ref_text=None,  # no transcript in VoiceReference; see module docstring
        )
        return self._to_16k(waveform, native_rate)

    @staticmethod
    def _to_16k(raw: object, native_rate: int) -> AudioBuffer:
        """Downmix/resample the model's raw waveform to 16 kHz mono.

        Uses plain numpy linear interpolation rather than scipy: numpy is the
        project's only hard runtime dependency (see pyproject.toml), and this
        keeps the resampling helper — exercised by the fast contract test —
        free of any optional/heavy dependency.
        """
        import numpy as np

        samples = np.asarray(raw, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=-1).astype(np.float32)
        if native_rate == SAMPLE_RATE or samples.shape[0] == 0:
            clipped = np.clip(samples, -1.0, 1.0).astype(np.float32)
            return AudioBuffer(clipped, SAMPLE_RATE)

        duration_s = samples.shape[0] / native_rate
        out_samples = round(duration_s * SAMPLE_RATE)
        src_times = np.arange(samples.shape[0], dtype=np.float64) / native_rate
        dst_times = np.arange(out_samples, dtype=np.float64) / SAMPLE_RATE
        resampled = np.interp(dst_times, src_times, samples)
        clipped = np.clip(resampled, -1.0, 1.0).astype(np.float32)
        return AudioBuffer(clipped, SAMPLE_RATE)
