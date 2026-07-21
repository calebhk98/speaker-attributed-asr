"""Chatterbox-Turbo (Resemble AI) TTS engine — faster Chatterbox variant.

Turbo is Resemble AI's distilled/optimised Chatterbox: the same zero-shot,
codec-token voice-cloning model as the base ``chatterbox`` engine, but with a
lighter inference path (and native paralinguistic tags such as ``[laugh]``).
It is loaded through a *different* class in the same package
(``chatterbox.tts_turbo.ChatterboxTurboTTS``), so it lives in its own engine
file rather than reusing the base one — both stay registered and selectable
independently (design §4.2). Like base Chatterbox it clones from an on-disk
reference WAV, so this engine is cloning-only and requires
``VoiceReference.reference_audio`` (design §4.4).

The ``chatterbox-tts`` package and its torch weights are a heavy, GPU-hungry
optional dependency not installed in this environment, so every import of it
(and of the other optional audio libs it needs) is kept strictly inside
:meth:`ChatterboxTurboTTSEngine._render` and its private helpers — module
import, and therefore registry discovery, never touches them.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

# Chatterbox-Turbo's S3Gen vocoder output rate, per the resemble-ai/chatterbox
# Turbo model card / README (``ChatterboxTurboTTS.sr``); used as a fallback if
# a loaded model does not expose ``sr`` for some reason.
_TURBO_NATIVE_SAMPLE_RATE = 24_000


@TTS_ENGINES.register("chatterbox_turbo")
class ChatterboxTurboTTSEngine(BaseTTSEngine):
    """Resemble AI Chatterbox-Turbo: faster zero-shot codec-token cloning TTS."""

    name = "chatterbox_turbo"
    supports_cloning = True
    #: Resemble AI's Chatterbox-Turbo checkpoint, fetched by ``from_pretrained()``
    #: under the hood; not parameterized in this code path, so declared here.
    model_ids: tuple[str, ...] = ("ResembleAI/chatterbox-turbo",)

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Clone ``voice.reference_audio`` and speak ``text`` with Turbo.

        Returns 16 kHz mono float32 audio, resampling down from Turbo's native
        vocoder output rate. ``BaseTTSEngine`` already guarantees
        ``voice.reference_audio`` is set before ``_render`` is ever called.
        """
        from chatterbox.tts_turbo import ChatterboxTurboTTS  # type: ignore

        model = ChatterboxTurboTTS.from_pretrained(device="cuda")
        native_rate = getattr(model, "sr", _TURBO_NATIVE_SAMPLE_RATE)

        with self._reference_wav_path(voice) as prompt_path:
            wav = model.generate(text, audio_prompt_path=prompt_path)

        return self._to_16k(wav, native_rate)

    @staticmethod
    @contextlib.contextmanager
    def _reference_wav_path(voice: VoiceReference) -> Iterator[str]:
        """Materialise ``voice.reference_audio`` as a temp WAV file path.

        Turbo's cloning API takes a file path, not an in-memory array, so the
        reference clip must live on disk for one ``generate`` call. The file is
        removed again once the caller's ``with`` block exits.
        """
        import soundfile as sf  # type: ignore

        reference = voice.reference_audio
        assert reference is not None  # enforced by BaseTTSEngine._check_voice

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(path, reference.samples, reference.sample_rate)
            yield path
        finally:
            os.remove(path)

    @staticmethod
    def _to_16k(raw: object, native_rate: int) -> AudioBuffer:
        """Downmix/resample Turbo's raw waveform to 16 kHz mono."""
        from scipy.signal import resample_poly  # type: ignore

        samples: NDArray[np.float32] = np.asarray(raw, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=0).astype(np.float32)
        resampled = resample_poly(samples, SAMPLE_RATE, native_rate)
        clipped = np.clip(resampled, -1.0, 1.0).astype(np.float32)
        return AudioBuffer(clipped, SAMPLE_RATE)
