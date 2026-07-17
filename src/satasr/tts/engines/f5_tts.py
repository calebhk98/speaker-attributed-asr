"""F5-TTS engine — zero-shot voice cloning via flow matching.

F5-TTS (SWivid/F5-TTS) is a community/academic, non-autoregressive
flow-matching TTS model distributed under CC-BY-NC 4.0 (non-commercial —
verify terms before production use). Like XTTS v2, it clones a voice from a
short reference clip at inference time rather than picking from a preset
bank, so it requires ``VoiceReference.reference_audio`` (design §4.4) and
never consults ``preset``.

The ``f5-tts`` package and its GPU weights are a heavy optional dependency
not installed in this environment, so every third-party import stays
strictly inside the render path — module import (and therefore registry
discovery) never touches them. Being a non-autoregressive/diffusion model,
it is also not suited to bulk CPU generation (design §3).
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

#: F5-TTS's own default checkpoint name, per the project's inference API.
_MODEL_NAME = "F5TTS_v1_Base"
#: F5-TTS's native vocoder output sample rate (Vocos @ 24 kHz mel target).
_F5_NATIVE_SAMPLE_RATE = 24_000
#: Empty reference transcript tells the F5-TTS API to auto-transcribe the
#: reference clip (it falls back to an internal ASR model); VoiceReference
#: carries no transcript field today, so this is the only option available.
_AUTO_TRANSCRIBE_REF_TEXT = ""


@TTS_ENGINES.register("f5_tts")
class F5TTSEngine(BaseTTSEngine):
    """F5-TTS: zero-shot cloning, non-autoregressive flow-matching TTS."""

    name = "f5_tts"
    supports_cloning = True
    #: F5-TTS checkpoint name this engine loads (see ``_MODEL_NAME`` above).
    model_ids: tuple[str, ...] = (_MODEL_NAME,)

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Clone ``voice.reference_audio`` and speak ``text`` with F5-TTS.

        ``voice.reference_audio`` is guaranteed non-None here —
        ``BaseTTSEngine._check_voice`` enforces the cloning contract before
        ``_render`` is ever called. Returns 16 kHz mono float32 audio,
        resampled down from F5-TTS's native 24 kHz output.
        """
        from f5_tts.api import F5TTS  # type: ignore

        model = F5TTS(model=_MODEL_NAME)
        with self._reference_wav_path(voice) as ref_path:
            wav, native_rate, _spectrogram = model.infer(
                ref_file=ref_path,
                ref_text=_AUTO_TRANSCRIBE_REF_TEXT,
                gen_text=text,
            )
        return self._to_16k(wav, native_rate)

    @staticmethod
    @contextmanager
    def _reference_wav_path(voice: VoiceReference) -> Iterator[str]:
        """Write the in-memory reference clip to a temp WAV F5-TTS can read.

        F5-TTS's ``F5TTS.infer`` API takes ``ref_file`` as a file path, not
        raw samples, so the ``AudioBuffer`` on ``voice`` is spilled to a
        scratch file for the duration of one synthesis call and cleaned up
        after.
        """
        import soundfile as sf  # type: ignore

        assert voice.reference_audio is not None  # enforced by base class
        with tempfile.NamedTemporaryFile(suffix=".wav") as ref_file:
            sf.write(
                ref_file.name,
                voice.reference_audio.samples,
                voice.reference_audio.sample_rate,
            )
            yield ref_file.name

    @staticmethod
    def _to_16k(wav: object, native_rate: int | None) -> AudioBuffer:
        """Downmix/resample F5-TTS's raw waveform to 16 kHz mono."""
        import numpy as np
        from scipy.signal import resample_poly  # type: ignore

        source_rate = native_rate or _F5_NATIVE_SAMPLE_RATE
        samples = np.asarray(wav, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=-1).astype(np.float32)
        resampled = resample_poly(samples, SAMPLE_RATE, source_rate)
        clipped = np.clip(resampled, -1.0, 1.0).astype(np.float32)
        return AudioBuffer(clipped, SAMPLE_RATE)
