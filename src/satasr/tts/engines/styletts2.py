"""StyleTTS 2 TTS engine — diffusion-based, reference-cloned voice style.

StyleTTS 2 (Li et al., academic release, MIT license — verify terms before
production use) is a non-autoregressive model that predicts speech styles
with a diffusion process instead of relying on a fixed reference embedding.
It clones a voice from a short reference clip at inference time (design
§4.4: clone real recorded speakers), so it requires
``VoiceReference.reference_audio`` and never consults ``preset``.

The ``styletts2`` package, its checkpoints, and (ideally) a GPU are a heavy
optional dependency not installed in this environment, so every third-party
import stays strictly inside the render path — module import (and therefore
registry discovery) never touches them.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

#: StyleTTS 2's native output sample rate, per the LibriTTS-trained release.
_STYLETTS2_NATIVE_SAMPLE_RATE = 24_000


@TTS_ENGINES.register("styletts2")
class StyleTTS2TTSEngine(BaseTTSEngine):
    """StyleTTS 2: diffusion-based, reference-cloned, non-autoregressive TTS."""

    name = "styletts2"
    supports_cloning = True

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Clone ``voice.reference_audio`` and speak ``text`` with StyleTTS 2.

        ``voice.reference_audio`` is guaranteed non-None here —
        ``BaseTTSEngine._check_voice`` enforces the cloning contract before
        ``_render`` is ever called. Returns 16 kHz mono float32 audio,
        resampled down from StyleTTS 2's native 24 kHz output.
        """
        from styletts2.tts import StyleTTS2  # type: ignore

        model = StyleTTS2()
        with self._reference_wav_path(voice) as ref_path:
            wav = model.inference(text, target_voice_path=ref_path)
        return self._to_16k(wav)

    @staticmethod
    @contextmanager
    def _reference_wav_path(voice: VoiceReference) -> Iterator[str]:
        """Write the in-memory reference clip to a temp WAV StyleTTS 2 can read.

        StyleTTS 2's ``inference`` API takes ``target_voice_path`` as a file
        path, not raw samples, so the ``AudioBuffer`` on ``voice`` is spilled
        to a scratch file for the duration of one synthesis call and cleaned
        up after.
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
    def _to_16k(wav: object) -> AudioBuffer:
        """Downmix/resample StyleTTS 2's raw waveform to project 16 kHz mono."""
        import numpy as np
        from scipy.signal import resample_poly  # type: ignore

        samples = np.asarray(wav, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=-1).astype(np.float32)
        resampled = resample_poly(samples, SAMPLE_RATE, _STYLETTS2_NATIVE_SAMPLE_RATE)
        clipped = np.clip(resampled, -1.0, 1.0).astype(np.float32)
        return AudioBuffer(clipped, SAMPLE_RATE)
