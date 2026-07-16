"""XTTS v2 (Coqui) TTS engine — zero-shot voice cloning via codec tokens.

XTTS v2 is an autoregressive codec-token model distributed by Coqui community
maintainers under the CPML license (non-commercial — verify terms before
production use). Unlike preset engines (e.g. Bark), XTTS v2 clones a voice
from a short reference clip at inference time, so it requires
``VoiceReference.reference_audio`` (design §4.4: clone real recorded
speakers, not a built-in bank) and never consults ``preset``.

The ``TTS`` (coqui-tts) package and its GPU weights are a heavy optional
dependency not installed in this environment, so every third-party import
stays strictly inside the render path — module import (and therefore
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

#: Coqui model id for the multilingual XTTS v2 checkpoint.
_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
#: XTTS v2's native output sample rate, per the model card.
_XTTS_NATIVE_SAMPLE_RATE = 24_000
#: Synthesis language. VoiceReference has no language field yet, so every
#: clip is synthesized as English; revisit if multilingual voices are needed.
_DEFAULT_LANGUAGE = "en"


@TTS_ENGINES.register("xtts_v2")
class XttsV2TTSEngine(BaseTTSEngine):
    """Coqui XTTS v2: zero-shot cloning, autoregressive codec-token TTS."""

    name = "xtts_v2"
    supports_cloning = True

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Clone ``voice.reference_audio`` and speak ``text`` with XTTS v2.

        ``voice.reference_audio`` is guaranteed non-None here —
        ``BaseTTSEngine._check_voice`` enforces the cloning contract before
        ``_render`` is ever called. Returns 16 kHz mono float32 audio,
        resampled down from XTTS's native 24 kHz output.
        """
        import torch  # type: ignore
        from TTS.api import TTS  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TTS(_MODEL_NAME).to(device)
        with self._reference_wav_path(voice) as ref_path:
            wav = model.tts(text=text, speaker_wav=ref_path, language=_DEFAULT_LANGUAGE)
        return self._to_16k(wav)

    @staticmethod
    @contextmanager
    def _reference_wav_path(voice: VoiceReference) -> Iterator[str]:
        """Write the in-memory reference clip to a temp WAV Coqui can read.

        Coqui's ``TTS.tts`` API takes ``speaker_wav`` as a file path, not raw
        samples, so the ``AudioBuffer`` on ``voice`` is spilled to a scratch
        file for the duration of one synthesis call and cleaned up after.
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
        """Downmix/resample XTTS's raw waveform to the project's 16 kHz mono."""
        import numpy as np
        from scipy.signal import resample_poly  # type: ignore

        samples = np.asarray(wav, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=-1).astype(np.float32)
        resampled = resample_poly(samples, SAMPLE_RATE, _XTTS_NATIVE_SAMPLE_RATE)
        clipped = np.clip(resampled, -1.0, 1.0).astype(np.float32)
        return AudioBuffer(clipped, SAMPLE_RATE)
