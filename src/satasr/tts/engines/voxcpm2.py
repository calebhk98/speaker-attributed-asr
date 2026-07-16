"""VoxCPM2 (OpenBMB) TTS engine — zero-shot voice cloning via codec tokens.

VoxCPM2 is an autoregressive codec-token model released by OpenBMB under the
Apache-2.0 license (verify terms before production use). Like other cloning
engines (e.g. XTTS v2), it synthesizes speech in the timbre of a short
reference clip rather than picking from a preset bank, so it requires
``VoiceReference.reference_audio`` (design §4.4: clone real recorded
speakers) and never consults ``preset``.

The ``voxcpm`` package and its GPU weights are a heavy optional dependency
not installed in this environment, so every third-party import stays
strictly inside the render path — module import (and therefore registry
discovery) never touches them.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

#: OpenBMB's published VoxCPM2 checkpoint id on the Hugging Face Hub.
_MODEL_NAME = "openbmb/VoxCPM2"
#: VoxCPM2's native output sample rate, per the model card.
_VOXCPM2_NATIVE_SAMPLE_RATE = 24_000


@TTS_ENGINES.register("voxcpm2")
class VoxCpm2TTSEngine(BaseTTSEngine):
    """OpenBMB VoxCPM2: zero-shot cloning, autoregressive codec-token TTS."""

    name = "voxcpm2"
    supports_cloning = True
    #: OpenBMB checkpoint id VoxCPM2 loads (see ``_MODEL_NAME`` above).
    model_ids: tuple[str, ...] = (_MODEL_NAME,)

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Clone ``voice.reference_audio`` and speak ``text`` with VoxCPM2.

        ``voice.reference_audio`` is guaranteed non-None here —
        ``BaseTTSEngine._check_voice`` enforces the cloning contract before
        ``_render`` is ever called. Returns 16 kHz mono float32 audio,
        resampled down from VoxCPM2's native 24 kHz output.
        """
        from voxcpm import VoxCPM  # type: ignore

        model = VoxCPM.from_pretrained(_MODEL_NAME)
        with self._reference_wav_path(voice) as ref_path:
            wav = model.generate(text=text, prompt_wav_path=ref_path)
        return self._to_16k(wav)

    @staticmethod
    @contextmanager
    def _reference_wav_path(voice: VoiceReference) -> Iterator[str]:
        """Write the in-memory reference clip to a temp WAV VoxCPM2 can read.

        VoxCPM2's ``generate`` API takes ``prompt_wav_path`` as a file path,
        not raw samples, so the ``AudioBuffer`` on ``voice`` is spilled to a
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
    def _to_16k(wav: object) -> AudioBuffer:
        """Downmix/resample VoxCPM2's raw waveform to the project's 16 kHz mono."""
        import numpy as np
        from scipy.signal import resample_poly  # type: ignore

        samples = np.asarray(wav, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=-1).astype(np.float32)
        resampled = resample_poly(samples, SAMPLE_RATE, _VOXCPM2_NATIVE_SAMPLE_RATE)
        clipped = np.clip(resampled, -1.0, 1.0).astype(np.float32)
        return AudioBuffer(clipped, SAMPLE_RATE)
