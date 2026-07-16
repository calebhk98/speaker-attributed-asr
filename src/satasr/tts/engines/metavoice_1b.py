"""MetaVoice-1B TTS engine — zero-shot voice cloning via codec tokens.

MetaVoice-1B is an autoregressive codec-token model released by MetaVoice
under the Apache-2.0 license (verify terms before production use). Like other
cloning engines (e.g. XTTS v2), it clones a voice from a short reference clip
at inference time, so it requires ``VoiceReference.reference_audio`` (design
§4.4: clone real recorded speakers, not a built-in bank) and never consults
``preset``.

The ``fam`` package (MetaVoice's inference library) and its GPU weights are a
heavy optional dependency not installed in this environment, so every
third-party import stays strictly inside the render path — module import
(and therefore registry discovery) never touches them.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES


@TTS_ENGINES.register("metavoice_1b")
class MetaVoice1BTTSEngine(BaseTTSEngine):
    """MetaVoice-1B: zero-shot cloning, autoregressive codec-token TTS."""

    name = "metavoice_1b"
    supports_cloning = True

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Clone ``voice.reference_audio`` and speak ``text`` with MetaVoice-1B.

        ``voice.reference_audio`` is guaranteed non-None here —
        ``BaseTTSEngine._check_voice`` enforces the cloning contract before
        ``_render`` is ever called. Returns 16 kHz mono float32 audio,
        resampled down from whatever sample rate MetaVoice renders at
        (published as 24 kHz, EnCodec-based).
        """
        from fam.llm.fast_inference import TTS  # type: ignore

        model = TTS()
        with self._reference_wav_path(voice) as ref_path:
            wav_path = model.synthesise(text=text, spk_ref_path=ref_path)
        return self._load_and_resample(wav_path)

    @staticmethod
    @contextmanager
    def _reference_wav_path(voice: VoiceReference) -> Iterator[str]:
        """Write the in-memory reference clip to a temp WAV MetaVoice can read.

        MetaVoice's ``TTS.synthesise`` API takes ``spk_ref_path`` as a file
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
    def _load_and_resample(wav_path: str) -> AudioBuffer:
        """Load MetaVoice's rendered wav file and downmix/resample to 16 kHz.

        ``TTS.synthesise`` writes its output to disk and returns the path
        rather than raw samples, so the result is read back before being
        normalised to the project's mono 16 kHz convention.
        """
        import numpy as np
        import soundfile as sf  # already type: ignore'd on its first import above
        from scipy.signal import resample_poly  # type: ignore

        samples, source_rate = sf.read(wav_path, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=-1).astype(np.float32)
        resampled = resample_poly(samples, SAMPLE_RATE, source_rate)
        clipped = np.clip(resampled, -1.0, 1.0).astype(np.float32)
        return AudioBuffer(clipped, SAMPLE_RATE)
