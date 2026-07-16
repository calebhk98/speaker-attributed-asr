"""GPT-SoVITS TTS engine — few-shot voice cloning via codec-token AR + VITS.

GPT-SoVITS (community, RVC-Boss, MIT license — verify terms before production
use) is an autoregressive codec-token model paired with a VITS-style decoder.
It clones a voice from a short reference clip at inference time (design §4.4:
clone real recorded speakers), so it requires ``VoiceReference.reference_audio``
and never consults ``preset``.

GPT-SoVITS also supports an optional "prompt text" transcript of the reference
clip to improve cloning quality. ``VoiceReference`` carries no such field, so
this engine relies on GPT-SoVITS's no-reference-text mode (empty
``prompt_text``), trading a little fidelity for API simplicity — revisit if
``VoiceReference`` grows a transcript field.

The ``GPT_SoVITS`` package, its pretrained checkpoints, and a GPU are a heavy
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

#: Packaged default inference config shipped with the GPT-SoVITS repo; points
#: at the pretrained GPT + SoVITS checkpoints on disk.
_CONFIG_PATH = "GPT_SoVITS/configs/tts_infer.yaml"
#: GPT-SoVITS's native decoder output rate, per the model's inference pack.
_GPT_SOVITS_NATIVE_SAMPLE_RATE = 32_000
#: Synthesis language. VoiceReference has no language field yet, so every
#: clip is synthesized as English; revisit if multilingual voices are needed.
_DEFAULT_LANGUAGE = "en"


@TTS_ENGINES.register("gpt_sovits")
class GptSovitsTTSEngine(BaseTTSEngine):
    """GPT-SoVITS: few-shot cloning, autoregressive codec-token TTS."""

    name = "gpt_sovits"
    supports_cloning = True

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Clone ``voice.reference_audio`` and speak ``text`` with GPT-SoVITS.

        ``voice.reference_audio`` is guaranteed non-None here —
        ``BaseTTSEngine._check_voice`` enforces the cloning contract before
        ``_render`` is ever called. Returns 16 kHz mono float32 audio,
        resampled down from GPT-SoVITS's native 32 kHz output.
        """
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # type: ignore

        pipeline = TTS(TTS_Config(_CONFIG_PATH))
        with self._reference_wav_path(voice) as ref_path:
            request = {
                "text": text,
                "text_lang": _DEFAULT_LANGUAGE,
                "ref_audio_path": ref_path,
                "prompt_text": "",
                "prompt_lang": _DEFAULT_LANGUAGE,
            }
            sample_rate, wav = next(pipeline.run(request))
        return self._to_16k(wav, sample_rate)

    @staticmethod
    @contextmanager
    def _reference_wav_path(voice: VoiceReference) -> Iterator[str]:
        """Write the in-memory reference clip to a temp WAV GPT-SoVITS reads.

        GPT-SoVITS's inference pipeline takes ``ref_audio_path`` as a file
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
    def _to_16k(wav: object, native_sample_rate: int) -> AudioBuffer:
        """Downmix/resample GPT-SoVITS's raw waveform to 16 kHz mono."""
        import numpy as np
        from scipy.signal import resample_poly  # type: ignore

        samples = np.asarray(wav, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=-1).astype(np.float32)
        rate = native_sample_rate or _GPT_SOVITS_NATIVE_SAMPLE_RATE
        resampled = resample_poly(samples, SAMPLE_RATE, rate)
        clipped = np.clip(resampled, -1.0, 1.0).astype(np.float32)
        return AudioBuffer(clipped, SAMPLE_RATE)
