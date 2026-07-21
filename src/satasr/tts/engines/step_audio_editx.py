"""Step-Audio-EditX (StepFun) TTS engine — LLM-based zero-shot cloning TTS.

Step-Audio-EditX is StepFun's 3B-parameter, LLM-based audio model (Apache 2.0)
that, alongside its emotion/style editing, does robust zero-shot text-to-speech
cloning: given a short reference clip it reproduces that speaker for new text
(design §4.4). This engine uses only that cloning path, so it is cloning-only
and requires ``VoiceReference.reference_audio``.

ASSUMPTIONS (unverified here — no GPU/weights in this environment; from the
``stepfun-ai/Step-Audio-EditX`` repo's ``tts.py`` and README):
- The optional inference code installs importably as ``step_audio_editx``,
  exposing ``StepAudioTTS`` and ``StepAudioTokenizer``.
- ``StepAudioTTS(model_id, tokenizer).clone(prompt_wav_path=, prompt_text=,
  target_text=)`` returns a ``(waveform, sample_rate)`` tuple at 24 kHz.
- The clone API takes the reference as an on-disk WAV path (like Chatterbox).
- ``VoiceReference`` has no transcript field, so ``prompt_text`` is passed as
  ``""`` and relies on the model's zero-shot behaviour — unverified without
  real weights (the same limitation the ``qwen3_tts`` engine documents).

The Step-Audio inference code and its torch weights are a heavy, GPU-hungry
optional dependency not installed here, so every import (and the other optional
audio libs it needs) is kept strictly inside
:meth:`StepAudioEditXTTSEngine._render` and its helpers — module import, and
therefore registry discovery, never touches them.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator

import numpy as np

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

#: Step-Audio-EditX vocoder output rate, per ``tts.py`` (``clone`` returns 24k).
_STEP_NATIVE_SAMPLE_RATE = 24_000
#: The EditX model plus its companion audio tokenizer, both required at runtime.
_MODEL_ID = "stepfun-ai/Step-Audio-EditX"
_TOKENIZER_ID = "stepfun-ai/Step-Audio-Tokenizer"


@TTS_ENGINES.register("step_audio_editx")
class StepAudioEditXTTSEngine(BaseTTSEngine):
    """StepFun Step-Audio-EditX: LLM-based zero-shot voice-cloning TTS."""

    name = "step_audio_editx"
    supports_cloning = True
    #: Weights fetched ahead of time by the downloader (see module docstring).
    model_ids: tuple[str, ...] = (_MODEL_ID, _TOKENIZER_ID)

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Clone ``voice.reference_audio`` and speak ``text`` with Step-Audio.

        Returns 16 kHz mono float32 audio, resampled down from the model's
        native 24 kHz output. ``BaseTTSEngine`` already guarantees
        ``voice.reference_audio`` is set before ``_render`` is ever called.
        """
        from step_audio_editx import StepAudioTokenizer, StepAudioTTS  # type: ignore

        model = StepAudioTTS(_MODEL_ID, StepAudioTokenizer(_TOKENIZER_ID))
        with self._reference_wav_path(voice) as prompt_path:
            waveform, native_rate = model.clone(
                prompt_wav_path=prompt_path,
                prompt_text="",  # no transcript in VoiceReference; see docstring
                target_text=text,
            )
        return self._to_16k(waveform, native_rate)

    @staticmethod
    @contextlib.contextmanager
    def _reference_wav_path(voice: VoiceReference) -> Iterator[str]:
        """Materialise ``voice.reference_audio`` as a temp WAV file path.

        Step-Audio's clone API takes a file path, not an in-memory array, so
        the reference clip lives on disk for one ``clone`` call and is removed
        once the caller's ``with`` block exits.
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
        """Downmix/resample Step-Audio's raw waveform to the project 16 kHz mono.

        Uses plain numpy linear interpolation rather than scipy so this helper
        (and its fast test) needs no extra dependency beyond numpy.
        """
        samples = np.asarray(raw, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=-1).astype(np.float32)
        resampled = _linear_resample(samples, native_rate, SAMPLE_RATE)
        clipped = np.clip(resampled, -1.0, 1.0).astype(np.float32)
        return AudioBuffer(clipped, SAMPLE_RATE)


def _linear_resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a 1-D float32 array from ``src_rate`` to ``dst_rate`` Hz."""
    if samples.size == 0 or src_rate == dst_rate:
        return samples
    duration_s = samples.shape[0] / src_rate
    dst_count = max(1, round(duration_s * dst_rate))
    src_times = np.arange(samples.shape[0], dtype=np.float64) / src_rate
    dst_times = np.arange(dst_count, dtype=np.float64) / dst_rate
    interpolated: np.ndarray = np.interp(dst_times, src_times, samples)
    return interpolated.astype(np.float32)
