"""VibeVoice (Microsoft) TTS engine — dialogue-native, cloned-voice synthesis.

VibeVoice is a long-form conversational TTS model built around continuous
acoustic/semantic speech tokenizers (7.5 Hz frame rate) feeding an LLM
backbone with a diffusion generation head. It natively supports multi-speaker
dialogue, but here it is driven one speaker at a time through the shared
:class:`~satasr.tts.base.BaseTTSEngine` contract. Voices are cloned from a
short reference clip (``voice_samples`` in the model's own API) rather than
picked from a preset bank, so this engine requires
``VoiceReference.reference_audio`` (design §4.4).

The ``vibevoice`` package, its Qwen2.5-backed weights, and a GPU are a heavy
optional dependency not installed in this environment, so every third-party
import stays strictly inside the render path — module import, and therefore
registry discovery, never touches them.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
from numpy.typing import NDArray

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

#: HF checkpoint for the smallest published VibeVoice model.
_MODEL_NAME = "microsoft/VibeVoice-1.5B"
#: VibeVoice's native output rate (its tokenizers downsample 24 kHz audio
#: 3200x), per the model card.
_VIBEVOICE_NATIVE_SAMPLE_RATE = 24_000


@TTS_ENGINES.register("vibevoice")
class VibeVoiceTTSEngine(BaseTTSEngine):
    """Microsoft VibeVoice: dialogue-native, zero-shot voice-cloning TTS."""

    name = "vibevoice"
    supports_cloning = True
    #: VibeVoice checkpoint this engine loads (see ``_MODEL_NAME`` above).
    model_ids: tuple[str, ...] = (_MODEL_NAME,)

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Clone ``voice.reference_audio`` and speak ``text`` with VibeVoice.

        ``voice.reference_audio`` is guaranteed non-None here —
        ``BaseTTSEngine._check_voice`` enforces the cloning contract before
        ``_render`` is ever called. VibeVoice is dialogue-native (it can
        script several speakers at once), but this call always renders a
        single-speaker line so ``synthesize`` keeps returning one clip per
        the shared engine contract. Returns 16 kHz mono float32 audio,
        resampled down from VibeVoice's native 24 kHz output.
        """
        import torch  # type: ignore
        from vibevoice.modular.modeling_vibevoice_inference import (  # type: ignore
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_processor import (  # type: ignore
            VibeVoiceProcessor,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = VibeVoiceProcessor.from_pretrained(_MODEL_NAME)
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            _MODEL_NAME
        ).to(device)

        with self._reference_wav_path(voice) as ref_path:
            inputs = processor(
                text=[text], voice_samples=[[ref_path]], return_tensors="pt"
            ).to(device)
            output = model.generate(**inputs, tokenizer=processor.tokenizer)

        wav = output.speech_outputs[0]
        return self._to_16k(wav)

    @staticmethod
    @contextmanager
    def _reference_wav_path(voice: VoiceReference) -> Iterator[str]:
        """Materialise ``voice.reference_audio`` as a temp WAV file path.

        VibeVoice's processor takes ``voice_samples`` as file paths, not raw
        arrays, so the reference clip is spilled to a scratch file for the
        duration of one synthesis call and removed once the caller's ``with``
        block exits.
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
    def _to_16k(raw: object) -> AudioBuffer:
        """Downmix/resample VibeVoice's raw waveform to 16 kHz mono."""
        from scipy.signal import resample_poly  # type: ignore

        samples: NDArray[np.float32] = np.asarray(raw, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=0).astype(np.float32)
        resampled = resample_poly(samples, SAMPLE_RATE, _VIBEVOICE_NATIVE_SAMPLE_RATE)
        clipped = np.clip(resampled, -1.0, 1.0).astype(np.float32)
        return AudioBuffer(clipped, SAMPLE_RATE)
