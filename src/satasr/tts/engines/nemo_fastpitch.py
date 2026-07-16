"""NVIDIA NeMo FastPitch TTS engine — multi-speaker, non-autoregressive.

FastPitch (Lancucki, 2021) is a feed-forward, non-autoregressive
transformer that predicts pitch and duration explicitly before generating a
mel-spectrogram, which a separate neural vocoder (HiFi-GAN) converts to
waveform. This module loads NVIDIA's pretrained multi-speaker FastPitch
checkpoint (trained on the HiFiTTS corpus) via the NeMo toolkit's
`nemo.collections.tts` API, paired with a matching HiFi-GAN vocoder
checkpoint.

FastPitch has no reference-audio voice-cloning API of its own: HiFiTTS
speakers are baked into the checkpoint as a fixed embedding table, selected
by integer speaker id. This engine is therefore preset-only and honors
``VoiceReference.preset`` as that HiFiTTS speaker id (e.g. ``"92"``;
design §4.4).

License note: NeMo is Apache-2.0, but the pretrained checkpoints are
distributed under NVIDIA's own model license (see the NGC/HuggingFace model
card) — verify terms before production use, per issue #21.

The ``nemo_toolkit[tts]`` package and its torch checkpoints are a heavy,
GPU-hungry optional dependency not installed in this environment, so the
import is kept strictly inside :meth:`NemoFastPitchTTSEngine._render` —
module import (and therefore registry discovery) never touches it.
"""

from __future__ import annotations

import numpy as np

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

#: Pretrained multi-speaker FastPitch checkpoint (HiFiTTS), per the NeMo/NGC
#: model zoo. Start with FastPitch+HiFi-GAN; RAD-TTS/Mixer-TTS can register
#: as sibling engines later (issue #21).
_SPEC_MODEL_NAME = "tts_en_fastpitch_multispeaker"
#: Matching HiFi-GAN vocoder checkpoint, fine-tuned for the above spectrogram
#: model's mel statistics.
_VOCODER_MODEL_NAME = "tts_en_hifitts_hifigan_ft_fastpitch"
#: HiFiTTS studio recordings' native sample rate, per the model card.
_NEMO_NATIVE_SAMPLE_RATE = 44_100
#: Fallback HiFiTTS speaker id used only if a caller passes an empty string
#: (guard clause; BaseTTSEngine already rejects ``preset is None`` before we
#: get here).
_DEFAULT_SPEAKER_ID = "92"


@TTS_ENGINES.register("nemo_fastpitch")
class NemoFastPitchTTSEngine(BaseTTSEngine):
    """NeMo FastPitch + HiFi-GAN: preset multi-speaker, non-autoregressive TTS."""

    name = "nemo_fastpitch"
    supports_cloning = False

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Synthesize ``text`` with FastPitch's preset ``voice.preset`` speaker.

        Generates a mel-spectrogram with FastPitch conditioned on the
        requested HiFiTTS speaker id, vocodes it with HiFi-GAN, and resamples
        the result down to the project's 16 kHz mono float32 convention.
        """
        from nemo.collections.tts.models import (  # type: ignore
            FastPitchModel,
            HifiGanModel,
        )

        spec_model = FastPitchModel.from_pretrained(_SPEC_MODEL_NAME)
        vocoder = HifiGanModel.from_pretrained(_VOCODER_MODEL_NAME)
        speaker_id = int(voice.preset or _DEFAULT_SPEAKER_ID)

        tokens = spec_model.parse(text)
        spectrogram = spec_model.generate_spectrogram(tokens=tokens, speaker=speaker_id)
        raw_audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
        return self._to_16k(raw_audio.detach().cpu().numpy())

    @staticmethod
    def _to_16k(raw: np.ndarray) -> AudioBuffer:
        """Downmix/resample NeMo's raw waveform to the project's 16 kHz mono.

        Uses plain numpy linear interpolation rather than scipy so this
        helper (and its fast test) needs no extra dependency beyond the
        project's existing numpy requirement.
        """
        samples = np.asarray(raw, dtype=np.float32).reshape(-1)
        resampled = _linear_resample(samples, _NEMO_NATIVE_SAMPLE_RATE, SAMPLE_RATE)
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
