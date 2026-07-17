"""Kokoro (hexgrad) TTS engine — lightweight decoder-only synthesis.

Kokoro-82M is a small, fast, Apache-2.0 licensed model shipped with a fixed
bank of 54 preset voices (no voice-cloning API). Callers pick a voice via
``VoiceReference.preset`` naming one of those presets (e.g. ``"af_heart"``);
this engine is preset-only, mirroring Bark's structure.

The ``kokoro`` package and its torch weights are a heavy, GPU-hungry optional
dependency not installed in this environment, so the import is kept strictly
inside :meth:`KokoroTTSEngine._render` — module import (and therefore registry
discovery) never touches it.
"""

from __future__ import annotations

import numpy as np

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

#: Kokoro's native decoder output sample rate, per the model card / KPipeline docs.
_KOKORO_NATIVE_SAMPLE_RATE = 24_000

#: The 54 preset voices shipped with Kokoro-82M v1.0 (hexgrad/Kokoro-82M
#: VOICES.md), one whitespace-separated group per language their first
#: letter selects: American/British English, Japanese, Mandarin, Spanish,
#: French, Hindi, Italian, Brazilian Portuguese.
_VOICE_GROUPS = (
    "af_alloy af_aoede af_bella af_heart af_jessica af_kore af_nicole af_nova "
    "af_river af_sarah af_sky am_adam am_echo am_eric am_fenrir am_liam "
    "am_michael am_onyx am_puck am_santa",
    "bf_alice bf_emma bf_isabella bf_lily bm_daniel bm_fable bm_george bm_lewis",
    "jf_alpha jf_gongitsune jf_nezumi jf_tebukuro jm_kumo",
    "zf_xiaobei zf_xiaoni zf_xiaoxiao zf_xiaoyi zm_yunjian zm_yunxi zm_yunxia "
    "zm_yunyang",
    "ef_dora em_alex em_santa",
    "ff_siwis",
    "hf_alpha hf_beta hm_omega hm_psi",
    "if_sara im_nicola",
    "pf_dora pm_alex pm_santa",
)
_VOICES = frozenset(word for group in _VOICE_GROUPS for word in group.split())


@TTS_ENGINES.register("kokoro")
class KokoroTTSEngine(BaseTTSEngine):
    """Kokoro-82M: preset-voice, lightweight decoder-only TTS."""

    name = "kokoro"
    supports_cloning = False
    #: Kokoro-82M's HF repo (see the VOICES.md reference above);
    #: ``KPipeline()`` fetches it by default, not parameterized in this code
    #: path, so declared explicitly here.
    model_ids: tuple[str, ...] = ("hexgrad/Kokoro-82M",)

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Synthesize ``text`` with Kokoro's preset ``voice.preset``.

        Returns 16 kHz mono float32 audio, resampled down from Kokoro's
        native 24 kHz decoder output. Raises ``ValueError`` up front if the
        preset isn't one of the 54 shipped voices, before any heavy import.
        """
        preset = self._known_preset(voice.preset)
        from kokoro import KPipeline  # type: ignore

        pipeline = KPipeline(lang_code=preset[0])
        chunks = [audio for _, _, audio in pipeline(text, voice=preset)]
        raw = np.concatenate([np.asarray(c, dtype=np.float32) for c in chunks])
        return self._to_16k(raw)

    @staticmethod
    def _known_preset(preset: str | None) -> str:
        """Validate ``preset`` against Kokoro's 54-voice bank, or raise."""
        if preset not in _VOICES:
            raise ValueError(
                f"kokoro has no preset voice {preset!r}; expected one of "
                f"the 54 shipped presets (e.g. 'af_heart')"
            )
        return preset

    @staticmethod
    def _to_16k(raw: np.ndarray) -> AudioBuffer:
        """Downmix/resample Kokoro's raw waveform to the project's 16 kHz mono.

        Uses plain numpy linear interpolation rather than scipy so this
        helper (and its fast test) needs no extra dependency.
        """
        samples = np.asarray(raw, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=-1).astype(np.float32)
        resampled = _linear_resample(samples, _KOKORO_NATIVE_SAMPLE_RATE, SAMPLE_RATE)
        clipped = np.clip(resampled, -1.0, 1.0).astype(np.float32)
        return AudioBuffer(clipped, SAMPLE_RATE)


def _linear_resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a 1-D float32 array from ``src_rate`` to ``dst_rate`` Hz."""
    if samples.size == 0:
        return samples
    duration_s = samples.shape[0] / src_rate
    dst_count = max(1, round(duration_s * dst_rate))
    src_times = np.arange(samples.shape[0], dtype=np.float64) / src_rate
    dst_times = np.arange(dst_count, dtype=np.float64) / dst_rate
    interpolated: np.ndarray = np.interp(dst_times, src_times, samples)
    return interpolated.astype(np.float32)
