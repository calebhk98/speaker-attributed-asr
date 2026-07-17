"""MeloTTS (MyShell) engine — multilingual, multi-speaker TTS.

MeloTTS ships one model per language, each with a small fixed bank of preset
speakers (no voice-cloning API): e.g. the English model exposes "EN-US",
"EN-BR", "EN_INDIA", "EN-AU" and "EN-Default"; the other languages each ship a
single default speaker named after the language code. Callers pick a voice via
``VoiceReference.preset`` naming one of those speaker ids (design §4.4); this
engine is preset-only.

The ``melo`` package (MeloTTS) and its torch weights are a heavy, GPU-hungry
optional dependency not installed in this environment, so the import is kept
strictly inside :meth:`MeloTTSEngine._render` — module import (and therefore
registry discovery) never touches it.
"""

from __future__ import annotations

import numpy as np

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

#: MeloTTS's native decoder output sample rate, per every shipped model's
#: config.json ("data": {"sampling_rate": 44100}).
_MELOTTS_NATIVE_SAMPLE_RATE = 44_100

#: The preset speaker ids MeloTTS ships, mapped to the language model that
#: must be loaded to synthesize them (MyShell-ai/MeloTTS README speaker table).
_PRESET_LANGUAGES = {
    "EN-US": "EN",
    "EN-BR": "EN",
    "EN_INDIA": "EN",
    "EN-AU": "EN",
    "EN-Default": "EN",
    "ES": "ES",
    "FR": "FR",
    "ZH": "ZH",
    "JP": "JP",
    "KR": "KR",
}


@TTS_ENGINES.register("melotts")
class MeloTTSEngine(BaseTTSEngine):
    """MeloTTS: preset-voice, multilingual, multi-speaker TTS."""

    name = "melotts"
    supports_cloning = False
    #: Each preset selects a different per-language model
    #: (``melo.api.TTS(language=...)``); no single fixed repo id is resolved
    #: in this code, so none is declared here.
    model_ids: tuple[str, ...] = ()

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Synthesize ``text`` with MeloTTS's preset ``voice.preset`` speaker.

        Loads the language model implied by the preset, synthesizes raw audio
        at MeloTTS's native 44.1 kHz, and resamples down to the project's
        16 kHz mono float32 convention. Raises ``ValueError`` up front if the
        preset isn't one of the shipped speaker ids, before any heavy import.
        """
        language = self._known_preset(voice.preset)
        from melo.api import TTS  # type: ignore

        model = TTS(language=language, device="cpu")
        speaker_id = model.hps.data.spk2id[voice.preset]
        raw = model.tts_to_file(text, speaker_id, output_path=None, quiet=True)
        return self._to_16k(raw)

    @staticmethod
    def _known_preset(preset: str | None) -> str:
        """Validate ``preset`` against MeloTTS's speaker bank, or raise."""
        language = _PRESET_LANGUAGES.get(preset) if preset is not None else None
        if language is None:
            raise ValueError(
                f"melotts has no preset voice {preset!r}; expected one of "
                f"{sorted(_PRESET_LANGUAGES)}"
            )
        return language

    @staticmethod
    def _to_16k(raw: np.ndarray) -> AudioBuffer:
        """Downmix/resample MeloTTS's raw waveform to the project's 16 kHz mono.

        Uses plain numpy linear interpolation rather than scipy so this
        helper (and its fast test) needs no extra dependency.
        """
        samples = np.asarray(raw, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=-1).astype(np.float32)
        resampled = _linear_resample(samples, _MELOTTS_NATIVE_SAMPLE_RATE, SAMPLE_RATE)
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
