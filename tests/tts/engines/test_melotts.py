"""Tests for the MeloTTS engine: fast contract checks plus a slow real-
synthesis smoke test. The fast tests never call the real ``melo`` package
and need no model weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from satasr.core.interfaces import VoiceReference
from satasr.tts.engines.melotts import MeloTTSEngine
from satasr.tts.registry import TTS_ENGINES


def test_melotts_is_registered() -> None:
    assert "melotts" in TTS_ENGINES.available()


def test_melotts_metadata() -> None:
    engine = TTS_ENGINES.create("melotts")
    assert engine.name == "melotts"
    assert engine.supports_cloning is False


def test_melotts_requires_preset_not_reference_audio() -> None:
    engine = MeloTTSEngine()
    with pytest.raises(ValueError, match="preset"):
        engine.synthesize("hello", VoiceReference("A"))


def test_melotts_rejects_empty_text() -> None:
    engine = MeloTTSEngine()
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ", VoiceReference("A", preset="EN-US"))


def test_melotts_unknown_preset_raises_clearly() -> None:
    # Validation happens before the heavy `melo` import, so this stays fast
    # and needs no installed package or weights.
    engine = MeloTTSEngine()
    with pytest.raises(ValueError, match="no preset voice"):
        engine.synthesize("hello", VoiceReference("A", preset="not_a_real_voice"))


def test_melotts_to_16k_downmixes_and_resamples() -> None:
    # Exercise the pure-numpy resampling helper directly, bypassing the heavy
    # melo/torch import path inside _render.
    stereo_44100 = np.zeros((8820, 2), dtype=np.float32)
    buffer = MeloTTSEngine._to_16k(stereo_44100)
    assert buffer.sample_rate == 16_000
    assert buffer.samples.ndim == 1
    assert buffer.samples.dtype == np.float32


def test_melotts_known_preset_maps_to_language() -> None:
    assert MeloTTSEngine._known_preset("ZH") == "ZH"
    assert MeloTTSEngine._known_preset("EN-BR") == "EN"


@pytest.mark.slow
def test_melotts_real_synthesis() -> None:
    """Real synthesis: requires the ``melo`` package and downloaded weights."""
    engine = TTS_ENGINES.create("melotts")
    voice = VoiceReference("SpeakerA", preset="EN-US")
    clip = engine.synthesize("Hello, this is a test.", voice)
    assert clip.audio.sample_rate == 16_000
    assert clip.audio.duration_s > 0
