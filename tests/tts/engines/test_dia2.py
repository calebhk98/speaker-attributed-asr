"""Tests for the Dia2 engine: fast contract checks plus a slow real-synthesis
smoke test. The fast tests never call ``_render`` and need no model weights
or GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from satasr.core.interfaces import VoiceReference
from satasr.tts.engines.dia2 import Dia2TTSEngine
from satasr.tts.registry import TTS_ENGINES


def test_dia2_is_registered() -> None:
    assert "dia2" in TTS_ENGINES.available()


def test_dia2_is_distinct_from_dia1() -> None:
    # Dia (v1) must remain registered alongside the new Dia2 engine.
    assert "dia" in TTS_ENGINES.available()
    assert "dia2" in TTS_ENGINES.available()


def test_dia2_metadata() -> None:
    engine = TTS_ENGINES.create("dia2")
    assert engine.name == "dia2"
    assert engine.supports_cloning is False


def test_dia2_requires_preset_not_reference_audio() -> None:
    engine = Dia2TTSEngine()
    with pytest.raises(ValueError, match="preset"):
        engine.synthesize("hello", VoiceReference("A"))


def test_dia2_rejects_empty_text() -> None:
    engine = Dia2TTSEngine()
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ", VoiceReference("A", preset="S1"))


def test_dia2_build_prompt_wraps_bare_preset_in_speaker_tag() -> None:
    voice = VoiceReference("A", preset="S2")
    prompt = Dia2TTSEngine._build_prompt("hello there", voice)
    assert prompt == "[S2] hello there"


def test_dia2_build_prompt_accepts_already_bracketed_preset() -> None:
    voice = VoiceReference("A", preset="[S3]")
    prompt = Dia2TTSEngine._build_prompt("hi", voice)
    assert prompt == "[S3] hi"


def test_dia2_build_prompt_defaults_when_preset_is_falsy_string() -> None:
    # BaseTTSEngine only checks `preset is None`, so an empty string reaches
    # _build_prompt; it should fall back to the default speaker tag.
    voice = VoiceReference("A", preset="")
    prompt = Dia2TTSEngine._build_prompt("hi", voice)
    assert prompt == "[S1] hi"


def test_dia2_to_16k_downmixes_and_resamples() -> None:
    # Exercise the pure-numpy resampling helper directly, bypassing the heavy
    # dia2/torch import path inside _render. Dia2's Mimi codec runs at 24 kHz.
    stereo_24k = np.zeros((4800, 2), dtype=np.float32)
    buffer = Dia2TTSEngine._to_16k(stereo_24k)
    assert buffer.sample_rate == 16_000
    assert buffer.samples.ndim == 1
    assert buffer.samples.dtype == np.float32


@pytest.mark.slow
def test_dia2_real_synthesis() -> None:
    """Real synthesis: requires the ``dia2`` package and downloaded weights."""
    engine = TTS_ENGINES.create("dia2")
    voice = VoiceReference("SpeakerA", preset="S1")
    clip = engine.synthesize("Hello, this is a test.", voice)
    assert clip.audio.sample_rate == 16_000
    assert clip.audio.duration_s > 0
