"""Tests for the Dia engine: fast contract checks plus a slow real-synthesis
smoke test. The fast tests never call ``_render`` and need no model weights
or GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from satasr.core.interfaces import VoiceReference
from satasr.tts.engines.dia import DiaTTSEngine
from satasr.tts.registry import TTS_ENGINES


def test_dia_is_registered() -> None:
    assert "dia" in TTS_ENGINES.available()


def test_dia_metadata() -> None:
    engine = TTS_ENGINES.create("dia")
    assert engine.name == "dia"
    assert engine.supports_cloning is False


def test_dia_requires_preset_not_reference_audio() -> None:
    engine = DiaTTSEngine()
    with pytest.raises(ValueError, match="preset"):
        engine.synthesize("hello", VoiceReference("A"))


def test_dia_rejects_empty_text() -> None:
    engine = DiaTTSEngine()
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ", VoiceReference("A", preset="S1"))


def test_dia_build_prompt_wraps_bare_preset_in_speaker_tag() -> None:
    voice = VoiceReference("A", preset="S2")
    prompt = DiaTTSEngine._build_prompt("hello there", voice)
    assert prompt == "[S2] hello there"


def test_dia_build_prompt_accepts_already_bracketed_preset() -> None:
    voice = VoiceReference("A", preset="[S3]")
    prompt = DiaTTSEngine._build_prompt("hi", voice)
    assert prompt == "[S3] hi"


def test_dia_build_prompt_defaults_when_preset_is_falsy_string() -> None:
    # BaseTTSEngine only checks `preset is None`, so an empty string reaches
    # _build_prompt; it should fall back to the default speaker tag.
    voice = VoiceReference("A", preset="")
    prompt = DiaTTSEngine._build_prompt("hi", voice)
    assert prompt == "[S1] hi"


def test_dia_to_16k_downmixes_and_resamples() -> None:
    # Exercise the pure-numpy resampling helper directly, bypassing the heavy
    # dia/torch import path inside _render.
    stereo_44k = np.zeros((4410, 2), dtype=np.float32)
    buffer = DiaTTSEngine._to_16k(stereo_44k)
    assert buffer.sample_rate == 16_000
    assert buffer.samples.ndim == 1
    assert buffer.samples.dtype == np.float32


@pytest.mark.slow
def test_dia_real_synthesis() -> None:
    """Real synthesis: requires the ``dia`` package and downloaded weights."""
    engine = TTS_ENGINES.create("dia")
    voice = VoiceReference("SpeakerA", preset="S1")
    clip = engine.synthesize("Hello, this is a test.", voice)
    assert clip.audio.sample_rate == 16_000
    assert clip.audio.duration_s > 0
