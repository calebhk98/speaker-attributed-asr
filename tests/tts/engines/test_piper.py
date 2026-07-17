"""Tests for the Piper engine: fast contract checks plus a slow real-synthesis
smoke test. The fast tests never call ``_render`` and need no model weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from satasr.core.interfaces import VoiceReference
from satasr.tts.engines.piper import PiperTTSEngine
from satasr.tts.registry import TTS_ENGINES


def test_piper_is_registered() -> None:
    assert "piper" in TTS_ENGINES.available()


def test_piper_metadata() -> None:
    engine = TTS_ENGINES.create("piper")
    assert engine.name == "piper"
    assert engine.supports_cloning is False


def test_piper_requires_preset_not_reference_audio() -> None:
    engine = PiperTTSEngine()
    with pytest.raises(ValueError, match="preset"):
        engine.synthesize("hello", VoiceReference("A"))


def test_piper_rejects_empty_text() -> None:
    engine = PiperTTSEngine()
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ", VoiceReference("A", preset="en_US-lessac-medium"))


def test_piper_to_16k_downmixes_and_resamples() -> None:
    # Exercise the pure-numpy PCM-decode + resample helper directly, bypassing
    # the heavy piper/onnxruntime import path inside _render.
    silence_22050 = np.zeros(2205, dtype=np.int16).tobytes()
    buffer = PiperTTSEngine._to_16k([silence_22050], 22_050)
    assert buffer.sample_rate == 16_000
    assert buffer.samples.ndim == 1
    assert buffer.samples.dtype == np.float32


def test_piper_to_16k_is_noop_at_native_rate() -> None:
    # When the voice's native rate already matches 16 kHz, no resampling
    # arithmetic should change the sample count.
    silence_16k = np.zeros(1600, dtype=np.int16).tobytes()
    buffer = PiperTTSEngine._to_16k([silence_16k], 16_000)
    assert buffer.samples.shape[0] == 1600


@pytest.mark.slow
def test_piper_real_synthesis() -> None:
    """Real synthesis: requires the ``piper-tts`` package and voice weights."""
    engine = TTS_ENGINES.create("piper")
    voice = VoiceReference("SpeakerA", preset="en_US-lessac-medium")
    clip = engine.synthesize("Hello, this is a test.", voice)
    assert clip.audio.sample_rate == 16_000
    assert clip.audio.duration_s > 0
