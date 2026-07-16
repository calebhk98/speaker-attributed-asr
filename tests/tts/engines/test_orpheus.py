"""Tests for the Orpheus engine: fast contract checks plus a slow real-
synthesis smoke test. The fast tests never call ``_render`` and need no
model weights or GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.engines.orpheus import OrpheusTTSEngine
from satasr.tts.registry import TTS_ENGINES


def _reference_clip() -> AudioBuffer:
    """A tiny silent clip standing in for a real reference recording."""
    return AudioBuffer(np.zeros(SAMPLE_RATE // 10, dtype=np.float32), SAMPLE_RATE)


def test_orpheus_is_registered() -> None:
    assert "orpheus" in TTS_ENGINES.available()


def test_orpheus_metadata() -> None:
    engine = TTS_ENGINES.create("orpheus")
    assert engine.name == "orpheus"
    assert engine.supports_cloning is True


def test_orpheus_requires_reference_audio_not_preset() -> None:
    engine = OrpheusTTSEngine()
    with pytest.raises(ValueError, match="reference_audio"):
        engine.synthesize("hello", VoiceReference("A", preset="tara"))


def test_orpheus_rejects_empty_text() -> None:
    engine = OrpheusTTSEngine()
    voice = VoiceReference("A", reference_audio=_reference_clip())
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ", voice)


def test_orpheus_to_16k_downmixes_and_resamples() -> None:
    # Exercise the pure-numpy resampling helper directly, bypassing the heavy
    # orpheus_tts/vLLM import path inside _render.
    silent_24k_pcm16 = (np.zeros(2400, dtype=np.int16)).tobytes()
    buffer = OrpheusTTSEngine._to_16k([silent_24k_pcm16])
    assert buffer.sample_rate == 16_000
    assert buffer.samples.ndim == 1
    assert buffer.samples.dtype == np.float32


@pytest.mark.slow
def test_orpheus_real_synthesis() -> None:
    """Real synthesis: requires ``orpheus-speech`` weights and a GPU."""
    engine = TTS_ENGINES.create("orpheus")
    voice = VoiceReference("SpeakerA", reference_audio=_reference_clip())
    clip = engine.synthesize("Hello, this is a test.", voice)
    assert clip.audio.sample_rate == 16_000
    assert clip.audio.duration_s > 0
