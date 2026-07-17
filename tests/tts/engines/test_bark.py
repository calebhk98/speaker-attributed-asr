"""Tests for the Bark engine: fast contract checks plus a slow real-synthesis
smoke test. The fast tests never call ``_render`` and need no model weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from satasr.core.interfaces import VoiceReference
from satasr.tts.engines.bark import BarkTTSEngine
from satasr.tts.registry import TTS_ENGINES


def test_bark_is_registered() -> None:
    assert "bark" in TTS_ENGINES.available()


def test_bark_metadata() -> None:
    engine = TTS_ENGINES.create("bark")
    assert engine.name == "bark"
    assert engine.supports_cloning is False


def test_bark_requires_preset_not_reference_audio() -> None:
    engine = BarkTTSEngine()
    with pytest.raises(ValueError, match="preset"):
        engine.synthesize("hello", VoiceReference("A"))


def test_bark_rejects_empty_text() -> None:
    engine = BarkTTSEngine()
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ", VoiceReference("A", preset="v2/en_speaker_6"))


def test_bark_to_16k_downmixes_and_resamples() -> None:
    # Exercise the pure-numpy resampling helper directly, bypassing the heavy
    # bark/torch import path inside _render.
    stereo_24k = np.zeros((4800, 2), dtype=np.float32)
    buffer = BarkTTSEngine._to_16k(stereo_24k)
    assert buffer.sample_rate == 16_000
    assert buffer.samples.ndim == 1
    assert buffer.samples.dtype == np.float32


@pytest.mark.slow
def test_bark_real_synthesis() -> None:
    """Real synthesis: requires the ``bark`` package and downloaded weights."""
    engine = TTS_ENGINES.create("bark")
    voice = VoiceReference("SpeakerA", preset="v2/en_speaker_6")
    clip = engine.synthesize("Hello, this is a test.", voice)
    assert clip.audio.sample_rate == 16_000
    assert clip.audio.duration_s > 0
