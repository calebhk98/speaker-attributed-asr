"""Tests for the VITS engine: fast contract checks plus a slow real-synthesis
smoke test. The fast tests never call ``_render`` and need no model weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from satasr.core.interfaces import VoiceReference
from satasr.tts.engines.vits import VitsTTSEngine
from satasr.tts.registry import TTS_ENGINES


def test_vits_is_registered() -> None:
    assert "vits" in TTS_ENGINES.available()


def test_vits_metadata() -> None:
    engine = TTS_ENGINES.create("vits")
    assert engine.name == "vits"
    assert engine.supports_cloning is False


def test_vits_requires_preset_not_reference_audio() -> None:
    engine = VitsTTSEngine()
    with pytest.raises(ValueError, match="preset"):
        engine.synthesize("hello", VoiceReference("A"))


def test_vits_rejects_empty_text() -> None:
    engine = VitsTTSEngine()
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ", VoiceReference("A", preset="p225"))


def test_vits_to_16k_downmixes_and_resamples() -> None:
    # Exercise the pure-numpy resampling helper directly, bypassing the heavy
    # TTS/torch import path inside _render.
    stereo_22k = np.zeros((4410, 2), dtype=np.float32)
    buffer = VitsTTSEngine._to_16k(stereo_22k)
    assert buffer.sample_rate == 16_000
    assert buffer.samples.ndim == 1
    assert buffer.samples.dtype == np.float32


@pytest.mark.slow
def test_vits_real_synthesis() -> None:
    """Real synthesis: requires the ``TTS`` package and downloaded weights."""
    engine = TTS_ENGINES.create("vits")
    voice = VoiceReference("SpeakerA", preset="p225")
    clip = engine.synthesize("Hello, this is a test.", voice)
    assert clip.audio.sample_rate == 16_000
    assert clip.audio.duration_s > 0
