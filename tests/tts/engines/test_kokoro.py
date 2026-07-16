"""Tests for the Kokoro engine: fast contract checks plus a slow real-synthesis
smoke test. The fast tests never call the real ``kokoro`` package and need no
model weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from satasr.core.interfaces import VoiceReference
from satasr.tts.engines.kokoro import KokoroTTSEngine
from satasr.tts.registry import TTS_ENGINES


def test_kokoro_is_registered() -> None:
    assert "kokoro" in TTS_ENGINES.available()


def test_kokoro_metadata() -> None:
    engine = TTS_ENGINES.create("kokoro")
    assert engine.name == "kokoro"
    assert engine.supports_cloning is False


def test_kokoro_requires_preset_not_reference_audio() -> None:
    engine = KokoroTTSEngine()
    with pytest.raises(ValueError, match="preset"):
        engine.synthesize("hello", VoiceReference("A"))


def test_kokoro_rejects_empty_text() -> None:
    engine = KokoroTTSEngine()
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ", VoiceReference("A", preset="af_heart"))


def test_kokoro_unknown_preset_raises_clearly() -> None:
    # Validation happens before the heavy `kokoro` import, so this stays fast
    # and needs no installed package or weights.
    engine = KokoroTTSEngine()
    with pytest.raises(ValueError, match="no preset voice"):
        engine.synthesize("hello", VoiceReference("A", preset="not_a_real_voice"))


def test_kokoro_to_16k_downmixes_and_resamples() -> None:
    # Exercise the pure-numpy resampling helper directly, bypassing the heavy
    # kokoro/torch import path inside _render.
    stereo_24k = np.zeros((4800, 2), dtype=np.float32)
    buffer = KokoroTTSEngine._to_16k(stereo_24k)
    assert buffer.sample_rate == 16_000
    assert buffer.samples.ndim == 1
    assert buffer.samples.dtype == np.float32


@pytest.mark.slow
def test_kokoro_real_synthesis() -> None:
    """Real synthesis: requires the ``kokoro`` package and downloaded weights."""
    engine = TTS_ENGINES.create("kokoro")
    voice = VoiceReference("SpeakerA", preset="af_heart")
    clip = engine.synthesize("Hello, this is a test.", voice)
    assert clip.audio.sample_rate == 16_000
    assert clip.audio.duration_s > 0
