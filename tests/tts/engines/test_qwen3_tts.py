"""Tests for the Qwen3-TTS engine: fast contract checks plus a slow real-
synthesis smoke test. The fast tests never call ``_render`` and need no model
weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from satasr.core.audio import AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.engines.qwen3_tts import Qwen3TTSEngine
from satasr.tts.registry import TTS_ENGINES


def test_qwen3_tts_is_registered() -> None:
    assert "qwen3_tts" in TTS_ENGINES.available()


def test_qwen3_tts_metadata() -> None:
    engine = TTS_ENGINES.create("qwen3_tts")
    assert engine.name == "qwen3_tts"
    assert engine.supports_cloning is True


def test_qwen3_tts_requires_reference_audio_not_preset() -> None:
    engine = Qwen3TTSEngine()
    with pytest.raises(ValueError, match="reference_audio"):
        engine.synthesize("hello", VoiceReference("A", preset="some_preset"))


def test_qwen3_tts_rejects_empty_text() -> None:
    engine = Qwen3TTSEngine()
    reference = AudioBuffer(np.zeros(1600, dtype=np.float32))
    voice = VoiceReference("A", reference_audio=reference)
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ", voice)


def test_qwen3_tts_to_16k_downmixes_and_resamples() -> None:
    # Exercise the pure-numpy resampling helper directly, bypassing the heavy
    # qwen_tts/torch import path inside _render.
    stereo_24k = np.zeros((2400, 2), dtype=np.float32)
    buffer = Qwen3TTSEngine._to_16k(stereo_24k, native_rate=24_000)
    assert buffer.sample_rate == 16_000
    assert buffer.samples.ndim == 1
    assert buffer.samples.dtype == np.float32


@pytest.mark.slow
def test_qwen3_tts_real_synthesis() -> None:
    """Real synthesis: requires the ``qwen_tts`` package and downloaded weights."""
    engine = TTS_ENGINES.create("qwen3_tts")
    reference = AudioBuffer(np.zeros(16_000, dtype=np.float32))
    voice = VoiceReference("SpeakerA", reference_audio=reference)
    clip = engine.synthesize("Hello, this is a test.", voice)
    assert clip.audio.sample_rate == 16_000
    assert clip.audio.duration_s > 0
