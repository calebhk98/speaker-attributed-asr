"""Tests for the Fish Speech engine: fast contract checks plus a slow
real-synthesis smoke test. The fast tests never call ``_render`` and need no
model weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from satasr.core.audio import AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.engines.fish_speech import FishSpeechTTSEngine
from satasr.tts.registry import TTS_ENGINES


def test_fish_speech_is_registered() -> None:
    assert "fish_speech" in TTS_ENGINES.available()


def test_fish_speech_metadata() -> None:
    engine = TTS_ENGINES.create("fish_speech")
    assert engine.name == "fish_speech"
    assert engine.supports_cloning is True


def test_fish_speech_requires_reference_audio_not_preset() -> None:
    engine = FishSpeechTTSEngine()
    with pytest.raises(ValueError, match="reference_audio"):
        engine.synthesize("hello", VoiceReference("A", preset="alice"))


def test_fish_speech_rejects_empty_text() -> None:
    engine = FishSpeechTTSEngine()
    reference = AudioBuffer(np.zeros(1600, dtype=np.float32))
    voice = VoiceReference("A", reference_audio=reference)
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ", voice)


def test_fish_speech_to_16k_downmixes_and_resamples() -> None:
    # Exercise the pure-numpy resampling helper directly, bypassing the heavy
    # fish_speech/torch import path inside _render.
    stereo_44k1 = np.zeros((8820, 2), dtype=np.float32)
    buffer = FishSpeechTTSEngine._to_16k(stereo_44k1)
    assert buffer.sample_rate == 16_000
    assert buffer.samples.ndim == 1
    assert buffer.samples.dtype == np.float32


def test_fish_speech_require_reference_audio_helper() -> None:
    reference = AudioBuffer(np.zeros(100, dtype=np.float32))
    voice = VoiceReference("A", reference_audio=reference)
    assert FishSpeechTTSEngine._require_reference_audio(voice) is reference


@pytest.mark.slow
def test_fish_speech_real_synthesis() -> None:
    """Real synthesis: requires the ``fish-speech`` package and weights."""
    engine = TTS_ENGINES.create("fish_speech")
    reference = AudioBuffer(np.zeros(16_000, dtype=np.float32))
    voice = VoiceReference("SpeakerA", reference_audio=reference)
    clip = engine.synthesize("Hello, this is a test.", voice)
    assert clip.audio.sample_rate == 16_000
    assert clip.audio.duration_s > 0
