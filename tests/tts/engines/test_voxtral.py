"""Tests for the Voxtral (Mistral) preset-voice engine and its registration.

The fast tests never call ``_render`` (which needs a locally running
vLLM-Omni server plus the ``openai`` client package, neither available
here) and instead exercise the pure-stdlib/numpy WAV-decode + resample
helper directly.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from satasr.core.interfaces import VoiceReference
from satasr.tts.engines.voxtral import VoxtralTTSEngine
from satasr.tts.registry import TTS_ENGINES


def _silent_wav_bytes(num_frames: int, sample_rate: int) -> bytes:
    """Build a minimal mono 16-bit PCM WAV file of silence for decode tests."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(np.zeros(num_frames, dtype=np.int16).tobytes())
    return buffer.getvalue()


def test_voxtral_is_registered() -> None:
    assert "voxtral" in TTS_ENGINES.available()


def test_voxtral_metadata() -> None:
    engine = TTS_ENGINES.create("voxtral")
    assert engine.name == "voxtral"
    assert engine.supports_cloning is False


def test_voxtral_requires_preset_not_reference_audio() -> None:
    engine = VoxtralTTSEngine()
    with pytest.raises(ValueError, match="preset"):
        engine.synthesize("hello", VoiceReference("A"))


def test_voxtral_rejects_empty_text() -> None:
    engine = VoxtralTTSEngine()
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ", VoiceReference("A", preset="casual_male"))


def test_voxtral_to_16k_downmixes_and_resamples() -> None:
    # 24 kHz stand-in for Voxtral's WAV response, decoded with no network call.
    wav_bytes = _silent_wav_bytes(2400, 24_000)
    buffer = VoxtralTTSEngine._to_16k(wav_bytes)
    assert buffer.sample_rate == 16_000
    assert buffer.samples.ndim == 1
    assert buffer.samples.dtype == np.float32


def test_voxtral_to_16k_is_noop_at_native_rate() -> None:
    # When the response is already 16 kHz, resampling should be a no-op.
    wav_bytes = _silent_wav_bytes(1600, 16_000)
    buffer = VoxtralTTSEngine._to_16k(wav_bytes)
    assert buffer.samples.shape[0] == 1600


@pytest.mark.slow
def test_voxtral_real_synthesis() -> None:
    """Real synthesis: requires a running vLLM-Omni server hosting
    mistralai/Voxtral-4B-TTS-2603 plus the ``openai`` client package —
    neither is available in this environment.
    """
    engine = TTS_ENGINES.create("voxtral")
    voice = VoiceReference("SpeakerA", preset="casual_male")
    clip = engine.synthesize("Hello, this is a test.", voice)
    assert clip.audio.sample_rate == 16_000
    assert clip.audio.duration_s > 0
