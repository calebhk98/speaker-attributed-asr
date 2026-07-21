"""Tests for the Step-Audio-EditX engine: fast contract checks plus a slow
real-synthesis smoke test. The fast tests never call ``_render`` and need no
model weights, GPU, or the heavy ``torch`` / Step-Audio inference packages.
"""

from __future__ import annotations

import numpy as np
import pytest

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.engines.step_audio_editx import StepAudioEditXTTSEngine
from satasr.tts.registry import TTS_ENGINES


def _reference_voice(speaker_id: str = "A") -> VoiceReference:
    """A short synthetic reference clip, standing in for a recorded speaker."""
    reference = AudioBuffer(np.zeros(SAMPLE_RATE // 2, dtype=np.float32), SAMPLE_RATE)
    return VoiceReference(speaker_id, reference_audio=reference)


def test_step_audio_editx_is_registered() -> None:
    assert "step_audio_editx" in TTS_ENGINES.available()


def test_step_audio_editx_metadata() -> None:
    engine = TTS_ENGINES.create("step_audio_editx")
    assert engine.name == "step_audio_editx"
    assert engine.supports_cloning is True


def test_step_audio_editx_rejects_preset_only_voice() -> None:
    """Cloning engines must reject a voice with no reference_audio (§4.4)."""
    engine = StepAudioEditXTTSEngine()
    with pytest.raises(ValueError, match="reference_audio"):
        engine.synthesize("hello", VoiceReference("A", preset="alice"))


def test_step_audio_editx_rejects_empty_text() -> None:
    engine = StepAudioEditXTTSEngine()
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ", _reference_voice())


def test_step_audio_editx_to_16k_downmixes_and_resamples() -> None:
    # Exercise the pure-numpy resampling helper directly, bypassing the heavy
    # torch import path inside _render. Step-Audio-EditX outputs 24 kHz audio.
    stereo_24k = np.zeros((4800, 2), dtype=np.float32)
    buffer = StepAudioEditXTTSEngine._to_16k(stereo_24k, 24_000)
    assert buffer.sample_rate == 16_000
    assert buffer.samples.ndim == 1
    assert buffer.samples.dtype == np.float32


@pytest.mark.slow
def test_step_audio_editx_real_synthesis() -> None:
    """Real synthesis: requires the Step-Audio-EditX weights and inference code."""
    engine = TTS_ENGINES.create("step_audio_editx")
    clip = engine.synthesize("Hello, this is a test.", _reference_voice())
    assert clip.audio.sample_rate == SAMPLE_RATE
    assert clip.audio.duration_s > 0
