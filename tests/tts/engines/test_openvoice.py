"""Tests for the OpenVoice (MyShell) voice-conversion engine and its registration."""

from __future__ import annotations

import numpy as np
import pytest

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.registry import TTS_ENGINES


def _reference_voice(speaker_id: str = "A") -> VoiceReference:
    """A short synthetic reference clip, standing in for a recorded speaker."""
    reference = AudioBuffer(np.zeros(SAMPLE_RATE // 2, dtype=np.float32), SAMPLE_RATE)
    return VoiceReference(speaker_id, reference_audio=reference)


def test_openvoice_is_registered() -> None:
    assert "openvoice" in TTS_ENGINES.available()


def test_openvoice_declares_cloning_support() -> None:
    engine = TTS_ENGINES.create("openvoice")
    assert engine.name == "openvoice"
    assert engine.supports_cloning is True


def test_openvoice_rejects_preset_only_voice() -> None:
    """Cloning engines must reject a voice with no reference_audio (§4.4)."""
    engine = TTS_ENGINES.create("openvoice")
    with pytest.raises(ValueError, match="reference_audio"):
        engine.synthesize("hello", VoiceReference("A", preset="alice"))


def test_openvoice_rejects_empty_text() -> None:
    engine = TTS_ENGINES.create("openvoice")
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ", _reference_voice())


@pytest.mark.slow
def test_openvoice_real_synthesis_produces_16k_mono_audio() -> None:
    """Real-weights smoke test: needs the ``melo``/``openvoice`` packages,
    torch, and their GPU checkpoints, none of which are available in this
    environment, so this is expected to be exercised in CI / on a GPU box,
    not here.
    """
    engine = TTS_ENGINES.create("openvoice")
    clip = engine.synthesize("This is a voice-converted utterance.", _reference_voice())
    assert clip.audio.sample_rate == SAMPLE_RATE
    assert clip.audio.samples.dtype == np.float32
    assert clip.audio.samples.ndim == 1
    assert clip.audio.num_samples > 0
