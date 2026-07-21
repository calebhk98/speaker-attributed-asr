"""Tests for the Chatterbox-Turbo engine: fast contract checks plus a slow
real-synthesis smoke test. The fast tests never call ``_render`` and need no
model weights, GPU, or optional audio libraries (torch, chatterbox-tts,
scipy, soundfile).
"""

from __future__ import annotations

import numpy as np
import pytest

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.engines.chatterbox_turbo import ChatterboxTurboTTSEngine
from satasr.tts.registry import TTS_ENGINES


def _reference_voice(speaker_id: str = "A") -> VoiceReference:
    """A short synthetic reference clip, standing in for a recorded speaker."""
    reference = AudioBuffer(np.zeros(SAMPLE_RATE // 2, dtype=np.float32), SAMPLE_RATE)
    return VoiceReference(speaker_id, reference_audio=reference)


def test_chatterbox_turbo_is_registered() -> None:
    assert "chatterbox_turbo" in TTS_ENGINES.available()


def test_chatterbox_turbo_is_distinct_from_base_chatterbox() -> None:
    # The base engine must remain registered alongside the new turbo one.
    assert "chatterbox" in TTS_ENGINES.available()
    assert "chatterbox_turbo" in TTS_ENGINES.available()


def test_chatterbox_turbo_metadata() -> None:
    engine = TTS_ENGINES.create("chatterbox_turbo")
    assert engine.name == "chatterbox_turbo"
    assert engine.supports_cloning is True


def test_chatterbox_turbo_rejects_preset_only_voice() -> None:
    """Cloning engines must reject a voice with no reference_audio (§4.4)."""
    engine = ChatterboxTurboTTSEngine()
    with pytest.raises(ValueError, match="reference_audio"):
        engine.synthesize("hello", VoiceReference("A", preset="alice"))


def test_chatterbox_turbo_rejects_empty_text() -> None:
    engine = ChatterboxTurboTTSEngine()
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ", _reference_voice())


@pytest.mark.slow
def test_chatterbox_turbo_real_synthesis_produces_16k_mono_audio() -> None:
    """Real-weights smoke test: needs the ``chatterbox-tts`` package, torch,
    and a GPU (or a slow CPU fallback) that are not available in this
    environment, so this is expected to be exercised in CI / on a GPU box,
    not here.
    """
    engine = TTS_ENGINES.create("chatterbox_turbo")
    clip = engine.synthesize("This is a cloned voice speaking.", _reference_voice())
    assert clip.audio.sample_rate == SAMPLE_RATE
    assert clip.audio.samples.dtype == np.float32
    assert clip.audio.samples.ndim == 1
    assert clip.audio.num_samples > 0
