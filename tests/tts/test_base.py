"""Tests for the shared BaseTTSEngine voice contract."""

from __future__ import annotations

import numpy as np
import pytest

from satasr.core.audio import AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.base import BaseTTSEngine


class _CloningEngine(BaseTTSEngine):
    name = "fake_clone"
    supports_cloning = True

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        return AudioBuffer(np.zeros(100, dtype=np.float32))


class _PresetEngine(BaseTTSEngine):
    name = "fake_preset"
    supports_cloning = False

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        return AudioBuffer(np.zeros(100, dtype=np.float32))


def test_empty_text_is_rejected() -> None:
    engine = _PresetEngine()
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ", VoiceReference("A", preset="x"))


def test_cloning_engine_requires_reference_audio() -> None:
    engine = _CloningEngine()
    with pytest.raises(ValueError, match="reference_audio"):
        engine.synthesize("hi", VoiceReference("A", preset="x"))


def test_preset_engine_requires_preset() -> None:
    engine = _PresetEngine()
    with pytest.raises(ValueError, match="preset"):
        engine.synthesize("hi", VoiceReference("A"))


def test_synthesize_packages_clip_with_metadata() -> None:
    engine = _PresetEngine()
    clip = engine.synthesize("hello", VoiceReference("SpeakerA", preset="x"))
    assert clip.speaker_id == "SpeakerA"
    assert clip.text == "hello"
