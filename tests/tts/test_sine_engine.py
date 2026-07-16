"""Tests for the reference sine engine and its registration."""

from __future__ import annotations

import numpy as np

from satasr.core.interfaces import VoiceReference
from satasr.tts.registry import TTS_ENGINES


def test_sine_is_registered() -> None:
    assert "sine" in TTS_ENGINES.available()


def test_synthesis_is_deterministic() -> None:
    engine = TTS_ENGINES.create("sine")
    voice = VoiceReference("A", preset="alice")
    first = engine.synthesize("hello world", voice)
    second = engine.synthesize("hello world", voice)
    assert np.array_equal(first.audio.samples, second.audio.samples)


def test_longer_text_makes_longer_audio() -> None:
    engine = TTS_ENGINES.create("sine")
    voice = VoiceReference("A", preset="alice")
    short = engine.synthesize("hi", voice)
    long = engine.synthesize("hi there, this is a much longer sentence", voice)
    assert long.audio.duration_s > short.audio.duration_s


def test_different_presets_produce_different_audio() -> None:
    engine = TTS_ENGINES.create("sine")
    alice = engine.synthesize("hello", VoiceReference("A", preset="alice"))
    bob = engine.synthesize("hello", VoiceReference("B", preset="bob"))
    assert not np.array_equal(alice.audio.samples, bob.audio.samples)
