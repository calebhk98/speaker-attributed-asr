"""Tests for the Moshi engine: fast contract checks plus a slow real-synthesis
smoke test. The fast tests never call ``_render`` and need no model weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from satasr.core.interfaces import VoiceReference
from satasr.tts.engines.moshi import MoshiTTSEngine
from satasr.tts.registry import TTS_ENGINES


def test_moshi_is_registered() -> None:
    assert "moshi" in TTS_ENGINES.available()


def test_moshi_metadata() -> None:
    engine = TTS_ENGINES.create("moshi")
    assert engine.name == "moshi"
    assert engine.supports_cloning is False


def test_moshi_requires_preset_not_reference_audio() -> None:
    engine = MoshiTTSEngine()
    with pytest.raises(ValueError, match="preset"):
        engine.synthesize("hello", VoiceReference("A"))


def test_moshi_rejects_empty_text() -> None:
    engine = MoshiTTSEngine()
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ", VoiceReference("A", preset="moshika"))


def test_moshi_to_16k_downmixes_and_resamples() -> None:
    # Exercise the pure-numpy resampling helper directly, bypassing the heavy
    # moshi/torch import path inside _render.
    stereo_24k = np.zeros((2, 4800), dtype=np.float32)
    buffer = MoshiTTSEngine._to_16k(stereo_24k)
    assert buffer.sample_rate == 16_000
    assert buffer.samples.ndim == 1
    assert buffer.samples.dtype == np.float32


def test_moshi_to_16k_is_noop_at_native_rate() -> None:
    # Sanity check the helper's arithmetic without invoking any model code.
    mono_24k = np.zeros(2400, dtype=np.float32)
    buffer = MoshiTTSEngine._to_16k(mono_24k)
    assert buffer.samples.shape[0] == 1600


@pytest.mark.slow
def test_moshi_real_synthesis() -> None:
    """Real synthesis: requires the ``moshi`` package, torch, a GPU, and the
    multi-gigabyte Mimi codec + LM weights downloaded from Hugging Face.
    """
    engine = TTS_ENGINES.create("moshi")
    voice = VoiceReference("SpeakerA", preset="moshika")
    clip = engine.synthesize("Hello, this is a test.", voice)
    assert clip.audio.sample_rate == 16_000
    assert clip.audio.duration_s > 0
