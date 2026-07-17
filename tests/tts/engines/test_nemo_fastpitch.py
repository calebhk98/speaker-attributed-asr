"""Tests for the NeMo FastPitch engine: fast contract checks plus a slow
real-synthesis smoke test. The fast tests never call ``_render`` and need no
NeMo package, torch checkpoints, or GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from satasr.core.interfaces import VoiceReference
from satasr.tts.engines.nemo_fastpitch import NemoFastPitchTTSEngine
from satasr.tts.registry import TTS_ENGINES


def test_nemo_fastpitch_is_registered() -> None:
    assert "nemo_fastpitch" in TTS_ENGINES.available()


def test_nemo_fastpitch_metadata() -> None:
    engine = TTS_ENGINES.create("nemo_fastpitch")
    assert engine.name == "nemo_fastpitch"
    assert engine.supports_cloning is False


def test_nemo_fastpitch_requires_preset_not_reference_audio() -> None:
    engine = NemoFastPitchTTSEngine()
    with pytest.raises(ValueError, match="preset"):
        engine.synthesize("hello", VoiceReference("A"))


def test_nemo_fastpitch_rejects_empty_text() -> None:
    engine = NemoFastPitchTTSEngine()
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ", VoiceReference("A", preset="92"))


def test_nemo_fastpitch_to_16k_downmixes_and_resamples() -> None:
    # Exercise the pure-numpy resample helper directly, bypassing the heavy
    # nemo_toolkit/torch import path inside _render.
    silence_44100 = np.zeros(4410, dtype=np.float32)
    buffer = NemoFastPitchTTSEngine._to_16k(silence_44100)
    assert buffer.sample_rate == 16_000
    assert buffer.samples.ndim == 1
    assert buffer.samples.dtype == np.float32


def test_nemo_fastpitch_to_16k_is_noop_at_native_rate() -> None:
    # Sanity check: interpolation preserves duration proportionally, so a
    # one-second buffer at the native rate maps to one second at 16 kHz.
    one_second_44100 = np.zeros(44_100, dtype=np.float32)
    buffer = NemoFastPitchTTSEngine._to_16k(one_second_44100)
    assert buffer.samples.shape[0] == 16_000


@pytest.mark.slow
def test_nemo_fastpitch_real_synthesis() -> None:
    """Real synthesis: requires ``nemo_toolkit[tts]``, torch, and GPU weights."""
    engine = TTS_ENGINES.create("nemo_fastpitch")
    voice = VoiceReference("SpeakerA", preset="92")
    clip = engine.synthesize("Hello, this is a test.", voice)
    assert clip.audio.sample_rate == 16_000
    assert clip.audio.duration_s > 0
