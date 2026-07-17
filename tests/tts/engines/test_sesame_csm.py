"""Tests for the Sesame CSM engine: fast contract checks plus a slow
real-synthesis smoke test. The fast tests never call ``_render`` and need no
model weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.tts.engines.sesame_csm import SesameCsmTTSEngine
from satasr.tts.registry import TTS_ENGINES


def _reference_voice(speaker_id: str = "A") -> VoiceReference:
    """A short synthetic reference clip, standing in for a recorded speaker."""
    reference = AudioBuffer(np.zeros(SAMPLE_RATE // 2, dtype=np.float32), SAMPLE_RATE)
    return VoiceReference(speaker_id, reference_audio=reference)


def test_sesame_csm_is_registered() -> None:
    assert "sesame_csm" in TTS_ENGINES.available()


def test_sesame_csm_declares_cloning_support() -> None:
    engine = TTS_ENGINES.create("sesame_csm")
    assert engine.name == "sesame_csm"
    assert engine.supports_cloning is True


def test_sesame_csm_rejects_preset_only_voice() -> None:
    """Cloning engines must reject a voice with no reference_audio (§4.4)."""
    engine = TTS_ENGINES.create("sesame_csm")
    with pytest.raises(ValueError, match="reference_audio"):
        engine.synthesize("hello", VoiceReference("A", preset="alice"))


def test_sesame_csm_rejects_empty_text() -> None:
    engine = TTS_ENGINES.create("sesame_csm")
    with pytest.raises(ValueError, match="empty"):
        engine.synthesize("   ", _reference_voice())


def test_sesame_csm_resample_is_identity_at_matching_rate() -> None:
    # Exercise the pure-numpy resampling helper directly, bypassing the heavy
    # csm/torch import path inside _render.
    samples = np.zeros(100, dtype=np.float32)
    result = SesameCsmTTSEngine._resample(samples, SAMPLE_RATE, SAMPLE_RATE)
    assert np.array_equal(result, samples)


def test_sesame_csm_to_16k_downmixes_and_resamples() -> None:
    stereo_24k = np.zeros((4800, 2), dtype=np.float32)
    buffer = SesameCsmTTSEngine._to_16k(stereo_24k, 24_000)
    assert buffer.sample_rate == 16_000
    assert buffer.samples.ndim == 1
    assert buffer.samples.dtype == np.float32


@pytest.mark.slow
def test_sesame_csm_real_synthesis() -> None:
    """Real synthesis: requires the ``csm`` package, torch, and a GPU."""
    engine = TTS_ENGINES.create("sesame_csm")
    voice = _reference_voice("SpeakerA")
    clip = engine.synthesize("Hello, this is a test.", voice)
    assert clip.audio.sample_rate == SAMPLE_RATE
    assert clip.audio.samples.dtype == np.float32
    assert clip.audio.samples.ndim == 1
    assert clip.audio.num_samples > 0
