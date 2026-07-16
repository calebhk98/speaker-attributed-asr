"""Tests for AugmentChain: ordered composition of augmenters."""

from __future__ import annotations

import numpy as np
import pytest

from satasr.augment.chain import AugmentChain
from satasr.augment.gain import GainAugmenter
from satasr.core.audio import SAMPLE_RATE, AudioBuffer


def _tone(amplitude: float = 0.2, duration_s: float = 0.1) -> AudioBuffer:
    times = np.arange(round(SAMPLE_RATE * duration_s), dtype=np.float32) / SAMPLE_RATE
    wave = amplitude * np.sin(2.0 * np.pi * 220.0 * times)
    return AudioBuffer(wave.astype(np.float32), SAMPLE_RATE)


def test_two_gains_compose_to_the_summed_db_when_unclipped() -> None:
    audio = _tone()
    chained = AugmentChain([GainAugmenter(3.0), GainAugmenter(4.0)]).apply(audio)
    direct = GainAugmenter(7.0).apply(audio)
    assert chained.samples == pytest.approx(direct.samples, abs=1e-5)


def test_empty_chain_is_identity() -> None:
    audio = _tone()
    result = AugmentChain([]).apply(audio)
    assert np.array_equal(result.samples, audio.samples)


def test_order_matters_when_an_earlier_stage_clips() -> None:
    audio = _tone(amplitude=0.5)
    # Amplify enough to clip, then attenuate: the clipping is irreversible.
    clip_then_attenuate = AugmentChain(
        [GainAugmenter(40.0), GainAugmenter(-40.0)]
    ).apply(audio)
    # Attenuate first (no-op here since attenuating never clips), then amplify
    # back: nothing was lost, so this differs from the other ordering.
    attenuate_then_amplify = AugmentChain(
        [GainAugmenter(-40.0), GainAugmenter(40.0)]
    ).apply(audio)
    assert not np.array_equal(
        clip_then_attenuate.samples, attenuate_then_amplify.samples
    )


def test_output_is_float32_mono_in_range() -> None:
    audio = _tone()
    result = AugmentChain([GainAugmenter(10.0), GainAugmenter(-2.0)]).apply(audio)
    assert result.samples.dtype == np.float32
    assert result.samples.ndim == 1
    assert result.samples.max() <= 1.0
    assert result.samples.min() >= -1.0
