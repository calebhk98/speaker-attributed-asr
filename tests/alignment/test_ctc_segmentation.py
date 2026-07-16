"""Tests for the CTC-segmentation aligner stub."""

from __future__ import annotations

import numpy as np
import pytest

from satasr.alignment.registry import ALIGNERS
from satasr.core.audio import SAMPLE_RATE, AudioBuffer


def test_ctc_segmentation_is_registered() -> None:
    assert "ctc_segmentation" in ALIGNERS.available()


def test_align_raises_not_implemented() -> None:
    aligner = ALIGNERS.create("ctc_segmentation")
    audio = AudioBuffer(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    with pytest.raises(NotImplementedError, match="CtcSegmentationAligner"):
        aligner.align(audio, "hello world")
