"""Amplitude gain — the simplest acoustic transform (design §4.9).

Useful on its own (loudness variety) and as a building block that
:class:`~satasr.augment.chain.AugmentChain` composes with other augmenters.
"""

from __future__ import annotations

import numpy as np

from satasr.augment.registry import AUGMENTERS
from satasr.core.audio import AudioBuffer

_MIN_SAMPLE = -1.0
_MAX_SAMPLE = 1.0


@AUGMENTERS.register("gain")
class GainAugmenter:
    """Scale amplitude by a fixed gain in dB, clipping to the valid range."""

    def __init__(self, gain_db: float) -> None:
        self._gain_db = gain_db

    def apply(self, audio: AudioBuffer) -> AudioBuffer:
        """Return ``audio`` scaled by this augmenter's gain, clipped to [-1, 1]."""
        factor = np.float32(10.0 ** (self._gain_db / 20.0))
        scaled = audio.samples * factor
        clipped = np.clip(scaled, _MIN_SAMPLE, _MAX_SAMPLE).astype(np.float32)
        return AudioBuffer(clipped, audio.sample_rate)
