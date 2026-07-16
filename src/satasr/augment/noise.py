"""Additive noise augmenters (design §4.9).

Reverb aside, varied noise is the other half of acoustic coverage: since there
is no single fixed target environment, we want a broad spread of SNRs rather
than one "realistic" level. :class:`WhiteNoiseAugmenter` is the dependency-free
work-horse (Gaussian noise scaled to a target SNR); :class:`MusanNoiseAugmenter`
is a stub for real, varied recorded noise (MUSAN) and is filled in separately.
"""

from __future__ import annotations

import random

import numpy as np
from numpy.typing import NDArray

from satasr.augment.registry import AUGMENTERS
from satasr.core.audio import AudioBuffer

_MIN_SAMPLE = -1.0
_MAX_SAMPLE = 1.0

_MUSAN_NOT_IMPLEMENTED = (
    "MusanNoiseAugmenter is a stub: loading and mixing real MUSAN noise clips "
    "is not implemented yet. Track/implement it via a follow-up GitHub issue "
    "on github.com/calebhk98/speaker-attributed-asr (design §4.9)."
)


def _rms(samples: NDArray[np.float32]) -> float:
    """Root-mean-square level of a signal; 0.0 for an empty or silent buffer."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


@AUGMENTERS.register("white_noise")
class WhiteNoiseAugmenter:
    """Add Gaussian noise scaled to hit a target SNR relative to the signal."""

    def __init__(self, snr_db: float, rng: random.Random) -> None:
        self._snr_db = snr_db
        self._rng = rng

    def apply(self, audio: AudioBuffer) -> AudioBuffer:
        """Return ``audio`` plus Gaussian noise at this augmenter's target SNR."""
        signal_rms = _rms(audio.samples)
        noise_rms = signal_rms / (10.0 ** (self._snr_db / 20.0))
        noise = self._draw_noise(audio.num_samples, noise_rms)
        mixed = audio.samples + noise
        clipped = np.clip(mixed, _MIN_SAMPLE, _MAX_SAMPLE).astype(np.float32)
        return AudioBuffer(clipped, audio.sample_rate)

    def _draw_noise(self, count: int, std: float) -> NDArray[np.float32]:
        """Sample ``count`` Gaussian values from this augmenter's own rng.

        Drawing straight from the injected ``random.Random`` (rather than
        seeding a separate numpy generator) keeps determinism obvious: the
        same seeded rng always yields the same noise, byte for byte.
        """
        values = [self._rng.gauss(0.0, std) for _ in range(count)]
        return np.array(values, dtype=np.float32)


@AUGMENTERS.register("musan")
class MusanNoiseAugmenter:
    """Stub for MUSAN-backed noise (real recorded noise, not synthetic).

    Intentionally does *not* touch disk or a corpus at import or construction
    time — only ``apply`` raises, once someone actually tries to use it.
    """

    def __init__(self, snr_db: float, corpus_dir: str | None = None) -> None:
        self._snr_db = snr_db
        self._corpus_dir = corpus_dir

    def apply(self, audio: AudioBuffer) -> AudioBuffer:
        raise NotImplementedError(_MUSAN_NOT_IMPLEMENTED)
