"""The one and only in-memory audio representation used across the project.

Every module — TTS engines, the mixer, augmenters, aligners — passes audio
around as an :class:`AudioBuffer`. Keeping a single value object here means
sample-rate handling and channel assumptions live in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Project-wide canonical sample rate. Whisper's front-end expects 16 kHz, so
# this is the single number the whole pipeline standardises on.
SAMPLE_RATE: int = 16_000


@dataclass(frozen=True)
class AudioBuffer:
    """Immutable mono audio: float32 samples in [-1, 1] plus a sample rate."""

    samples: NDArray[np.float32]
    sample_rate: int = SAMPLE_RATE

    def __post_init__(self) -> None:
        # Guard clauses keep the invariants obvious and the body flat.
        if self.samples.ndim != 1:
            raise ValueError(f"expected mono (1-D) audio, got {self.samples.ndim}-D")
        if self.samples.dtype != np.float32:
            raise ValueError(f"expected float32 samples, got {self.samples.dtype}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")

    @property
    def num_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration_s(self) -> float:
        """Length of the buffer in seconds."""
        return self.num_samples / self.sample_rate

    @classmethod
    def silence(cls, duration_s: float, sample_rate: int = SAMPLE_RATE) -> AudioBuffer:
        """A silent buffer of the given length — the mixer's blank canvas."""
        count = max(0, round(duration_s * sample_rate))
        return cls(np.zeros(count, dtype=np.float32), sample_rate)
