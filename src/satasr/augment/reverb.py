"""Room-impulse-response reverb (design §4.9).

Reverb matters a lot for diarization even where it is roughly ASR-neutral, so
it is not optional coverage. :class:`ReverbAugmenter` convolves audio with a
synthetic, exponentially-decaying impulse response — dependency-free and
parameterized by decay time, which stands in for room size (short decay ~=
small/damped room, long decay ~= large/live room). Real recorded RIRs (e.g.
OpenSLR RIR corpora) are a follow-up; :class:`RirFileReverbAugmenter` is the
stub for that.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from satasr.augment.registry import AUGMENTERS
from satasr.core.audio import AudioBuffer

_MIN_SAMPLE = -1.0
_MAX_SAMPLE = 1.0
_DEFAULT_DECAY_S = 0.3
_DEFAULT_WET_LEVEL = 0.5
_TAIL_DECAY_CONSTANTS = 6  # how many decay constants of tail to render

_RIR_FILE_NOT_IMPLEMENTED = (
    "RirFileReverbAugmenter is a stub: convolving with real, recorded room "
    "impulse responses is not implemented yet. Track/implement it via a "
    "follow-up GitHub issue on github.com/calebhk98/speaker-attributed-asr."
)


def _synthetic_rir(decay_s: float, sample_rate: int) -> NDArray[np.float32]:
    """Exponentially-decaying synthetic impulse response, unit-gain normalized."""
    length = max(2, round(sample_rate * decay_s * _TAIL_DECAY_CONSTANTS))
    steps = np.arange(length, dtype=np.float64)
    envelope = np.exp(-steps / (decay_s * sample_rate))
    normalized = envelope / np.sum(envelope)  # unit gain: reverb keeps dry level
    rir: NDArray[np.float32] = normalized.astype(np.float32)
    return rir


@AUGMENTERS.register("reverb")
class ReverbAugmenter:
    """Convolve audio with a synthetic room impulse response."""

    def __init__(
        self, decay_s: float = _DEFAULT_DECAY_S, wet_level: float = _DEFAULT_WET_LEVEL
    ) -> None:
        if decay_s <= 0.0:
            raise ValueError(f"decay_s must be positive, got {decay_s}")
        if not 0.0 <= wet_level <= 1.0:
            raise ValueError(f"wet_level must be within [0, 1], got {wet_level}")
        self._decay_s = decay_s
        self._wet_level = wet_level

    def apply(self, audio: AudioBuffer) -> AudioBuffer:
        """Return ``audio`` mixed with its convolution against the synthetic RIR."""
        rir = _synthetic_rir(self._decay_s, audio.sample_rate)
        wet = np.convolve(audio.samples, rir, mode="full")[: audio.num_samples]
        dry_level = 1.0 - self._wet_level
        mixed = dry_level * audio.samples + self._wet_level * wet
        clipped = np.clip(mixed, _MIN_SAMPLE, _MAX_SAMPLE).astype(np.float32)
        return AudioBuffer(clipped, audio.sample_rate)


@AUGMENTERS.register("rir_file")
class RirFileReverbAugmenter:
    """Stub for convolving with a real, file-loaded room impulse response."""

    def __init__(self, rir_path: str | None = None) -> None:
        self._rir_path = rir_path

    def apply(self, audio: AudioBuffer) -> AudioBuffer:
        raise NotImplementedError(_RIR_FILE_NOT_IMPLEMENTED)
