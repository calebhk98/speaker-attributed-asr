"""Room-impulse-response reverb (design §4.9).

Reverb matters a lot for diarization even where it is roughly ASR-neutral, so
it is not optional coverage. :class:`ReverbAugmenter` convolves audio with a
synthetic, exponentially-decaying impulse response — dependency-free and
parameterized by decay time, which stands in for room size (short decay ~=
small/damped room, long decay ~= large/live room).
:class:`RirFileReverbAugmenter` convolves with a *real* recorded/simulated RIR
sampled from a configured corpus directory (e.g. an OpenSLR RIR corpus),
covering the room-size / mic-distance variety §4.9 calls for that a single
synthetic decay curve cannot. Both share one convolve-and-mix implementation
(:func:`_convolve_and_mix`) — no duplicated DSP.
"""

from __future__ import annotations

import random
import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from satasr.augment.registry import AUGMENTERS
from satasr.core.audio import AudioBuffer

_MIN_SAMPLE = -1.0
_MAX_SAMPLE = 1.0
_DEFAULT_DECAY_S = 0.3
_DEFAULT_WET_LEVEL = 0.5
_TAIL_DECAY_CONSTANTS = 6  # how many decay constants of tail to render
_RIR_GLOB = "*.wav"
_PCM16_FULL_SCALE = 32768.0


def _synthetic_rir(decay_s: float, sample_rate: int) -> NDArray[np.float32]:
    """Exponentially-decaying synthetic impulse response, unit-gain normalized."""
    length = max(2, round(sample_rate * decay_s * _TAIL_DECAY_CONSTANTS))
    steps = np.arange(length, dtype=np.float64)
    envelope = np.exp(-steps / (decay_s * sample_rate))
    normalized = envelope / np.sum(envelope)  # unit gain: reverb keeps dry level
    rir: NDArray[np.float32] = normalized.astype(np.float32)
    return rir


def _convolve_and_mix(
    audio: AudioBuffer, rir: NDArray[np.float32], wet_level: float
) -> AudioBuffer:
    """Convolve ``audio`` with ``rir``, mix at ``wet_level``, trim to input length.

    Single source of truth for the reverb DSP (§4.9): both
    :class:`ReverbAugmenter` (synthetic RIR) and :class:`RirFileReverbAugmenter`
    (real, file-loaded RIR) call this instead of duplicating the
    convolve/mix/clip logic.
    """
    wet = np.convolve(audio.samples, rir, mode="full")[: audio.num_samples]
    dry_level = 1.0 - wet_level
    mixed = dry_level * audio.samples + wet_level * wet
    clipped = np.clip(mixed, _MIN_SAMPLE, _MAX_SAMPLE).astype(np.float32)
    return AudioBuffer(clipped, audio.sample_rate)


def _linear_resample(
    samples: NDArray[np.float32], src_rate: int, dst_rate: int
) -> NDArray[np.float32]:
    """Resample a 1-D float32 array from ``src_rate`` to ``dst_rate`` Hz.

    Plain numpy interpolation (no scipy) mirrors the pattern already used by
    several TTS engines (e.g. ``tts/engines/bark.py``), so loading a RIR file
    needs no extra dependency beyond the project's existing numpy requirement.
    """
    if samples.size == 0 or src_rate == dst_rate:
        return samples
    duration_s = samples.shape[0] / src_rate
    dst_count = max(1, round(duration_s * dst_rate))
    src_times = np.arange(samples.shape[0], dtype=np.float64) / src_rate
    dst_times = np.arange(dst_count, dtype=np.float64) / dst_rate
    interpolated: NDArray[np.float32] = np.interp(dst_times, src_times, samples).astype(
        np.float32
    )
    return interpolated


def _load_rir_wav(path: Path, sample_rate: int) -> NDArray[np.float32]:
    """Decode a 16-bit PCM WAV RIR file and resample/downmix it to ``sample_rate``.

    Uses the standard library's ``wave`` module rather than soundfile (mirrors
    ``tts/engines/voxtral.py``'s WAV decoding), so this stays dependency-free.
    """
    with wave.open(str(path), "rb") as wav_file:
        native_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        raw = wav_file.readframes(wav_file.getnframes())
    pcm16 = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        pcm16 = pcm16.reshape(-1, channels).mean(axis=1)
    samples = (pcm16.astype(np.float32) / _PCM16_FULL_SCALE).astype(np.float32)
    return _linear_resample(samples, native_rate, sample_rate)


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
        return _convolve_and_mix(audio, rir, self._wet_level)


@AUGMENTERS.register("rir_file")
class RirFileReverbAugmenter:
    """Convolve audio with a real RIR sampled from a corpus directory (§4.9).

    Real recorded/simulated RIRs vary far more in room size and mic distance
    than one decay curve can, and that variety matters substantially more for
    diarization than for ASR (§4.9) — so this is not optional coverage.
    Nothing is read from disk until :meth:`apply` runs: construction only
    remembers the directory and RNG, so importing/instantiating this class
    never touches the filesystem.
    """

    def __init__(
        self,
        corpus_dir: str,
        wet_level: float = _DEFAULT_WET_LEVEL,
        rng: random.Random | None = None,
    ) -> None:
        if not 0.0 <= wet_level <= 1.0:
            raise ValueError(f"wet_level must be within [0, 1], got {wet_level}")
        self._corpus_dir = corpus_dir
        self._wet_level = wet_level
        self._rng = rng if rng is not None else random.Random()

    def apply(self, audio: AudioBuffer) -> AudioBuffer:
        """Convolve ``audio`` with one RIR file sampled from the corpus dir."""
        rir = self._sample_rir(audio.sample_rate)
        return _convolve_and_mix(audio, rir, self._wet_level)

    def _sample_rir(self, sample_rate: int) -> NDArray[np.float32]:
        """Pick one ``*.wav`` file from the corpus dir at random and load it."""
        paths = sorted(Path(self._corpus_dir).glob(_RIR_GLOB))
        if not paths:
            raise FileNotFoundError(f"no RIR .wav files found in {self._corpus_dir!r}")
        return _load_rir_wav(self._rng.choice(paths), sample_rate)
