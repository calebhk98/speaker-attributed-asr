"""Additive noise augmenters (design §4.9).

Reverb aside, varied noise is the other half of acoustic coverage: since there
is no single fixed target environment, we want a broad spread of SNRs and
noise *types* rather than one "realistic" level. :class:`WhiteNoiseAugmenter`
is the dependency-free work-horse (Gaussian noise scaled to a target SNR);
:class:`MusanNoiseAugmenter` mixes in real recorded noise from a MUSAN-style
corpus (the standard corpus pyannote's own embedding model trains against) at
a target SNR. Both share the SNR math in :func:`_mix_at_snr` so "what scale
hits a given SNR" is defined exactly once.
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
_INT16_FULL_SCALE = 32768.0  # 16-bit PCM full-scale divisor for WAV decoding


def _rms(samples: NDArray[np.float32]) -> float:
    """Root-mean-square level of a signal; 0.0 for an empty or silent buffer."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _noise_rms_for_snr(signal_rms: float, snr_db: float) -> float:
    """RMS level a noise signal must have to sit ``snr_db`` under ``signal_rms``."""
    return float(signal_rms / (10.0 ** (snr_db / 20.0)))


def _mix_at_snr(
    signal: NDArray[np.float32], noise: NDArray[np.float32], snr_db: float
) -> NDArray[np.float32]:
    """Rescale ``noise`` to sit ``snr_db`` under ``signal`` (by RMS), mix, clip.

    The single source of truth for "what scale hits a target SNR" (§4.9):
    every noise augmenter in this module — synthetic or MUSAN-backed — routes
    its raw noise through this one function rather than each computing (and
    risking drifting from) its own copy of the SNR math.
    """
    target_rms = _noise_rms_for_snr(_rms(signal), snr_db)
    noise_rms = _rms(noise)
    scale = target_rms / noise_rms if noise_rms > 0.0 else 0.0
    mixed = signal + noise * np.float32(scale)
    return np.clip(mixed, _MIN_SAMPLE, _MAX_SAMPLE).astype(np.float32)


@AUGMENTERS.register("white_noise")
class WhiteNoiseAugmenter:
    """Add Gaussian noise scaled to hit a target SNR relative to the signal."""

    def __init__(self, snr_db: float, rng: random.Random) -> None:
        self._snr_db = snr_db
        self._rng = rng

    def apply(self, audio: AudioBuffer) -> AudioBuffer:
        """Return ``audio`` plus Gaussian noise at this augmenter's target SNR."""
        noise = self._draw_noise(audio.num_samples)
        mixed = _mix_at_snr(audio.samples, noise, self._snr_db)
        return AudioBuffer(mixed, audio.sample_rate)

    def _draw_noise(self, count: int) -> NDArray[np.float32]:
        """Draw ``count`` unit-variance Gaussian samples from this augmenter's rng.

        Drawing straight from the injected ``random.Random`` (rather than
        seeding a separate numpy generator) keeps determinism obvious: the
        same seeded rng always yields the same noise, byte for byte. The
        actual target level is applied afterwards by :func:`_mix_at_snr`.
        """
        values = [self._rng.gauss(0.0, 1.0) for _ in range(count)]
        return np.array(values, dtype=np.float32)


def _list_noise_paths(corpus_dir: str) -> tuple[str, ...]:
    """Sorted ``.wav`` paths under ``corpus_dir`` (recursive, MUSAN-style tree).

    Sorting makes the listing deterministic across filesystems/OSes so a
    seeded ``rng.choice`` over it always picks the same clip.
    """
    paths = sorted(str(path) for path in Path(corpus_dir).rglob("*.wav"))
    if not paths:
        raise ValueError(f"no .wav files found under MUSAN corpus_dir {corpus_dir!r}")
    return tuple(paths)


def _read_wav_mono(path: str) -> tuple[NDArray[np.float32], int]:
    """Decode a 16-bit PCM WAV file into float32 mono samples plus its rate."""
    with wave.open(path, "rb") as wav_file:
        native_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        raw = wav_file.readframes(wav_file.getnframes())
    pcm16 = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        pcm16 = pcm16.reshape(-1, channels).mean(axis=1)
    samples = (pcm16.astype(np.float32) / _INT16_FULL_SCALE).astype(np.float32)
    return samples, native_rate


def _resample_linear(
    samples: NDArray[np.float32], src_rate: int, dst_rate: int
) -> NDArray[np.float32]:
    """Resample 1-D float32 ``samples`` from ``src_rate`` to ``dst_rate`` Hz.

    Plain numpy linear interpolation — matching the project's other
    dependency-free resamplers — so decoding a MUSAN clip needs nothing
    beyond the stdlib ``wave`` module and numpy.
    """
    if samples.size == 0 or src_rate == dst_rate:
        return samples
    duration_s = samples.shape[0] / src_rate
    dst_count = max(1, round(duration_s * dst_rate))
    src_times = np.arange(samples.shape[0], dtype=np.float64) / src_rate
    dst_times = np.arange(dst_count, dtype=np.float64) / dst_rate
    resampled: NDArray[np.float32] = np.interp(dst_times, src_times, samples).astype(
        np.float32
    )
    return resampled


def _fit_length(
    samples: NDArray[np.float32], count: int, rng: random.Random
) -> NDArray[np.float32]:
    """Loop or crop ``samples`` to exactly ``count`` frames.

    MUSAN clips are rarely the same length as the target audio: shorter noise
    is tiled (looped) to cover it; longer noise is cropped at a random offset
    so repeated applications of the same clip sample different sections.
    """
    if samples.size == 0:
        return np.zeros(count, dtype=np.float32)
    if samples.size < count:
        reps = -(-count // samples.size)  # ceil division, dependency-free
        return np.tile(samples, reps)[:count]
    if samples.size == count:
        return samples
    start = rng.randrange(samples.size - count + 1)
    return samples[start : start + count]


@AUGMENTERS.register("musan")
class MusanNoiseAugmenter:
    """Mix in a random real recorded noise clip (MUSAN corpus) at a target SNR.

    Nothing touches disk until :meth:`apply` actually runs: construction only
    records ``corpus_dir``, and the directory listing is discovered lazily on
    first use (and cached), so registering/instantiating this augmenter stays
    IO-free (§4.9 / house style).
    """

    def __init__(self, snr_db: float, corpus_dir: str, rng: random.Random) -> None:
        self._snr_db = snr_db
        self._corpus_dir = corpus_dir
        self._rng = rng
        self._noise_paths: tuple[str, ...] | None = None

    def apply(self, audio: AudioBuffer) -> AudioBuffer:
        """Return ``audio`` mixed with a random MUSAN clip at the target SNR."""
        raw, native_rate = self._sample_clip()
        resampled = _resample_linear(raw, native_rate, audio.sample_rate)
        noise = _fit_length(resampled, audio.num_samples, self._rng)
        mixed = _mix_at_snr(audio.samples, noise, self._snr_db)
        return AudioBuffer(mixed, audio.sample_rate)

    def _sample_clip(self) -> tuple[NDArray[np.float32], int]:
        """Pick and decode one random MUSAN wav file from the corpus."""
        if self._noise_paths is None:
            self._noise_paths = _list_noise_paths(self._corpus_dir)
        path = self._rng.choice(self._noise_paths)
        return _read_wav_mono(path)
