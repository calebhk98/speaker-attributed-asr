"""Tests for WhiteNoiseAugmenter and MusanNoiseAugmenter (SNR accuracy, §4.9)."""

from __future__ import annotations

import itertools
import os
import random
import wave
from pathlib import Path

import numpy as np
import pytest

from satasr.augment.noise import MusanNoiseAugmenter, WhiteNoiseAugmenter
from satasr.core.audio import SAMPLE_RATE, AudioBuffer

_SNR_TOLERANCE_DB = 1.0


def _tone(amplitude: float = 0.3, duration_s: float = 1.0) -> AudioBuffer:
    times = np.arange(round(SAMPLE_RATE * duration_s), dtype=np.float32) / SAMPLE_RATE
    wave_form = amplitude * np.sin(2.0 * np.pi * 220.0 * times)
    return AudioBuffer(wave_form.astype(np.float32), SAMPLE_RATE)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _measured_snr_db(dry: AudioBuffer, wet: AudioBuffer) -> float:
    noise_only = wet.samples - dry.samples
    return float(20.0 * np.log10(_rms(dry.samples) / _rms(noise_only)))


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write 16-bit PCM mono WAV using only the stdlib (no soundfile needed)."""
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def _write_noise_corpus(
    corpus_dir: Path,
    rng: random.Random,
    num_files: int = 3,
    duration_s: float = 1.0,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    """A tiny synthetic "MUSAN-style" corpus: a few random-noise wav files."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    count = round(duration_s * sample_rate)
    for index in range(num_files):
        samples = np.array(
            [rng.uniform(-0.5, 0.5) for _ in range(count)], dtype=np.float32
        )
        _write_wav(corpus_dir / f"noise_{index}.wav", samples, sample_rate)
    return corpus_dir


def test_white_noise_achieves_approximately_the_requested_snr() -> None:
    audio = _tone()
    target_snr_db = 10.0
    augmenter = WhiteNoiseAugmenter(snr_db=target_snr_db, rng=random.Random(1))

    noisy = augmenter.apply(audio)

    assert _measured_snr_db(audio, noisy) == pytest.approx(
        target_snr_db, abs=_SNR_TOLERANCE_DB
    )


def test_white_noise_deterministic_given_a_seeded_rng() -> None:
    audio = _tone()
    first = WhiteNoiseAugmenter(snr_db=5.0, rng=random.Random(42)).apply(audio)
    second = WhiteNoiseAugmenter(snr_db=5.0, rng=random.Random(42)).apply(audio)
    assert np.array_equal(first.samples, second.samples)


def test_white_noise_different_seeds_give_different_noise() -> None:
    audio = _tone()
    first = WhiteNoiseAugmenter(snr_db=5.0, rng=random.Random(1)).apply(audio)
    second = WhiteNoiseAugmenter(snr_db=5.0, rng=random.Random(2)).apply(audio)
    assert not np.array_equal(first.samples, second.samples)


def test_white_noise_output_is_float32_mono_in_range() -> None:
    audio = _tone()
    result = WhiteNoiseAugmenter(snr_db=0.0, rng=random.Random(0)).apply(audio)
    assert result.samples.dtype == np.float32
    assert result.samples.ndim == 1
    assert result.samples.max() <= 1.0
    assert result.samples.min() >= -1.0


def test_musan_achieves_approximately_the_requested_snr(tmp_path: Path) -> None:
    corpus_dir = _write_noise_corpus(tmp_path / "musan", random.Random(7))
    audio = _tone()
    target_snr_db = 8.0
    augmenter = MusanNoiseAugmenter(
        snr_db=target_snr_db, corpus_dir=str(corpus_dir), rng=random.Random(1)
    )

    noisy = augmenter.apply(audio)

    assert _measured_snr_db(audio, noisy) == pytest.approx(
        target_snr_db, abs=_SNR_TOLERANCE_DB
    )


def test_musan_output_is_float32_mono_in_range(tmp_path: Path) -> None:
    corpus_dir = _write_noise_corpus(tmp_path / "musan", random.Random(3))
    augmenter = MusanNoiseAugmenter(
        snr_db=0.0, corpus_dir=str(corpus_dir), rng=random.Random(0)
    )
    result = augmenter.apply(_tone())
    assert result.samples.dtype == np.float32
    assert result.samples.ndim == 1
    assert result.samples.max() <= 1.0
    assert result.samples.min() >= -1.0


def test_musan_deterministic_given_a_seeded_rng(tmp_path: Path) -> None:
    corpus_dir = _write_noise_corpus(tmp_path / "musan", random.Random(11))
    audio = _tone()
    first = MusanNoiseAugmenter(
        snr_db=5.0, corpus_dir=str(corpus_dir), rng=random.Random(42)
    ).apply(audio)
    second = MusanNoiseAugmenter(
        snr_db=5.0, corpus_dir=str(corpus_dir), rng=random.Random(42)
    ).apply(audio)
    assert np.array_equal(first.samples, second.samples)


def test_musan_resamples_clips_at_a_different_native_rate(tmp_path: Path) -> None:
    corpus_dir = _write_noise_corpus(
        tmp_path / "musan", random.Random(5), sample_rate=8_000
    )
    audio = _tone()
    target_snr_db = 6.0
    augmenter = MusanNoiseAugmenter(
        snr_db=target_snr_db, corpus_dir=str(corpus_dir), rng=random.Random(2)
    )

    noisy = augmenter.apply(audio)

    assert noisy.sample_rate == SAMPLE_RATE
    assert noisy.num_samples == audio.num_samples
    assert _measured_snr_db(audio, noisy) == pytest.approx(
        target_snr_db, abs=_SNR_TOLERANCE_DB
    )


def test_musan_raises_when_corpus_has_no_wav_files(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_musan"
    empty_dir.mkdir()
    augmenter = MusanNoiseAugmenter(
        snr_db=5.0, corpus_dir=str(empty_dir), rng=random.Random(0)
    )
    with pytest.raises(ValueError, match="no .wav files"):
        augmenter.apply(_tone())


@pytest.mark.slow
def test_musan_matches_target_snr_broadly_across_seeds_and_levels(
    tmp_path: Path,
) -> None:
    """Broad SNR/seed sweep (§4.9: vary noise widely, not one fixed level).

    Marked slow because it multiplies the SNR-accuracy check across many
    seeds and target levels rather than checking just one combination.
    """
    corpus_dir = _write_noise_corpus(
        tmp_path / "musan", random.Random(99), num_files=5, duration_s=2.0
    )
    audio = _tone(duration_s=2.0)
    snr_levels = (-5.0, 0.0, 5.0, 10.0, 20.0)
    seeds = range(5)
    for seed, target_snr_db in itertools.product(seeds, snr_levels):
        augmenter = MusanNoiseAugmenter(
            snr_db=target_snr_db, corpus_dir=str(corpus_dir), rng=random.Random(seed)
        )
        noisy = augmenter.apply(audio)
        assert _measured_snr_db(audio, noisy) == pytest.approx(
            target_snr_db, abs=_SNR_TOLERANCE_DB
        )


@pytest.mark.slow
def test_musan_against_a_real_corpus_dir_if_configured() -> None:
    """Opt-in check against a real, locally-downloaded MUSAN ``noise/`` dir.

    Skips unless ``MUSAN_CORPUS_DIR`` is set, since the real corpus is a
    multi-gigabyte download nobody should be forced to fetch just to run
    ``-m slow`` locally or in CI.
    """
    corpus_dir = os.environ.get("MUSAN_CORPUS_DIR")
    if not corpus_dir:
        pytest.skip("set MUSAN_CORPUS_DIR to a real MUSAN noise/ dir to run this")
    augmenter = MusanNoiseAugmenter(
        snr_db=10.0, corpus_dir=corpus_dir, rng=random.Random(0)
    )
    noisy = augmenter.apply(_tone())
    assert _measured_snr_db(_tone(), noisy) == pytest.approx(
        10.0, abs=_SNR_TOLERANCE_DB
    )
