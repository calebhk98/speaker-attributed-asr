"""Tests for VoiceBank: real-speaker reference sampling (§4.4/§4.8).

Fast tests over a tmp fixture of tiny synthetic wav clips (no real corpus
needed): distinct speakers map to distinct reference audio, non-cloning
engines get a preset instead of audio, and sampling is deterministic given a
seed. Also covers laziness (no disk access until first use) and the
missing-speaker error path.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from satasr.pipeline.voice_bank import VoiceBank, VoiceBankConfig

_SAMPLE_RATE = 16_000
_TONE_DURATION_S = 0.05


def _tone(frequency_hz: float) -> np.ndarray:
    """A short int16 PCM sine tone — enough to give clips distinct content."""
    times = np.arange(round(_TONE_DURATION_S * _SAMPLE_RATE)) / _SAMPLE_RATE
    waveform = 0.2 * np.sin(2.0 * np.pi * frequency_hz * times)
    return (waveform * 32767).astype(np.int16)


def _write_wav(path: Path, pcm16: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(_SAMPLE_RATE)
        wav_file.writeframes(pcm16.tobytes())


def _make_speaker(root: Path, speaker_id: str, frequencies: list[float]) -> None:
    speaker_dir = root / speaker_id
    speaker_dir.mkdir(parents=True, exist_ok=True)
    for index, frequency in enumerate(frequencies):
        _write_wav(speaker_dir / f"clip{index}.wav", _tone(frequency))


def test_distinct_speakers_map_to_distinct_reference_audio(tmp_path: Path) -> None:
    root = tmp_path / "voices"
    _make_speaker(root, "alice", [220.0])
    _make_speaker(root, "bob", [440.0])
    bank = VoiceBank(VoiceBankConfig(source_dir=str(root), seed=0))

    alice_ref = bank.reference("alice", supports_cloning=True)
    bob_ref = bank.reference("bob", supports_cloning=True)

    assert alice_ref.reference_audio is not None
    assert bob_ref.reference_audio is not None
    assert alice_ref.preset is None
    assert not np.array_equal(
        alice_ref.reference_audio.samples, bob_ref.reference_audio.samples
    )


def test_preset_engine_gets_a_configured_preset_not_audio(tmp_path: Path) -> None:
    root = tmp_path / "voices"
    _make_speaker(root, "alice", [220.0])
    bank = VoiceBank(
        VoiceBankConfig(source_dir=str(root), preset_names=("calm", "bright"), seed=0)
    )

    ref = bank.reference("alice", supports_cloning=False)

    assert ref.reference_audio is None
    assert ref.preset in {"calm", "bright"}


def test_sampling_is_deterministic_given_a_seed(tmp_path: Path) -> None:
    root = tmp_path / "voices"
    _make_speaker(root, "alice", [220.0, 330.0, 440.0, 550.0])

    first = VoiceBank(VoiceBankConfig(source_dir=str(root), seed=7))
    second = VoiceBank(VoiceBankConfig(source_dir=str(root), seed=7))

    first_ref = first.reference("alice", supports_cloning=True)
    second_ref = second.reference("alice", supports_cloning=True)
    repeat_ref = first.reference("alice", supports_cloning=True)

    assert first_ref.reference_audio is not None
    assert second_ref.reference_audio is not None
    assert repeat_ref.reference_audio is not None
    assert np.array_equal(
        first_ref.reference_audio.samples, second_ref.reference_audio.samples
    )
    assert np.array_equal(
        first_ref.reference_audio.samples, repeat_ref.reference_audio.samples
    )


def test_different_seeds_can_sample_different_clips(tmp_path: Path) -> None:
    root = tmp_path / "voices"
    for i in range(10):
        _make_speaker(root, f"speaker{i}", [110.0, 220.0, 330.0, 440.0])

    seed_a = VoiceBank(VoiceBankConfig(source_dir=str(root), seed=1))
    seed_b = VoiceBank(VoiceBankConfig(source_dir=str(root), seed=2))

    picks_a = [
        seed_a.reference(
            f"speaker{i}", supports_cloning=True
        ).reference_audio.samples.tobytes()  # type: ignore[union-attr]
        for i in range(10)
    ]
    picks_b = [
        seed_b.reference(
            f"speaker{i}", supports_cloning=True
        ).reference_audio.samples.tobytes()  # type: ignore[union-attr]
        for i in range(10)
    ]

    assert picks_a != picks_b


def test_indexing_is_lazy_until_first_access(tmp_path: Path) -> None:
    root = tmp_path / "voices"  # does not exist yet
    bank = VoiceBank(
        VoiceBankConfig(source_dir=str(root), seed=0)
    )  # must not touch disk

    root.mkdir()
    _make_speaker(root, "alice", [220.0])

    assert bank.speakers() == ("alice",)


def test_speakers_lists_all_discovered_directories_sorted(tmp_path: Path) -> None:
    root = tmp_path / "voices"
    _make_speaker(root, "bob", [220.0])
    _make_speaker(root, "alice", [440.0])
    bank = VoiceBank(VoiceBankConfig(source_dir=str(root), seed=0))

    assert bank.speakers() == ("alice", "bob")


def test_missing_speaker_raises_key_error(tmp_path: Path) -> None:
    root = tmp_path / "voices"
    root.mkdir()
    bank = VoiceBank(VoiceBankConfig(source_dir=str(root), seed=0))

    with pytest.raises(KeyError):
        bank.reference("ghost", supports_cloning=True)


def test_missing_presets_raises_value_error(tmp_path: Path) -> None:
    root = tmp_path / "voices"
    _make_speaker(root, "alice", [220.0])
    bank = VoiceBank(VoiceBankConfig(source_dir=str(root), seed=0))

    with pytest.raises(ValueError):
        bank.reference("alice", supports_cloning=False)
