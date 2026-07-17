"""Tests for voice assignment (preset vs. VoiceBank-backed)."""

from __future__ import annotations

import random
import wave
from pathlib import Path

import numpy as np
import pytest

from satasr.run.config import DatasetStageConfig
from satasr.run.voices import (
    BankVoiceProvider,
    PresetVoiceProvider,
    build_voice_provider,
)


def _write_bank(root: Path, speakers: list[str]) -> str:
    """A minimal VoiceBank layout: one subdir of a short wav per speaker."""
    for speaker in speakers:
        path = root / speaker / "clip.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(np.zeros(1600, dtype=np.int16).tobytes())
    return str(root)


def test_preset_provider_rejects_cloning_engines() -> None:
    provider = PresetVoiceProvider()
    assert provider.capacity() is None
    assert provider.speaker_ids(3, random.Random(0)) == ["S1", "S2", "S3"]
    assert provider.reference("S1", supports_cloning=False).preset == "S1"
    with pytest.raises(ValueError, match="voice_source_dir"):
        provider.reference("S1", supports_cloning=True)


def test_bank_provider_supplies_reference_audio(tmp_path: Path) -> None:
    dataset = DatasetStageConfig(
        voice_source_dir=_write_bank(tmp_path, ["alice", "bob", "carol"]),
        voice_presets=("p0", "p1"),
    )
    provider = build_voice_provider(dataset, seed=0)

    assert isinstance(provider, BankVoiceProvider)
    assert provider.capacity() == 3
    ids = provider.speaker_ids(2, random.Random(1))
    assert len(set(ids)) == 2 and set(ids) <= {"alice", "bob", "carol"}

    cloned = provider.reference("alice", supports_cloning=True)
    assert cloned.reference_audio is not None
    preset = provider.reference("bob", supports_cloning=False)
    assert preset.preset in ("p0", "p1")


def test_bank_provider_errors_when_too_few_speakers(tmp_path: Path) -> None:
    dataset = DatasetStageConfig(voice_source_dir=_write_bank(tmp_path, ["solo"]))
    provider = build_voice_provider(dataset, seed=0)
    with pytest.raises(ValueError, match="distinct"):
        provider.speaker_ids(2, random.Random(0))


def test_no_source_dir_uses_preset_provider() -> None:
    provider = build_voice_provider(DatasetStageConfig(), seed=0)
    assert isinstance(provider, PresetVoiceProvider)
