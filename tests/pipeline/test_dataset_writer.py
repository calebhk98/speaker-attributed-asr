"""Tests for DatasetWriter: round trip fidelity and split discipline (§4.1/§4.7).

Fast, filesystem-only tests over ``tmp_path`` — no audio backends, no network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from satasr.core.audio import AudioBuffer
from satasr.core.models import MixedClip, PlacedUtterance, SpeakerCounts
from satasr.format.serialization import parse_utterances
from satasr.pipeline.dataset_writer import (
    DatasetWriter,
    SplitRatios,
    assign_split,
    read_audio,
    read_manifest,
)

_SAMPLE_RATE = 16_000


def _sine_clip(seconds: float = 0.25, freq: float = 220.0) -> AudioBuffer:
    """A tiny, deterministic non-silent buffer — cheap to encode/decode as PCM16."""
    t = np.arange(round(seconds * _SAMPLE_RATE), dtype=np.float32) / _SAMPLE_RATE
    samples = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return AudioBuffer(samples, _SAMPLE_RATE)


def _mixed_clip() -> MixedClip:
    utterances = (
        PlacedUtterance("alice", 0.0, 0.2, "hello there"),
        PlacedUtterance("bob", 0.1, 0.25, "hi"),
    )
    return MixedClip(
        audio=_sine_clip(),
        utterances=utterances,
        counts=SpeakerCounts(total=2, max_simultaneous=2),
        metadata={"engine": "sine"},
    )


def test_write_then_read_reproduces_utterances_times_and_counts(
    tmp_path: Path,
) -> None:
    clip = _mixed_clip()
    writer = DatasetWriter(tmp_path)

    written_row = writer.write("clip-001", clip)
    (row,) = read_manifest(tmp_path)

    assert row == written_row
    assert row.speaker_count_total == clip.counts.total
    assert row.speaker_count_max_simultaneous == clip.counts.max_simultaneous
    round_tripped = parse_utterances(row.transcript)
    assert round_tripped == tuple(
        sorted(clip.utterances, key=lambda u: (u.start_s, u.end_s, u.speaker_id))
    )


def test_write_then_read_reproduces_audio_samples(tmp_path: Path) -> None:
    clip = _mixed_clip()
    writer = DatasetWriter(tmp_path)

    row = writer.write("clip-002", clip)
    restored = read_audio(tmp_path, row)

    assert restored.sample_rate == clip.audio.sample_rate
    assert restored.num_samples == clip.audio.num_samples
    # 16-bit PCM round trip loses precision but must stay perceptually identical.
    assert np.abs(restored.samples - clip.audio.samples).max() < 1e-3


def test_metadata_round_trips_through_the_manifest(tmp_path: Path) -> None:
    clip = _mixed_clip()
    writer = DatasetWriter(tmp_path)

    writer.write("clip-003", clip)
    (row,) = read_manifest(tmp_path)

    assert row.metadata == clip.metadata


def test_audio_file_lands_under_its_assigned_splits_directory(tmp_path: Path) -> None:
    clip = _mixed_clip()
    writer = DatasetWriter(tmp_path)

    row = writer.write("clip-004", clip)

    assert row.audio_path.startswith(f"{row.split}/audio/")
    assert (tmp_path / row.audio_path).is_file()


def test_split_assignment_is_deterministic_across_calls_and_instances(
    tmp_path: Path,
) -> None:
    first = assign_split("stable-id")
    second = assign_split("stable-id")
    via_writer = DatasetWriter(tmp_path).write("stable-id", _mixed_clip()).split

    assert first == second == via_writer


def test_split_assignment_is_disjoint_across_many_clip_ids() -> None:
    ids = [f"clip-{i}" for i in range(500)]
    assignments = {clip_id: assign_split(clip_id) for clip_id in ids}

    buckets: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for clip_id, split in assignments.items():
        buckets[split].add(clip_id)

    # Every id lands in exactly one bucket -- the three sets partition the ids.
    assert buckets["train"] | buckets["val"] | buckets["test"] == set(ids)
    assert buckets["train"].isdisjoint(buckets["val"])
    assert buckets["train"].isdisjoint(buckets["test"])
    assert buckets["val"].isdisjoint(buckets["test"])
    # A skewed split config should visibly skew the resulting proportions.
    assert len(buckets["train"]) > len(buckets["val"])
    assert len(buckets["train"]) > len(buckets["test"])


def test_custom_ratios_change_which_split_a_clip_lands_in() -> None:
    ids = [f"clip-{i}" for i in range(200)]
    all_train = SplitRatios(train=1.0, val=0.0, test=0.0)

    assert all(assign_split(clip_id, all_train) == "train" for clip_id in ids)


def test_split_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        SplitRatios(train=0.5, val=0.5, test=0.5)


def test_manifest_accumulates_one_row_per_write(tmp_path: Path) -> None:
    writer = DatasetWriter(tmp_path)
    writer.write("clip-a", _mixed_clip())
    writer.write("clip-b", _mixed_clip())

    rows = read_manifest(tmp_path)

    assert [row.clip_id for row in rows] == ["clip-a", "clip-b"]


def test_read_manifest_on_empty_root_returns_empty_tuple(tmp_path: Path) -> None:
    assert read_manifest(tmp_path) == ()
