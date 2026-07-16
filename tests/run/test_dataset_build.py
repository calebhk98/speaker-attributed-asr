"""Tests for the end-to-end dataset builder (offline, sine engine)."""

from __future__ import annotations

from pathlib import Path

from satasr.core.config import MixingConfig
from satasr.pipeline.dataset_writer import read_manifest
from satasr.run.config import AugmentSpec, DatasetStageConfig, PipelineConfig
from satasr.run.dataset_build import build_dataset


def _config(tmp_path: Path, num: int = 6) -> PipelineConfig:
    return PipelineConfig(
        seed=1,
        dataset=DatasetStageConfig(
            output_dir=str(tmp_path / "ds"),
            num_examples=num,
            text_options={"sentences": ["One two three.", "Four five.", "Six seven."]},
            engine_weights={"sine": 1.0},
            augment=(
                AugmentSpec("gain", {"gain_db": -1.0}),
                AugmentSpec("white_noise", {"snr_db": 20.0}),
            ),
        ),
        mixing=MixingConfig(max_simultaneous_speakers=3),
    )


def test_builds_requested_number_of_examples(tmp_path: Path) -> None:
    config = _config(tmp_path, num=6)
    summary = build_dataset(config)
    rows = read_manifest(config.dataset.output_dir)

    assert summary.num_examples == 6
    assert len(rows) == 6
    assert sum(summary.splits.values()) == 6


def test_written_clips_are_well_formed(tmp_path: Path) -> None:
    config = _config(tmp_path, num=8)
    build_dataset(config)
    root = Path(config.dataset.output_dir)

    for row in read_manifest(root):
        assert (root / row.audio_path).exists()
        assert row.transcript  # serialized transcript is non-empty
        assert row.speaker_count_total >= 1
        # the run's ceiling (§4.6) is respected end-to-end
        assert row.speaker_count_max_simultaneous <= 3


def test_is_deterministic_for_a_fixed_seed(tmp_path: Path) -> None:
    first = build_dataset(_config(tmp_path / "a"))
    second = build_dataset(_config(tmp_path / "b"))
    assert first.splits == second.splits
