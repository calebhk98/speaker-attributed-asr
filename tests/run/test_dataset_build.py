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


def _write_bank(root: Path, speakers: list[str]) -> str:
    import wave

    import numpy as np

    for speaker in speakers:
        path = root / speaker / "clip.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(np.zeros(1600, dtype=np.int16).tobytes())
    return str(root)


def test_voice_bank_run_uses_real_speaker_ids(tmp_path: Path) -> None:
    # With a bank configured, a run assigns clips real speaker ids (and would
    # hand cloning engines real reference audio); the total-speaker count is
    # capped to the bank's size so it never asks for more voices than exist.
    speakers = ["libri_100", "libri_200", "libri_300"]
    config = PipelineConfig(
        seed=3,
        dataset=DatasetStageConfig(
            output_dir=str(tmp_path / "ds"),
            num_examples=5,
            text_options={"sentences": ["Alpha beta.", "Gamma delta epsilon."]},
            engine_weights={"sine": 1.0},  # non-cloning -> uses bank presets
            voice_source_dir=_write_bank(tmp_path / "bank", speakers),
            voice_presets=("preset_a", "preset_b"),
        ),
        mixing=MixingConfig(max_simultaneous_speakers=2),
    )
    build_dataset(config)

    rows = read_manifest(config.dataset.output_dir)
    assert len(rows) == 5
    for row in rows:
        assert row.speaker_count_total <= len(speakers)
        assert any(speaker in row.transcript for speaker in speakers)
