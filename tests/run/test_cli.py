"""Tests for the `satasr run` CLI and the stage runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from satasr.cli import main
from satasr.pipeline.dataset_writer import read_manifest
from satasr.run.config import PipelineConfig
from satasr.run.pipeline import PipelineError, run_pipeline


def _write_config(
    tmp_path: Path, stages: str = '["build_dataset"]'
) -> tuple[Path, Path]:
    out = tmp_path / "ds"
    text = f"""
stages = {stages}
[dataset]
output_dir = "{out.as_posix()}"
num_examples = 4
[dataset.text_options]
sentences = ["Alpha beta gamma.", "Delta epsilon."]
[dataset.engine_weights]
sine = 1.0
[mixing]
max_simultaneous_speakers = 2
"""
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path, out


def test_run_builds_the_dataset(tmp_path: Path) -> None:
    path, out = _write_config(tmp_path)
    assert main(["run", "--config", str(path)]) == 0
    assert len(read_manifest(out)) == 4


def test_dry_run_builds_nothing(tmp_path: Path) -> None:
    path, out = _write_config(tmp_path)
    assert main(["run", "--config", str(path), "--dry-run"]) == 0
    assert not out.exists()


def test_backend_stage_exits_nonzero(tmp_path: Path) -> None:
    path, _ = _write_config(tmp_path, stages='["phase1"]')
    assert main(["run", "--config", str(path)]) == 1


def test_unknown_stage_raises() -> None:
    with pytest.raises(PipelineError, match="unknown stage"):
        run_pipeline(PipelineConfig(stages=("nope",)))
