"""Tests for loading a PipelineConfig from TOML."""

from __future__ import annotations

from pathlib import Path

from satasr.run.config import AugmentSpec, PipelineConfig
from satasr.run.loader import load_config

_FULL = """
seed = 5
stages = ["build_dataset", "evaluate"]

[dataset]
output_dir = "out"
num_examples = 3

[dataset.text_options]
sentences = ["A.", "B."]

[dataset.engine_weights]
sine = 2.0

[[dataset.augment]]
name = "gain"
options = { gain_db = -2.0 }

[mixing]
max_simultaneous_speakers = 2

[evaluate]
metrics = ["der"]
"""


def _load(tmp_path: Path, text: str) -> PipelineConfig:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


def test_loads_all_sections(tmp_path: Path) -> None:
    config = _load(tmp_path, _FULL)
    assert config.seed == 5
    assert config.stages == ("build_dataset", "evaluate")
    assert config.dataset.num_examples == 3
    assert config.dataset.text_options["sentences"] == ["A.", "B."]
    assert config.dataset.engine_weights == {"sine": 2.0}
    assert config.dataset.augment == (AugmentSpec("gain", {"gain_db": -2.0}),)
    assert config.mixing.max_simultaneous_speakers == 2
    assert config.evaluate.metrics == ("der",)


def test_missing_tables_fall_back_to_defaults(tmp_path: Path) -> None:
    config = _load(tmp_path, "seed = 1\n")
    assert config.stages == ("build_dataset",)
    assert config.dataset.num_examples == 20
    assert config.dataset.engine_weights == {"sine": 1.0}
    assert config.evaluate.metrics == ("der", "cpwer")


def test_unknown_keys_are_ignored(tmp_path: Path) -> None:
    config = _load(tmp_path, "[dataset]\nnum_examples = 2\nnonsense = 9\n")
    assert config.dataset.num_examples == 2
