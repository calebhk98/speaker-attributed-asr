"""Single, file-loadable configuration for an end-to-end pipeline run.

One TOML file drives every stage. The *tunables* (MixingConfig, ...) stay
defined once in ``satasr.core.config``; this module only composes them with
run-level choices — which text source, which engines, where to write, and which
stages to run. Load one with :func:`satasr.run.loader.load_config`; see
``config.example.toml`` for the on-disk shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from satasr.core.config import MixingConfig


@dataclass(frozen=True)
class AugmentSpec:
    """One acoustic augmenter to apply, by registry name plus its kwargs (§4.9)."""

    name: str
    options: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetStageConfig:
    """The ``build_dataset`` stage: text -> TTS -> align -> mix -> augment -> disk."""

    output_dir: str = "data/synthetic"
    num_examples: int = 20
    text_source: str = "static"  # a satasr.text registry name
    text_options: dict[str, object] = field(default_factory=dict)
    aligner: str = "proportional"  # a satasr.alignment registry name
    engine_weights: dict[str, float] = field(default_factory=lambda: {"sine": 1.0})
    augment: tuple[AugmentSpec, ...] = ()


@dataclass(frozen=True)
class TrainingStageConfig:
    """A training phase. Real runs need the model/torch backend (design §5)."""

    dataset_root: str = "data/synthetic"
    epochs: int = 1


@dataclass(frozen=True)
class EvalStageConfig:
    """Evaluation over a written dataset (design §6)."""

    dataset_root: str = "data/synthetic"
    metrics: tuple[str, ...] = ("der", "cpwer")


@dataclass(frozen=True)
class PipelineConfig:
    """The whole run: an ordered list of stages plus each stage's settings."""

    seed: int = 0
    stages: tuple[str, ...] = ("build_dataset",)
    dataset: DatasetStageConfig = field(default_factory=DatasetStageConfig)
    mixing: MixingConfig = field(default_factory=MixingConfig)
    phase1: TrainingStageConfig = field(default_factory=TrainingStageConfig)
    phase2: TrainingStageConfig = field(default_factory=TrainingStageConfig)
    evaluate: EvalStageConfig = field(default_factory=EvalStageConfig)
