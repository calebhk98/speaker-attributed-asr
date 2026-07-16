"""Load a :class:`PipelineConfig` from a TOML file.

TOML is read with the standard library ``tomllib`` (Python 3.11+), so the config
loader adds no dependency. Unknown keys in a table are ignored rather than
fatal, and any table omitted entirely falls back to that dataclass's defaults.
"""

from __future__ import annotations

import tomllib
from collections.abc import Collection, Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

from satasr.core.config import MixingConfig
from satasr.run.config import (
    AugmentSpec,
    DatasetStageConfig,
    EvalStageConfig,
    PipelineConfig,
    TrainingStageConfig,
)


def load_config(path: str | Path) -> PipelineConfig:
    """Parse ``path`` (a TOML file) into a :class:`PipelineConfig`."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return from_mapping(data)


def from_mapping(data: Mapping[str, Any]) -> PipelineConfig:
    """Build a :class:`PipelineConfig` from an already-parsed mapping."""
    return PipelineConfig(
        seed=int(data.get("seed", 0)),
        stages=tuple(data.get("stages", ("build_dataset",))),
        dataset=_dataset(data.get("dataset", {})),
        mixing=_known(MixingConfig, data.get("mixing", {})),
        phase1=_known(TrainingStageConfig, data.get("phase1", {})),
        phase2=_known(TrainingStageConfig, data.get("phase2", {})),
        evaluate=_evaluate(data.get("evaluate", {})),
    )


def _dataset(table: Mapping[str, Any]) -> DatasetStageConfig:
    augment = tuple(
        AugmentSpec(name=item["name"], options=dict(item.get("options", {})))
        for item in table.get("augment", ())
    )
    return DatasetStageConfig(
        augment=augment, **_kwargs(DatasetStageConfig, table, {"augment"})
    )


def _evaluate(table: Mapping[str, Any]) -> EvalStageConfig:
    metrics = tuple(table.get("metrics", ("der", "cpwer")))
    return EvalStageConfig(
        metrics=metrics, **_kwargs(EvalStageConfig, table, {"metrics"})
    )


def _known(dataclass_type: type, table: Mapping[str, Any]) -> Any:
    """Construct a dataclass from the subset of ``table`` keys it declares."""
    return dataclass_type(**_kwargs(dataclass_type, table))


def _kwargs(
    dataclass_type: type, table: Mapping[str, Any], skip: Collection[str] = ()
) -> dict[str, Any]:
    names = {f.name for f in fields(dataclass_type)} - set(skip)
    return {key: value for key, value in table.items() if key in names}
