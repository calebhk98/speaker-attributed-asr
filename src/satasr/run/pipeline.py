"""Execute the ordered stages of a pipeline run.

``build_dataset`` runs fully here (offline, on the ``sine`` engine). The
training and evaluation stages need the model/torch backend and a checkpoint,
which this environment does not carry — they raise a clear
:class:`PipelineError` explaining what to install and where to run, rather than
failing obscurely. ``dry_run`` reports the plan without executing anything.
"""

from __future__ import annotations

from collections.abc import Callable

from satasr.run.config import PipelineConfig
from satasr.run.dataset_build import build_dataset


class PipelineError(RuntimeError):
    """A stage could not run (unknown stage, or a backend is unavailable)."""


def run_pipeline(
    config: PipelineConfig,
    *,
    dry_run: bool = False,
    on_stage: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run each stage in ``config.stages`` in order; return per-stage results."""
    results: dict[str, object] = {}
    for stage in config.stages:
        handler = _STAGES.get(stage)
        if handler is None:
            raise PipelineError(f"unknown stage {stage!r}; known: {sorted(_STAGES)}")
        if on_stage is not None:
            on_stage(stage)
        results[stage] = "planned" if dry_run else handler(config)
    return results


def _run_build_dataset(config: PipelineConfig) -> object:
    return build_dataset(config)


def _needs_backend(stage: str) -> Callable[[PipelineConfig], object]:
    def handler(_config: PipelineConfig) -> object:
        raise PipelineError(
            f"stage {stage!r} needs the model/torch backend and a DiCoW "
            "checkpoint (design §5/§6). Install the model extras and run on the "
            "GPU host; this environment builds data only."
        )

    return handler


_STAGES: dict[str, Callable[[PipelineConfig], object]] = {
    "build_dataset": _run_build_dataset,
    "phase1": _needs_backend("phase1"),
    "phase2": _needs_backend("phase2"),
    "evaluate": _needs_backend("evaluate"),
}
