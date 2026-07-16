"""Data Stage 1 orchestrator: drive generation to the 10k-hour budget (§4.2/§4.3).

Re-exports the pieces most callers need: :class:`GenerationOrchestrator` (the
entry point), :class:`GenerationState` (resumable progress), and
:class:`EngineBudget` / :func:`resolve_budgets` (the config-driven weighting).
"""

from satasr.pipeline.allocation import (
    EngineBudget,
    pick_weighted_engine,
    resolve_budgets,
)
from satasr.pipeline.generation import GeneratedClip, GenerationOrchestrator
from satasr.pipeline.state import GenerationState

__all__ = [
    "EngineBudget",
    "GeneratedClip",
    "GenerationOrchestrator",
    "GenerationState",
    "pick_weighted_engine",
    "resolve_budgets",
]
