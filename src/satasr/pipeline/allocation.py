"""Turn ``GenerationConfig`` + the engine registry into per-engine budgets, and
pick engines by those budgets (design §4.2/§4.3).

An engine is added, removed, or reweighted purely through config: it appears
in the mix the moment it registers in ``TTS_ENGINES`` and gets
``default_hours_per_engine`` hours unless ``hours_per_engine`` overrides it;
setting that override to zero removes it without touching orchestration code.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from satasr.core.config import GenerationConfig
from satasr.core.interfaces import TTSEngine
from satasr.core.registry import Registry


@dataclass(frozen=True)
class EngineBudget:
    """The target hours of audio one engine should contribute (§4.3)."""

    name: str
    target_hours: float


def resolve_budgets(
    config: GenerationConfig, registry: Registry[TTSEngine]
) -> tuple[EngineBudget, ...]:
    """Build one :class:`EngineBudget` per registered engine.

    ``registry.available()`` is the single source of truth for which engines
    exist (§4.2) — nothing here names a concrete engine. A configured hour
    target of zero or less excludes the engine entirely, which is how an
    engine is "removed" without editing this module.
    """
    budgets = []
    for name in registry.available():
        target = config.hours_per_engine.get(name, config.default_hours_per_engine)
        if target <= 0:
            continue
        budgets.append(EngineBudget(name, target))
    return tuple(budgets)


def pick_weighted_engine(
    budgets: Sequence[EngineBudget],
    remaining_hours: Mapping[str, float],
    rng: random.Random,
) -> str | None:
    """Draw an engine name weighted by its *configured* target hours (§4.3).

    Only engines with hours still remaining are eligible; the weight used is
    each engine's fixed target (not the shrinking remainder), so the mix
    reflects the configured ratio throughout the run, not just at the start.
    Returns ``None`` once every engine's budget is met.
    """
    eligible = [b for b in budgets if remaining_hours.get(b.name, 0.0) > 0]
    if not eligible:
        return None

    names = [b.name for b in eligible]
    weights = [b.target_hours for b in eligible]
    return str(rng.choices(names, weights=weights, k=1)[0])
