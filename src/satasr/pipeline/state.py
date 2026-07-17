"""Track generated hours per engine so a run can be paused and resumed (§4.2).

Generation is expected to take far longer than one process lifetime at the
10k-hour scale, so progress is a plain, serializable value object rather than
something buried in the orchestrator's call stack.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from satasr.pipeline.allocation import EngineBudget


@dataclass
class GenerationState:
    """Hours generated so far, keyed by engine name.

    Construct with a prior run's ``hours_generated`` to resume it (§4.2); a
    fresh run starts from the default empty mapping.
    """

    hours_generated: dict[str, float] = field(default_factory=dict)

    def record(self, engine_name: str, hours: float) -> None:
        """Add ``hours`` of newly generated audio credited to ``engine_name``."""
        current = self.hours_generated.get(engine_name, 0.0)
        self.hours_generated[engine_name] = current + hours

    def remaining(self, budgets: Iterable[EngineBudget]) -> dict[str, float]:
        """Hours left before each budget is met (never negative)."""
        return {budget.name: self._remaining_for(budget) for budget in budgets}

    def _remaining_for(self, budget: EngineBudget) -> float:
        generated = self.hours_generated.get(budget.name, 0.0)
        return max(0.0, budget.target_hours - generated)

    def is_complete(self, budgets: Iterable[EngineBudget]) -> bool:
        """True once every engine has reached its configured target hours."""
        return all(
            self.hours_generated.get(budget.name, 0.0) >= budget.target_hours
            for budget in budgets
        )
