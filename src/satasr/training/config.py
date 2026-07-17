"""Phase 1 tunables and result types — the one place these shapes live
(CLAUDE.md rule 4), split out of ``phase1.py`` to keep that module short
(rule 3).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from satasr.core.models import PlacedUtterance
from satasr.eval.cpwer import cpwer

#: Lower-is-better score for a reference/hypothesis pair (default: cpWER, §6).
MetricFn = Callable[[Sequence[PlacedUtterance], Sequence[PlacedUtterance]], float]


@dataclass(frozen=True)
class Phase1Config:
    """Tunables for one Phase 1 run (§5) — the one place these knobs live."""

    checkpoint_dir: Path
    max_epochs: int = 10
    patience: int = 3  # epochs without val improvement before early stop
    min_delta: float = 0.0  # smallest val_metric drop that counts as improved
    metric: MetricFn = field(default=cpwer)


@dataclass(frozen=True)
class EpochStats:
    """One epoch's mean training loss and validation metric."""

    epoch: int
    train_loss: float
    val_metric: float


@dataclass(frozen=True)
class Phase1Result:
    """What a completed (or early-stopped) Phase 1 run produced."""

    best_checkpoint: Path | None
    best_val_metric: float
    history: tuple[EpochStats, ...]
    stopped_early: bool
