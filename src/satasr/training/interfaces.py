"""Abstractions Phase 1 training depends on, besides ``satasr.model.Model``
(design §5, CLAUDE.md rule 5: dependency inversion).

Mirrors ``satasr.model.interfaces``: the loop in ``phase1.py`` depends only on
these Protocols for the optimizer step and checkpoint format, never on
torch/PEFT specifics. A caller wires up the concrete training step (forward
pass, loss, backward, optimizer update, and any multi-GPU placement across
the dual 3090s, §3) and checkpoint writer; this module never imports torch,
so it stays weight-free to import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from satasr.core.audio import AudioBuffer


@runtime_checkable
class TrainStep(Protocol):
    """One optimizer step on a single training example; returns its loss.

    The step is per-example rather than pre-batched so ``phase1.py`` never
    needs to know the model's batching/collation strategy — a caller free to
    batch internally (e.g. buffering a few calls before an actual backward
    pass) so long as each call returns that example's scalar loss.
    """

    def __call__(self, audio: AudioBuffer, transcript: str) -> float: ...


@runtime_checkable
class Checkpointer(Protocol):
    """Persists whatever the concrete model needs to resume or deploy later."""

    def save(self, path: Path) -> None: ...
