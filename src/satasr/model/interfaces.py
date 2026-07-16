"""The abstract boundary between orchestration (training/eval) and any
concrete ASR checkpoint (design §2, CLAUDE.md rule 5: dependency inversion).

Mirrors the pattern in :mod:`satasr.core.interfaces`: high-level code depends
only on the :class:`Model` Protocol below, never on a specific checkpoint
class. A new back-end (a different fine-tune, a future architecture) is added
by implementing this Protocol — no existing training/eval code changes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from satasr.core.audio import AudioBuffer
from satasr.core.models import PlacedUtterance


@runtime_checkable
class Model(Protocol):
    """A speaker-attributed ASR model: mixed audio in, placed utterances out.

    ``predict`` returns the same ``tuple[PlacedUtterance, ...]`` type that
    :attr:`~satasr.core.models.MixedClip.utterances` holds (§4.7) — the
    prediction is "MixedClip-compatible": a caller that also has the source
    audio and speaker counts on hand can wrap it straight into a
    :class:`~satasr.core.models.MixedClip` for scoring, with no extra
    conversion step.
    """

    def predict(self, audio: AudioBuffer) -> tuple[PlacedUtterance, ...]: ...
