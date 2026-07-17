"""Compose several augmenters into one (design §4.9).

This is the single place that knows how to apply multiple acoustic transforms
in sequence — callers build one list (e.g. noise then reverb) instead of every
call site re-implementing a loop. :class:`AugmentChain` depends only on the
:class:`~satasr.core.interfaces.Augmenter` protocol, never on concrete
augmenters, so it can chain any registered plugin without caring what it is
(dependency inversion).
"""

from __future__ import annotations

from collections.abc import Sequence

from satasr.core.audio import AudioBuffer
from satasr.core.interfaces import Augmenter


class AugmentChain:
    """An :class:`Augmenter` that applies a fixed sequence of augmenters in order."""

    def __init__(self, augmenters: Sequence[Augmenter]) -> None:
        self._augmenters = tuple(augmenters)

    def apply(self, audio: AudioBuffer) -> AudioBuffer:
        """Run ``audio`` through each augmenter in turn, feeding output to input."""
        for augmenter in self._augmenters:
            audio = augmenter.apply(audio)
        return audio
