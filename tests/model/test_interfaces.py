"""The ``Model`` Protocol should be satisfiable by a tiny fake — no torch, no
checkpoint — so training/eval code can be tested against the abstraction
alone (design §2, CLAUDE.md rule 5)."""

from __future__ import annotations

import numpy as np

from satasr.core.audio import AudioBuffer
from satasr.core.models import PlacedUtterance
from satasr.model import Model


class _FakeModel:
    """Minimal stand-in: always returns the same canned utterance."""

    def predict(self, audio: AudioBuffer) -> tuple[PlacedUtterance, ...]:
        return (PlacedUtterance("A", 0.0, audio.duration_s, "hi"),)


def test_fake_model_satisfies_protocol() -> None:
    assert isinstance(_FakeModel(), Model)


def test_fake_model_predicts_placed_utterances() -> None:
    audio = AudioBuffer(np.zeros(16_000, dtype=np.float32))
    utterances = _FakeModel().predict(audio)
    assert utterances == (PlacedUtterance("A", 0.0, 1.0, "hi"),)
