"""Tests for deliberate speaker-count sampling (design §4.6)."""

from __future__ import annotations

import random

from satasr.mixing.speaker_sampler import sample_simultaneous


def test_respects_weight_of_zero() -> None:
    # A count with weight 0 must never be drawn.
    weights = {1: 0.0, 2: 1.0}
    rng = random.Random(0)
    draws = {sample_simultaneous(weights, rng) for _ in range(200)}
    assert draws == {2}


def test_distribution_tracks_weights() -> None:
    # Heavier weight -> drawn more often. Uses a fixed seed for determinism.
    weights = {1: 1.0, 4: 9.0}
    rng = random.Random(42)
    counts = {1: 0, 4: 0}
    for _ in range(1000):
        counts[sample_simultaneous(weights, rng)] += 1
    assert counts[4] > counts[1] * 3


def test_only_returns_configured_counts() -> None:
    weights = {2: 1.0, 3: 1.0, 6: 1.0}
    rng = random.Random(7)
    draws = {sample_simultaneous(weights, rng) for _ in range(300)}
    assert draws <= {2, 3, 6}
