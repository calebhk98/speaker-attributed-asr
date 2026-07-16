"""Tests for dataset-wide speaker-count balancing (§4.6, issue #32)."""

from __future__ import annotations

import random

from satasr.core.models import SpeakerCounts
from satasr.pipeline.balancing import (
    BalancingConfig,
    plan_clip_counts,
    realized_vs_target,
)

_TOLERANCE = 0.05


def test_realized_total_distribution_tracks_target_within_tolerance() -> None:
    counts = plan_clip_counts(5000, None, random.Random(0))

    report = realized_vs_target(counts)

    for total, target in report.total_target.items():
        realized = report.total_realized.get(total, 0.0)
        assert abs(realized - target) < _TOLERANCE


def test_realized_simultaneous_distribution_tracks_target_within_tolerance() -> None:
    # A high total-speaker floor means max_simultaneous is almost never capped,
    # so its realized distribution should closely match the uncapped target.
    config = BalancingConfig(total_weights={20: 1.0})
    counts = plan_clip_counts(5000, config, random.Random(1))

    report = realized_vs_target(counts, config)

    for count, target in report.simultaneous_target.items():
        realized = report.simultaneous_realized.get(count, 0.0)
        assert abs(realized - target) < _TOLERANCE


def test_the_two_counts_are_sampled_independently() -> None:
    # A 4-total / 2-simultaneous clip must be representable: the simultaneous
    # draw is not forced to equal (or scale with) the total draw.
    config = BalancingConfig(
        total_weights={4: 1.0}, simultaneous_weights={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    )
    counts = plan_clip_counts(500, config, random.Random(2))

    assert SpeakerCounts(total=4, max_simultaneous=2) in counts
    # And every value 1..4 should show up for max_simultaneous, not just one.
    assert {c.max_simultaneous for c in counts} == {1, 2, 3, 4}


def test_max_simultaneous_never_exceeds_total() -> None:
    config = BalancingConfig(
        total_weights={1: 1.0, 2: 1.0, 3: 1.0},
        simultaneous_weights={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0},
    )
    counts = plan_clip_counts(500, config, random.Random(3))

    assert all(c.max_simultaneous <= c.total for c in counts)


def test_zero_weight_counts_are_never_produced() -> None:
    config = BalancingConfig(
        total_weights={1: 1.0, 2: 0.0, 3: 1.0},
        simultaneous_weights={1: 1.0, 2: 0.0, 3: 1.0},
    )
    counts = plan_clip_counts(500, config, random.Random(4))

    assert 2 not in {c.total for c in counts}
    assert 2 not in {c.max_simultaneous for c in counts}


def test_plan_clip_counts_is_deterministic_given_a_seeded_rng() -> None:
    config = BalancingConfig()

    first = plan_clip_counts(50, config, random.Random(7))
    second = plan_clip_counts(50, config, random.Random(7))

    assert first == second
