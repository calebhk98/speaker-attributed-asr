"""Dataset-wide speaker-count balancing (design §4.6).

``sample_simultaneous`` (mixing/speaker_sampler.py) already draws one weighted
count for a single clip. This module plans a *whole dataset's* worth of clips
by drawing both of ``SpeakerCounts``' fields — total distinct speakers and max
simultaneous overlap — from their own target weight tables, treating them as
independent, then reports how closely the realized batch tracked those
targets. Reusing ``sample_simultaneous`` for both draws keeps the weighted-pick
logic in exactly one place (CLAUDE.md rule 4): this module only decides what
weights to draw from and how to check the result, never how to sample.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from satasr.core.config import MixingConfig
from satasr.core.models import SpeakerCounts
from satasr.mixing.speaker_sampler import sample_simultaneous

# Sortformer-style ~1/3/6/10 shape (§4.6), extended out to ~20 total speakers
# per issue #32: the peak stays around 4, then tapers to a long, thin tail so
# large-cast clips are rare but still representable in the dataset.
DEFAULT_TOTAL_WEIGHTS: dict[int, float] = {
    1: 1.0,
    2: 3.0,
    3: 6.0,
    4: 10.0,
    5: 8.0,
    6: 6.0,
    7: 4.0,
    8: 3.0,
    9: 2.0,
    10: 2.0,
    **{count: 1.0 for count in range(11, 21)},
}


@dataclass(frozen=True)
class BalancingConfig:
    """The two independent target weight tables this module plans against.

    ``simultaneous_weights`` defaults from ``MixingConfig`` — the existing
    single source of truth for that distribution (§4.6) — rather than
    redefining the Sortformer numbers here. ``total_weights`` defaults to
    ``DEFAULT_TOTAL_WEIGHTS``. Either can be overridden per run.
    """

    total_weights: Mapping[int, float] = field(
        default_factory=lambda: DEFAULT_TOTAL_WEIGHTS
    )
    simultaneous_weights: Mapping[int, float] = field(
        default_factory=lambda: dict(MixingConfig().simultaneous_weights)
    )


@dataclass(frozen=True)
class BalanceReport:
    """Realized vs. target proportions for both counts (§4.6)."""

    total_target: Mapping[int, float]
    total_realized: Mapping[int, float]
    simultaneous_target: Mapping[int, float]
    simultaneous_realized: Mapping[int, float]


def plan_clip_counts(
    num_clips: int,
    config: BalancingConfig | None,
    rng: random.Random,
) -> list[SpeakerCounts]:
    """Plan ``num_clips`` worth of ``(total, max_simultaneous)`` targets.

    The two counts are drawn independently — each from its own weight table,
    each via ``sample_simultaneous`` — with one physical constraint: a clip
    cannot have more simultaneous speakers than it has speakers at all, so
    ``max_simultaneous``'s draw is restricted to counts <= the clip's sampled
    ``total``. That is the only coupling between them (§4.6).
    """
    resolved = config if config is not None else BalancingConfig()
    return [_plan_one(resolved, rng) for _ in range(num_clips)]


def _plan_one(config: BalancingConfig, rng: random.Random) -> SpeakerCounts:
    total = sample_simultaneous(dict(config.total_weights), rng)
    capped = _cap_to_total(config.simultaneous_weights, total)
    max_simultaneous = sample_simultaneous(capped, rng)
    return SpeakerCounts(total=total, max_simultaneous=max_simultaneous)


def _cap_to_total(weights: Mapping[int, float], total: int) -> dict[int, float]:
    """Restrict simultaneous weights to counts that fit within ``total``."""
    capped = {count: weight for count, weight in weights.items() if count <= total}
    return capped if capped else {1: 1.0}


def realized_vs_target(
    counts: Sequence[SpeakerCounts], config: BalancingConfig | None = None
) -> BalanceReport:
    """Compare a planned batch's realized proportions to their targets."""
    resolved = config if config is not None else BalancingConfig()
    return BalanceReport(
        total_target=_normalize(resolved.total_weights),
        total_realized=_proportions(c.total for c in counts),
        simultaneous_target=_normalize(resolved.simultaneous_weights),
        simultaneous_realized=_proportions(c.max_simultaneous for c in counts),
    )


def _normalize(weights: Mapping[int, float]) -> dict[int, float]:
    """Turn a raw weight table into a probability distribution."""
    total_weight = sum(weights.values())
    return {count: weight / total_weight for count, weight in weights.items()}


def _proportions(values: Iterable[int]) -> dict[int, float]:
    """The empirical distribution of an already-realized sample."""
    tally = Counter(values)
    n = sum(tally.values())
    if n == 0:
        return {}
    return {value: count / n for value, count in tally.items()}
