"""Tunable parameters live here and nowhere else.

The design document repeatedly calls for knobs, not hardcoded numbers: the max
simultaneous-speaker count (§1, §4.6), truncation window (§4.6), per-engine hour
allocation (§4.2). Centralising them means a change is a one-line edit, and the
same defaults feed generation, mixing and tests alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MixingConfig:
    """Controls how single-speaker clips are spliced into overlap (§4.6)."""

    # Ceiling on how many speakers may overlap at one instant. Trivially tunable
    # per the scope decision in §1 — default 6 covers group-conversation cases.
    max_simultaneous_speakers: int = 6

    # Interruption truncation window, in seconds (§4.6): a cut lands somewhere in
    # this range after the interrupting speaker starts.
    min_truncation_s: float = 0.01
    max_truncation_s: float = 3.0

    # Fraction of interrupted utterances left uncut (speaker talks through it).
    uncut_probability: float = 0.5

    # Deliberate balancing of the simultaneous-speaker distribution (§4.6),
    # keyed by count. Mirrors the Sortformer ~1/3/6/10 weighting idea. Any count
    # above the largest key is disallowed; the mixer normalises these weights.
    simultaneous_weights: dict[int, float] = field(
        default_factory=lambda: {1: 1.0, 2: 3.0, 3: 6.0, 4: 10.0, 5: 6.0, 6: 3.0}
    )

    # Marker appended to a transcript when a cut lands mid-word (§4.7, Q4).
    truncation_marker: str = "-"


@dataclass(frozen=True)
class GenerationConfig:
    """Controls the synthetic-data generation budget (§4.2)."""

    target_hours: float = 10_000.0
    default_hours_per_engine: float = 1_000.0

    # Optional per-engine overrides for the weighted allocation strategy (§4.3).
    hours_per_engine: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    """Top-level config aggregate passed through the pipeline."""

    mixing: MixingConfig = field(default_factory=MixingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


def default_config() -> Config:
    """The single entry point for obtaining configuration."""
    return Config()
