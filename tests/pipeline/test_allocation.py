"""Tests for config-driven engine budgets and weighted picking (§4.2/§4.3)."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from satasr.core.config import GenerationConfig
from satasr.core.interfaces import TTSEngine
from satasr.core.registry import Registry
from satasr.pipeline.allocation import (
    EngineBudget,
    pick_weighted_engine,
    resolve_budgets,
)

RegistryFactory = Callable[[Sequence[str]], Registry[TTSEngine]]


def test_resolve_budgets_applies_default_and_overrides(
    sine_registry: RegistryFactory,
) -> None:
    registry = sine_registry(["heavy", "light"])
    config = GenerationConfig(
        default_hours_per_engine=100.0, hours_per_engine={"heavy": 900.0}
    )

    budgets = resolve_budgets(config, registry)

    by_name = {budget.name: budget.target_hours for budget in budgets}
    assert by_name == {"heavy": 900.0, "light": 100.0}


def test_resolve_budgets_excludes_zeroed_out_engines(
    sine_registry: RegistryFactory,
) -> None:
    # Setting an override to zero is how an engine is "removed" via config
    # alone (§4.2) — no orchestration code changes.
    registry = sine_registry(["kept", "dropped"])
    config = GenerationConfig(
        default_hours_per_engine=50.0, hours_per_engine={"dropped": 0.0}
    )

    budgets = resolve_budgets(config, registry)

    assert [budget.name for budget in budgets] == ["kept"]


def test_pick_weighted_engine_returns_none_once_all_budgets_met() -> None:
    budgets = (EngineBudget("only", 10.0),)
    rng = random.Random(0)

    result = pick_weighted_engine(budgets, {"only": 0.0}, rng)

    assert result is None


def test_pick_weighted_engine_ignores_engines_with_no_remaining_hours() -> None:
    budgets = (EngineBudget("done", 10.0), EngineBudget("active", 10.0))
    rng = random.Random(1)

    picks = {
        pick_weighted_engine(budgets, {"done": 0.0, "active": 5.0}, rng)
        for _ in range(50)
    }

    assert picks == {"active"}


def test_pick_weighted_engine_distribution_tracks_configured_weights() -> None:
    # Same pattern as mixing/speaker_sampler's weight test: a fixed seed makes
    # this deterministic, and a heavier weight should dominate the draws.
    budgets = (EngineBudget("heavy", 9.0), EngineBudget("light", 1.0))
    remaining = {"heavy": 1_000.0, "light": 1_000.0}
    rng = random.Random(42)

    counts = {"heavy": 0, "light": 0}
    for _ in range(1000):
        name = pick_weighted_engine(budgets, remaining, rng)
        assert name is not None
        counts[name] += 1

    assert counts["heavy"] > counts["light"] * 3
