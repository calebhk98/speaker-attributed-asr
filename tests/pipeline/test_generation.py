"""Tests for GenerationOrchestrator using only sine + static (§4.2/§4.3).

These are the fast tests the issue asks for: allocation respects configured
engine weights, generation stops once an engine's hour budget is met, and
reweighting the same config changes which engine dominates the mix.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Sequence

from satasr.core.config import GenerationConfig
from satasr.core.interfaces import TTSEngine, VoiceReference
from satasr.core.registry import Registry
from satasr.pipeline.generation import GenerationOrchestrator
from satasr.pipeline.state import GenerationState
from satasr.text.static import StaticTextSource

RegistryFactory = Callable[[Sequence[str]], Registry[TTSEngine]]

_VOICES = (VoiceReference(speaker_id="narrator", preset="calm"),)
_SENTENCE = "the quick fox runs"  # fixed length -> deterministic clip duration


def test_generation_stops_once_the_hour_budget_is_met(
    sine_registry: RegistryFactory,
) -> None:
    registry = sine_registry(["engine"])
    # A tiny budget forces completion after only a handful of short clips.
    config = GenerationConfig(hours_per_engine={"engine": 0.0005})
    orchestrator = GenerationOrchestrator(config, registry, rng=random.Random(0))
    text_source = StaticTextSource([_SENTENCE])
    state = GenerationState()

    clips = list(orchestrator.generate(text_source, _VOICES, state))

    assert clips  # some audio was generated before stopping
    (budget,) = orchestrator.budgets
    assert state.hours_generated["engine"] >= budget.target_hours
    # Overshoot is bounded by at most one extra clip's worth of audio.
    one_clip_hours = clips[0].clip.audio.duration_s / 3600.0
    assert state.hours_generated["engine"] < budget.target_hours + one_clip_hours + 1e-9
    assert state.is_complete(orchestrator.budgets)


def test_allocation_respects_configured_engine_weights(
    sine_registry: RegistryFactory,
) -> None:
    registry = sine_registry(["heavy", "light"])
    # Budgets far larger than the sampled prefix so neither engine completes
    # mid-sample — only the configured weight should drive the mix.
    config = GenerationConfig(hours_per_engine={"heavy": 8_000.0, "light": 2_000.0})
    orchestrator = GenerationOrchestrator(config, registry, rng=random.Random(42))
    text_source = StaticTextSource([_SENTENCE])

    sample = list(itertools.islice(orchestrator.generate(text_source, _VOICES), 500))

    counts = {"heavy": 0, "light": 0}
    for generated in sample:
        counts[generated.engine_name] += 1
    assert counts["heavy"] > counts["light"] * 2


def test_reweighting_flips_which_engine_dominates(
    sine_registry: RegistryFactory,
) -> None:
    registry = sine_registry(["a", "b"])
    text_source = StaticTextSource([_SENTENCE])

    a_heavy = GenerationConfig(hours_per_engine={"a": 9_000.0, "b": 1_000.0})
    b_heavy = GenerationConfig(hours_per_engine={"a": 1_000.0, "b": 9_000.0})

    def dominant_engine(config: GenerationConfig) -> str:
        orchestrator = GenerationOrchestrator(config, registry, rng=random.Random(7))
        sample = itertools.islice(orchestrator.generate(text_source, _VOICES), 300)
        counts = {"a": 0, "b": 0}
        for generated in sample:
            counts[generated.engine_name] += 1
        return max(counts, key=lambda name: counts[name])

    # Config alone flips the mix — no code change between the two runs.
    assert dominant_engine(a_heavy) == "a"
    assert dominant_engine(b_heavy) == "b"


def test_generation_resumes_from_prior_state(sine_registry: RegistryFactory) -> None:
    registry = sine_registry(["engine"])
    config = GenerationConfig(hours_per_engine={"engine": 0.001})
    text_source = StaticTextSource([_SENTENCE])

    first_run = GenerationOrchestrator(config, registry, rng=random.Random(1))
    state = GenerationState()
    list(first_run.generate(text_source, _VOICES, state))
    assert state.is_complete(first_run.budgets)

    hours_after_first_run = dict(state.hours_generated)

    # Resuming an already-complete state must not synthesize any more audio.
    second_run = GenerationOrchestrator(config, registry, rng=random.Random(2))
    more_clips = list(second_run.generate(text_source, _VOICES, state))

    assert more_clips == []
    assert state.hours_generated == hours_after_first_run
