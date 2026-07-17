"""Data Stage 1: the generation orchestrator (design §4.2/§4.3).

Drives synthetic-data generation toward the 10k-hour budget: pick a sentence,
pick an engine by its configured weight, synthesize, repeat until every
engine's hour budget is met. It depends only on the ``TTSEngine`` protocol via
a ``Registry`` (never a concrete engine), so the roster of engines is entirely
config-driven — see ``satasr.pipeline.allocation`` for the weighting.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from satasr.core.config import GenerationConfig
from satasr.core.interfaces import TextSource, TTSEngine, VoiceReference
from satasr.core.models import SpeakerClip
from satasr.core.registry import Registry
from satasr.pipeline.allocation import (
    EngineBudget,
    pick_weighted_engine,
    resolve_budgets,
)
from satasr.pipeline.state import GenerationState
from satasr.pipeline.text_feed import endless_sentences
from satasr.tts.registry import TTS_ENGINES

_SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class GeneratedClip:
    """One synthesized clip plus the engine that produced it."""

    engine_name: str
    clip: SpeakerClip


class GenerationOrchestrator:
    """Drives generation across engines per ``GenerationConfig`` (§4.2/§4.3).

    Engines are looked up by name in ``registry`` (default: the real
    ``TTS_ENGINES``) — a fresh engine is used purely by registering it and,
    optionally, giving it an entry in ``GenerationConfig.hours_per_engine``.
    No engine name is hardcoded here.
    """

    def __init__(
        self,
        config: GenerationConfig,
        registry: Registry[TTSEngine] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._config = config
        self._registry = registry if registry is not None else TTS_ENGINES
        self._rng = rng if rng is not None else random.Random()
        self._budgets = resolve_budgets(config, self._registry)

    @property
    def budgets(self) -> tuple[EngineBudget, ...]:
        """The resolved per-engine hour targets this run is driving toward."""
        return self._budgets

    def generate(
        self,
        text_source: TextSource,
        voices: Sequence[VoiceReference],
        state: GenerationState | None = None,
    ) -> Iterator[GeneratedClip]:
        """Yield clips until every engine's budget is met or text runs out.

        Pass a prior run's ``state`` to resume it (§4.2): already-generated
        hours are honored and generation continues from there.
        """
        if not voices:
            raise ValueError("need at least one VoiceReference to synthesize with")

        state = state if state is not None else GenerationState()
        engines: dict[str, TTSEngine] = {}
        voice_cycle = itertools.cycle(voices)

        for sentence in endless_sentences(text_source):
            engine_name = self._next_engine(state)
            if engine_name is None:
                return
            engine = engines.setdefault(engine_name, self._registry.create(engine_name))
            clip = engine.synthesize(sentence, next(voice_cycle))
            state.record(engine_name, clip.audio.duration_s / _SECONDS_PER_HOUR)
            yield GeneratedClip(engine_name, clip)

    def _next_engine(self, state: GenerationState) -> str | None:
        remaining = state.remaining(self._budgets)
        return pick_weighted_engine(self._budgets, remaining, self._rng)
