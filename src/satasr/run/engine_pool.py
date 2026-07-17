"""Pick and cache TTS engines by configured weight (design §4.3).

A run assigns each speaker an engine drawn in proportion to the configured
weights, then reuses one instance per engine. Weighting logic stays trivial and
local here; the engine roster itself comes from ``TTS_ENGINES`` (single source
of truth for which engines exist).
"""

from __future__ import annotations

import random

from satasr.core.interfaces import TTSEngine
from satasr.tts import TTS_ENGINES


class EnginePool:
    """Weighted engine picker with a one-instance-per-engine cache."""

    def __init__(self, weights: dict[str, float]) -> None:
        self._weights = {name: weight for name, weight in weights.items() if weight > 0}
        if not self._weights:
            raise ValueError("no engines configured with a positive weight")
        self._cache: dict[str, TTSEngine] = {}

    def pick(self, rng: random.Random) -> TTSEngine:
        """Draw an engine name by weight and return its cached instance."""
        names = list(self._weights)
        weights = [self._weights[name] for name in names]
        chosen = str(rng.choices(names, weights=weights, k=1)[0])
        return self._cache.setdefault(chosen, TTS_ENGINES.create(chosen))
