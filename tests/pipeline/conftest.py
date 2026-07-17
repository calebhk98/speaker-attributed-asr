"""Shared fixtures for pipeline tests: a dependency-free engine registry.

Tests build their own :class:`Registry` (rather than using the global
``TTS_ENGINES``) so several engine *names* can be exercised in one test run
without leaking registrations across tests or depending on which real engines
happen to import successfully in this environment.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from satasr.core.interfaces import TTSEngine
from satasr.core.registry import Registry
from satasr.tts.engines.sine import SineTTSEngine


def _make_sine_registry(names: Sequence[str]) -> Registry[TTSEngine]:
    """A fresh registry with the dependency-free ``sine`` engine under each name."""
    registry: Registry[TTSEngine] = Registry("test engine")
    for name in names:
        registry.register(name)(SineTTSEngine)
    return registry


@pytest.fixture
def sine_registry() -> Callable[[Sequence[str]], Registry[TTSEngine]]:
    """Factory fixture: call with engine names to get a registry of sine clones."""
    return _make_sine_registry
