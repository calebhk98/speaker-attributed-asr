"""Tests for auto-discovery of engine modules.

Guards the property the 20 parallel engines rely on: a module dropped into the
engines package registers itself with no shared-file edit, and one bad engine
does not break discovery of the others.
"""

from __future__ import annotations

import importlib

from satasr.tts import engines
from satasr.tts.registry import TTS_ENGINES


def test_reference_engine_is_auto_discovered() -> None:
    # `sine` is registered purely by living in the engines package.
    assert "sine" in TTS_ENGINES.available()


def test_discovery_is_resilient_to_a_broken_module(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def _explode(name: str) -> None:
        raise ImportError(f"pretend {name} has an uninstalled dependency")

    # Even if importing a module blows up, re-running discovery must not raise.
    monkeypatch.setattr(importlib, "import_module", _explode)
    importlib.reload(engines)  # should swallow the error, not propagate it
