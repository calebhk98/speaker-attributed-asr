"""Tests that importing satasr.text registers the full built-in roster."""

from __future__ import annotations

from satasr.text import TEXT_SOURCES


def test_all_built_in_sources_are_registered() -> None:
    # Containment, not equality: adding a new source must not break this test.
    registered = set(TEXT_SOURCES.available())
    assert {"plain_file", "static", "wikipedia", "llm"} <= registered
