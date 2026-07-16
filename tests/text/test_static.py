"""Tests for StaticTextSource (design §4.5)."""

from __future__ import annotations

from satasr.text.registry import TEXT_SOURCES
from satasr.text.static import StaticTextSource


def test_round_trips_given_sentences() -> None:
    source = StaticTextSource(["First.", "Second."])
    assert list(source.sentences()) == ["First.", "Second."]


def test_sentences_can_be_iterated_twice() -> None:
    source = StaticTextSource(["First.", "Second."])
    assert list(source.sentences()) == list(source.sentences())


def test_registered_under_static() -> None:
    assert "static" in TEXT_SOURCES.available()
    source = TEXT_SOURCES.create("static", sentences=["Only one."])
    assert list(source.sentences()) == ["Only one."]
