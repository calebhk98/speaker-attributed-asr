"""Tests for the WikipediaTextSource stub (design §4.5)."""

from __future__ import annotations

import pytest

from satasr.text.registry import TEXT_SOURCES
from satasr.text.wikipedia import WikipediaTextSource


def test_construction_does_not_raise_or_touch_network() -> None:
    source = WikipediaTextSource()  # must not raise, must not hit the network
    assert source is not None


def test_sentences_raises_not_implemented() -> None:
    source = WikipediaTextSource()
    with pytest.raises(NotImplementedError):
        source.sentences()


def test_registered_under_wikipedia() -> None:
    assert "wikipedia" in TEXT_SOURCES.available()
