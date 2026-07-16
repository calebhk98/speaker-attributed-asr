"""Tests for the generic plugin registry."""

from __future__ import annotations

import pytest

from satasr.core.registry import Registry


def _fresh() -> Registry[str]:
    return Registry[str]("widget")


def test_register_and_create() -> None:
    registry = _fresh()

    @registry.register("greeter")
    def _make(name: str = "world") -> str:
        return f"hello {name}"

    assert registry.create("greeter") == "hello world"
    assert registry.create("greeter", name="bob") == "hello bob"


def test_available_is_sorted() -> None:
    registry = _fresh()
    registry.register("b")(lambda: "b")
    registry.register("a")(lambda: "a")
    assert registry.available() == ["a", "b"]


def test_duplicate_name_is_rejected() -> None:
    registry = _fresh()
    registry.register("dup")(lambda: "x")
    with pytest.raises(ValueError, match="already registered"):
        registry.register("dup")(lambda: "y")


def test_unknown_name_lists_options() -> None:
    registry = _fresh()
    registry.register("known")(lambda: "x")
    with pytest.raises(KeyError, match="known"):
        registry.create("missing")
