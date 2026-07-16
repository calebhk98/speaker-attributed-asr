"""Tests for LlmTextSource (design §4.5)."""

from __future__ import annotations

import pytest

from satasr.text.llm import LlmProvider, LlmTextSource
from satasr.text.registry import TEXT_SOURCES


class _StubProvider:
    """A deterministic, call-counting stand-in for a real LlmProvider."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return self.text


class _RecordingProvider:
    """Captures every prompt it is asked to generate from."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Ok."


def test_construction_does_not_generate_or_touch_network() -> None:
    provider = _StubProvider("unused")

    LlmTextSource(provider=provider)  # must not raise, must not call generate

    assert provider.calls == 0


def test_sentences_splits_generated_text() -> None:
    provider = _StubProvider("Hello world. How are you?")
    source = LlmTextSource(provider=provider)

    assert list(source.sentences()) == ["Hello world.", "How are you?"]


def test_sentences_generates_fresh_each_call() -> None:
    provider = _StubProvider("One. Two.")
    source = LlmTextSource(provider=provider)

    list(source.sentences())
    list(source.sentences())

    assert provider.calls == 2


def test_prompt_is_passed_through_to_the_provider() -> None:
    provider = _RecordingProvider()
    source = LlmTextSource(prompt="custom prompt", provider=provider)

    list(source.sentences())

    assert provider.prompts == ["custom prompt"]


def test_stub_provider_satisfies_llm_provider_protocol() -> None:
    assert isinstance(_StubProvider("x"), LlmProvider)


def test_registered_under_llm() -> None:
    assert "llm" in TEXT_SOURCES.available()
    provider = _StubProvider("Registered fine.")

    source = TEXT_SOURCES.create("llm", provider=provider)

    assert list(source.sentences()) == ["Registered fine."]


@pytest.mark.slow
def test_default_provider_generates_real_text() -> None:
    source = LlmTextSource()

    assert list(source.sentences())
