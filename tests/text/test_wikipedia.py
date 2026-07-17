"""Tests for WikipediaTextSource (design §4.5).

Fast tests inject a fake ``fetch_article`` so they never touch the network,
per the "keep CI offline" rule; the one test that calls the real API is
marked slow and deselected by default.
"""

from __future__ import annotations

import pytest

from satasr.text.registry import TEXT_SOURCES
from satasr.text.splitting import split_sentences
from satasr.text.wikipedia import WikipediaTextSource

# A small sample of what the API's plaintext "extract" actually looks like:
# a heading line and a trailing reference marker survive explaintext=1.
_SAMPLE_ARTICLE = (
    "== Introduction ==\n"
    "The cat is a domestic species of small carnivorous mammal.[1] It is "
    "the only domesticated species in the family Felidae.\n\n"
    "== History ==\n"
    "Cats have lived alongside humans for at least 10,000 years.[2]"
)


def _fake_fetch(_language: str, _title: str) -> str:
    return _SAMPLE_ARTICLE


def test_construction_does_not_raise_or_touch_network() -> None:
    def fetch_that_must_not_be_called(language: str, title: str) -> str:
        raise AssertionError("fetch_article must not run at construction time")

    source = WikipediaTextSource(
        titles=("Cat",), fetch_article=fetch_that_must_not_be_called
    )
    assert source is not None


def test_registered_under_wikipedia() -> None:
    assert "wikipedia" in TEXT_SOURCES.available()


def test_sentences_yields_clean_split_text_from_fixture() -> None:
    source = WikipediaTextSource(titles=("Cat",), fetch_article=_fake_fetch)

    result = list(source.sentences())

    expected = split_sentences(
        "The cat is a domestic species of small carnivorous mammal. It is "
        "the only domesticated species in the family Felidae.\n\n"
        "Cats have lived alongside humans for at least 10,000 years."
    )
    assert result == expected
    assert all("==" not in sentence for sentence in result)
    assert all("[1]" not in sentence and "[2]" not in sentence for sentence in result)


def test_sentences_covers_multiple_titles_in_order() -> None:
    calls: list[str] = []

    def fetch(_language: str, title: str) -> str:
        calls.append(title)
        return "First sentence about the topic. Second sentence."

    source = WikipediaTextSource(titles=("Cat", "Dog"), fetch_article=fetch)

    result = list(source.sentences())

    assert calls == ["Cat", "Dog"]
    assert result == [
        "First sentence about the topic.",
        "Second sentence.",
        "First sentence about the topic.",
        "Second sentence.",
    ]


def test_sentences_is_lazy_until_iterated() -> None:
    def fetch_that_must_not_run(_language: str, _title: str) -> str:
        raise AssertionError("fetch_article must not run before iteration")

    source = WikipediaTextSource(titles=("Cat",), fetch_article=fetch_that_must_not_run)

    iterator = source.sentences()  # building the generator must not fetch

    with pytest.raises(AssertionError):
        next(iterator)  # only *consuming* it triggers the fetch


def test_sentences_returns_fresh_iterator_each_call() -> None:
    source = WikipediaTextSource(titles=("Cat",), fetch_article=_fake_fetch)

    first_pass = list(source.sentences())
    second_pass = list(source.sentences())

    assert first_pass == second_pass
    assert first_pass  # the fixture is non-empty, so this is a real check


@pytest.mark.slow
def test_sentences_fetches_a_real_article() -> None:
    source = WikipediaTextSource(titles=("Cat",))

    result = list(source.sentences())

    assert len(result) > 1
