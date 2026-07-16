"""Tests for the single-source-of-truth sentence splitter (design §4.5)."""

from __future__ import annotations

from satasr.text.splitting import split_sentences


def test_splits_multiple_sentences() -> None:
    text = "Hello world. How are you? I am fine!"
    assert split_sentences(text) == ["Hello world.", "How are you?", "I am fine!"]


def test_strips_surrounding_whitespace() -> None:
    text = "  Hello world.   \n\n  How are you?  "
    assert split_sentences(text) == ["Hello world.", "How are you?"]


def test_empty_input_yields_no_sentences() -> None:
    assert split_sentences("") == []
    assert split_sentences("   \n\t  ") == []


def test_no_terminal_punctuation_is_one_sentence() -> None:
    assert split_sentences("just some words with no ending") == [
        "just some words with no ending"
    ]


def test_repeated_punctuation_is_a_single_boundary() -> None:
    text = "Wait... really?! Yes."
    assert split_sentences(text) == ["Wait...", "really?!", "Yes."]


def test_collapses_internal_whitespace_runs_between_sentences() -> None:
    text = "First.\n\nSecond.\t\tThird."
    assert split_sentences(text) == ["First.", "Second.", "Third."]
