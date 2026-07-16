"""Tests for cutting an utterance short at a word boundary (design §4.7)."""

from __future__ import annotations

from satasr.core.models import Word
from satasr.mixing.truncation import truncate_words

WORDS = (
    Word("the", 0.0, 0.4),
    Word("lions", 0.4, 0.9),
    Word("are", 0.9, 1.2),
    Word("hungry", 1.2, 1.8),
)


def test_no_cut_keeps_everything() -> None:
    result = truncate_words(WORDS, cut_s=5.0, marker="-")
    assert result.text == "the lions are hungry"
    assert not result.truncated
    assert not result.mid_word


def test_cut_on_word_boundary_drops_later_words() -> None:
    # Cut exactly at 0.9s: "the lions" end at/by 0.9, later words dropped.
    result = truncate_words(WORDS, cut_s=0.9, marker="-")
    assert result.text == "the lions"
    assert result.truncated
    assert not result.mid_word


def test_cut_mid_word_drops_partial_word_and_marks() -> None:
    # Cut at 0.6s lands inside "lions" (0.4-0.9): drop it, append marker.
    result = truncate_words(WORDS, cut_s=0.6, marker="-")
    assert result.text == "the -"
    assert result.truncated
    assert result.mid_word


def test_cut_before_first_word_yields_only_marker() -> None:
    result = truncate_words(WORDS, cut_s=0.2, marker="-")
    assert result.text == "-"
    assert result.truncated
    assert result.mid_word
