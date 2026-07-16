"""Tests for PlainFileTextSource (design §4.5)."""

from __future__ import annotations

from pathlib import Path

from satasr.text.plain_file import PlainFileTextSource
from satasr.text.registry import TEXT_SOURCES


def test_reads_and_splits_file_contents(tmp_path: Path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_text("Hello world. How are you?", encoding="utf-8")

    source = PlainFileTextSource(path)

    assert list(source.sentences()) == ["Hello world.", "How are you?"]


def test_does_not_read_at_construction(tmp_path: Path) -> None:
    path = tmp_path / "missing.txt"  # never created

    PlainFileTextSource(path)  # must not raise

    assert not path.exists()


def test_sentences_can_be_iterated_twice(tmp_path: Path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_text("One. Two.", encoding="utf-8")
    source = PlainFileTextSource(path)

    assert list(source.sentences()) == list(source.sentences())


def test_registered_under_plain_file(tmp_path: Path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_text("Registered fine.", encoding="utf-8")

    assert "plain_file" in TEXT_SOURCES.available()
    source = TEXT_SOURCES.create("plain_file", path=path)
    assert list(source.sentences()) == ["Registered fine."]
