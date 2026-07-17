"""Tests for the canonical SOT transcript format (design §4.7)."""

from __future__ import annotations

import numpy as np
import pytest

from satasr.core.audio import AudioBuffer
from satasr.core.models import MixedClip, PlacedUtterance, SpeakerCounts
from satasr.format.serialization import (
    parse_utterances,
    serialize_clip,
    serialize_utterances,
)


def _clip(utterances: tuple[PlacedUtterance, ...]) -> MixedClip:
    audio = AudioBuffer(np.zeros(10, dtype=np.float32))
    counts = SpeakerCounts(total=2, max_simultaneous=2)
    return MixedClip(audio=audio, utterances=utterances, counts=counts)


def test_round_trip_single_utterance() -> None:
    original = (PlacedUtterance("A", 0.0, 1.5, "hello there"),)
    parsed = parse_utterances(serialize_utterances(original))
    assert parsed == original


def test_round_trip_preserves_timestamp_precision() -> None:
    # Not a round decimal: exercises str(float)'s exact round-trip guarantee.
    original = (PlacedUtterance("A", 0.123456789, 12.3456789, "hi"),)
    parsed = parse_utterances(serialize_utterances(original))
    assert parsed[0].start_s == original[0].start_s
    assert parsed[0].end_s == original[0].end_s


def test_truncated_utterance_keeps_marker_through_round_trip() -> None:
    original = (PlacedUtterance("A", 0.0, 0.6, "the -", truncated=True),)
    sot = serialize_utterances(original)
    assert sot.endswith(" -")

    parsed = parse_utterances(sot)
    assert parsed == original
    assert parsed[0].truncated


def test_whole_word_truncation_without_marker_is_not_flagged_on_parse() -> None:
    # No marker was appended (cut landed on a word boundary): the string can't
    # distinguish this from a naturally short utterance, by design (see the
    # module docstring's truncation contract).
    original = PlacedUtterance("A", 0.0, 0.9, "the lions", truncated=True)
    parsed = parse_utterances(serialize_utterances((original,)))
    assert not parsed[0].truncated


def test_overlapping_speakers_ordered_deterministically() -> None:
    # Out-of-chronological-order input; expect sort by (start, end, speaker).
    # a and c tie on start_s=0.0, so c (shorter, end=1.0) sorts before a
    # (end=1.5); b starts later still.
    b = PlacedUtterance("B", 0.5, 2.0, "wait what")
    a = PlacedUtterance("A", 0.0, 1.5, "hello there")
    c = PlacedUtterance("C", 0.0, 1.0, "also here")
    parsed = parse_utterances(serialize_utterances((b, a, c)))
    assert [u.speaker_id for u in parsed] == ["C", "A", "B"]


def test_full_tie_breaks_on_speaker_id() -> None:
    # Same start_s and end_s: tie-break alphabetically on speaker_id.
    z = PlacedUtterance("Z", 0.0, 1.0, "zed")
    a = PlacedUtterance("A", 0.0, 1.0, "ay")
    parsed = parse_utterances(serialize_utterances((z, a)))
    assert [u.speaker_id for u in parsed] == ["A", "Z"]


def test_serialize_clip_matches_serialize_utterances() -> None:
    utterances = (PlacedUtterance("A", 0.0, 1.0, "hi"),)
    clip = _clip(utterances)
    assert serialize_clip(clip) == serialize_utterances(utterances)


def test_empty_utterances_round_trip_to_empty_string() -> None:
    assert serialize_utterances(()) == ""
    assert parse_utterances("") == ()


def test_multiple_lines_are_newline_joined() -> None:
    utterances = (
        PlacedUtterance("A", 0.0, 1.0, "hi"),
        PlacedUtterance("B", 0.5, 1.5, "yo"),
    )
    sot = serialize_utterances(utterances)
    assert sot.count("\n") == 1
    assert sot.splitlines()[0].startswith("<A> ")
    assert sot.splitlines()[1].startswith("<B> ")


def test_parsed_words_are_always_empty() -> None:
    # Word-level timestamps are not part of the SOT string (utterance-level
    # target only, per design §1's example output).
    sot = serialize_utterances((PlacedUtterance("A", 0.0, 1.0, "hi"),))
    assert parse_utterances(sot)[0].words == ()


def test_negative_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        serialize_utterances((PlacedUtterance("A", -1.0, 1.0, "hi"),))


def test_speaker_id_with_delimiter_is_rejected() -> None:
    with pytest.raises(ValueError, match="speaker_id"):
        serialize_utterances((PlacedUtterance("A>B", 0.0, 1.0, "hi"),))


def test_malformed_line_raises_on_parse() -> None:
    with pytest.raises(ValueError, match="malformed SOT line"):
        parse_utterances("not a valid line")
