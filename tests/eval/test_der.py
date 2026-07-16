"""Tests for Diarization Error Rate (design §6)."""

from __future__ import annotations

from satasr.core.models import PlacedUtterance
from satasr.eval.der import der_from_sot, diarization_error_rate
from satasr.format import serialize_utterances


def test_perfect_prediction_has_zero_der() -> None:
    reference = (
        PlacedUtterance("A", 0.0, 5.0, "hello there"),
        PlacedUtterance("B", 3.0, 8.0, "wait what"),
    )
    result = diarization_error_rate(reference, reference)

    assert result.miss_s == 0.0
    assert result.false_alarm_s == 0.0
    assert result.confusion_s == 0.0
    assert result.rate == 0.0


def test_missed_speech_when_hypothesis_says_nothing() -> None:
    reference = (PlacedUtterance("A", 0.0, 10.0, "hello there"),)
    hypothesis: tuple[PlacedUtterance, ...] = ()

    result = diarization_error_rate(reference, hypothesis)

    # Reference speaks for all 10s with no matching hypothesis: 100% miss.
    assert result.miss_s == 10.0
    assert result.false_alarm_s == 0.0
    assert result.confusion_s == 0.0
    assert result.rate == 1.0


def test_false_alarm_when_hypothesis_adds_an_extra_speaker() -> None:
    # Reference: A speaks [0, 10]. Hypothesis correctly transcribes A for the
    # full span, but also hallucinates a second speaker B active [2, 5],
    # a 3s window where the reference has only one active speaker.
    reference = (PlacedUtterance("A", 0.0, 10.0, "hello there"),)
    hypothesis = (
        PlacedUtterance("A", 0.0, 10.0, "hello there"),
        PlacedUtterance("B", 2.0, 5.0, "extra"),
    )

    result = diarization_error_rate(reference, hypothesis)

    assert result.miss_s == 0.0
    assert result.false_alarm_s == 3.0
    assert result.confusion_s == 0.0
    assert result.rate == 3.0 / 10.0


def test_speaker_confusion_when_one_hypothesis_label_merges_two_speakers() -> None:
    # Reference: A speaks [0, 5], C speaks [5, 10] (sequential, non-overlapping).
    # Hypothesis uses a single label X throughout — correct for whichever
    # reference speaker it best overlaps (A, tie broken alphabetically), and a
    # confusion error for the other 5s where it's active but mismatched.
    reference = (
        PlacedUtterance("A", 0.0, 5.0, "hello there"),
        PlacedUtterance("C", 5.0, 10.0, "wait what"),
    )
    hypothesis = (
        PlacedUtterance("X", 0.0, 5.0, "hello there"),
        PlacedUtterance("X", 5.0, 10.0, "wait what"),
    )

    result = diarization_error_rate(reference, hypothesis)

    assert result.miss_s == 0.0
    assert result.false_alarm_s == 0.0
    assert result.confusion_s == 5.0
    assert result.rate == 0.5


def test_der_from_sot_parses_with_the_canonical_parser() -> None:
    reference = (PlacedUtterance("A", 0.0, 5.0, "hello there"),)
    hypothesis = (PlacedUtterance("A", 0.0, 5.0, "hello there"),)

    result = der_from_sot(
        serialize_utterances(reference), serialize_utterances(hypothesis)
    )

    assert result.rate == 0.0


def test_empty_reference_and_hypothesis_is_zero_der() -> None:
    result = diarization_error_rate((), ())

    assert result.total_ref_s == 0.0
    assert result.rate == 0.0
