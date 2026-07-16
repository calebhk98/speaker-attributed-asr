"""Tests for cpWER / tcpWER (design §6)."""

from __future__ import annotations

from satasr.core.models import PlacedUtterance
from satasr.eval.cpwer import cpwer, tcpwer
from satasr.format import parse_utterances, serialize_utterances


def test_perfect_prediction_scores_zero() -> None:
    reference = (
        PlacedUtterance("A", 0.0, 1.0, "hello there"),
        PlacedUtterance("B", 1.0, 2.0, "how are you"),
    )
    hypothesis = reference
    assert cpwer(reference, hypothesis) == 0.0
    assert tcpwer(reference, hypothesis) == 0.0


def test_perfect_prediction_round_tripped_through_sot_scores_zero() -> None:
    # Exercises the parse_utterances path (words=()), not just live objects.
    reference = (
        PlacedUtterance("A", 0.0, 1.0, "hello there"),
        PlacedUtterance("B", 1.2, 2.4, "how are you"),
    )
    hypothesis = parse_utterances(serialize_utterances(reference))
    assert cpwer(reference, hypothesis) == 0.0
    assert tcpwer(reference, hypothesis) == 0.0


def test_speaker_swapped_output_still_scores_zero_via_best_permutation() -> None:
    reference = (
        PlacedUtterance("A", 0.0, 1.0, "hello there"),
        PlacedUtterance("B", 1.0, 2.0, "how are you"),
    )
    # Labels swapped relative to reference, content/timing otherwise identical.
    hypothesis = (
        PlacedUtterance("B", 0.0, 1.0, "hello there"),
        PlacedUtterance("A", 1.0, 2.0, "how are you"),
    )
    assert cpwer(reference, hypothesis) == 0.0
    assert tcpwer(reference, hypothesis) == 0.0


def test_time_shifted_match_penalized_under_tcpwer_but_not_cpwer() -> None:
    reference = (PlacedUtterance("A", 0.0, 1.0, "hello there"),)
    # Same speaker, same text, but shifted far outside any reasonable collar.
    hypothesis = (PlacedUtterance("A", 10.0, 11.0, "hello there"),)
    assert cpwer(reference, hypothesis) == 0.0
    assert tcpwer(reference, hypothesis) > 0.0


def test_tcpwer_tolerates_shift_within_collar() -> None:
    reference = (PlacedUtterance("A", 0.0, 1.0, "hello there"),)
    hypothesis = (PlacedUtterance("A", 0.2, 1.2, "hello there"),)
    assert tcpwer(reference, hypothesis, collar_s=0.5) == 0.0


def test_substitution_counts_as_one_error_per_word() -> None:
    reference = (PlacedUtterance("A", 0.0, 1.0, "hello there friend"),)
    hypothesis = (PlacedUtterance("A", 0.0, 1.0, "hello there enemy"),)
    assert cpwer(reference, hypothesis) == 1 / 3


def test_extra_hypothesis_speaker_is_all_insertions() -> None:
    reference = (PlacedUtterance("A", 0.0, 1.0, "hello there"),)
    hypothesis = (
        PlacedUtterance("A", 0.0, 1.0, "hello there"),
        PlacedUtterance("Z", 5.0, 6.0, "extra words here"),
    )
    assert cpwer(reference, hypothesis) == 3 / 2


def test_missing_hypothesis_speaker_is_all_deletions() -> None:
    reference = (
        PlacedUtterance("A", 0.0, 1.0, "hello there"),
        PlacedUtterance("B", 1.0, 2.0, "how are you"),
    )
    hypothesis = (PlacedUtterance("A", 0.0, 1.0, "hello there"),)
    assert cpwer(reference, hypothesis) == 3 / 5


def test_empty_reference_and_hypothesis_scores_zero() -> None:
    assert cpwer((), ()) == 0.0
    assert tcpwer((), ()) == 0.0


def test_empty_reference_with_nonempty_hypothesis_scores_one() -> None:
    hypothesis = (PlacedUtterance("A", 0.0, 1.0, "hello"),)
    assert cpwer((), hypothesis) == 1.0
