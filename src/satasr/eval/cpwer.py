"""Concatenated (time-constrained) minimum-permutation WER (design §6, §7).

cpWER and tcpWER are the multi-speaker ASR metrics used by the DiCoW /
Sortformer research this project targets. Both score predicted vs. reference
:class:`~satasr.core.models.PlacedUtterance` sequences while being invariant
to which hypothesis speaker label lines up with which reference speaker
label: the correct pairing is found by trying every permutation of
hypothesis-to-reference speakers and keeping the cheapest one (§6).

* ``cpwer`` concatenates each speaker's words (in start-time order) and
  scores the pairing with plain word-level Levenshtein distance — text only,
  no timing.
* ``tcpwer`` (time-constrained cpWER) additionally requires a candidate
  word-to-word match to fall within ``collar_s`` of each other in time before
  it is credited as correct; a hypothesis word that is textually right but
  shifted in time still counts as an error. This is the one difference
  between the two metrics, and why a time-shifted hypothesis is penalised
  under ``tcpwer`` but not ``cpwer``.

Pure functions, no model dependency. Parse serialized-output strings with
``satasr.format.parse_utterances`` (the single source of truth, CLAUDE.md
rule 4) before calling these — do not hand-roll another SOT parser here.

The speaker-to-speaker pairing is found by brute-force permutation search,
which is fine for the speaker counts this project targets (§4.6 caps
simultaneous speakers at 6); it is not the algorithm to reach for if a
deployment needs to score dozens of speakers per clip.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations

from satasr.core.models import PlacedUtterance

DEFAULT_COLLAR_S = 0.0


@dataclass(frozen=True)
class _TimedWord:
    """One word token with the absolute time span it occupies."""

    text: str
    start_s: float
    end_s: float


def cpwer(
    reference: Sequence[PlacedUtterance], hypothesis: Sequence[PlacedUtterance]
) -> float:
    """Concatenated minimum-permutation WER: text-only, ignores timing (§6)."""
    return _score(reference, hypothesis, collar_s=None)


def tcpwer(
    reference: Sequence[PlacedUtterance],
    hypothesis: Sequence[PlacedUtterance],
    collar_s: float = DEFAULT_COLLAR_S,
) -> float:
    """Time-constrained cpWER: a word only counts as correct if its timing
    also overlaps (within ``collar_s``) the reference word it aligns to (§6)."""
    return _score(reference, hypothesis, collar_s=collar_s)


def _score(
    reference: Sequence[PlacedUtterance],
    hypothesis: Sequence[PlacedUtterance],
    collar_s: float | None,
) -> float:
    ref_streams = _speaker_streams(reference)
    hyp_streams = _speaker_streams(hypothesis)
    n_ref_words = sum(len(words) for words in ref_streams.values())
    if n_ref_words == 0:
        n_hyp_words = sum(len(words) for words in hyp_streams.values())
        return 0.0 if n_hyp_words == 0 else 1.0

    best = min(
        _permutation_errors(ref_streams, hyp_streams, mapping, collar_s)
        for mapping in _speaker_mappings(ref_streams, hyp_streams)
    )
    return best / n_ref_words


def _speaker_streams(
    utterances: Sequence[PlacedUtterance],
) -> dict[str, list[_TimedWord]]:
    """Group utterances by speaker, each speaker's words in start-time order."""
    streams: dict[str, list[_TimedWord]] = {}
    for u in sorted(utterances, key=lambda u: (u.start_s, u.end_s)):
        streams.setdefault(u.speaker_id, []).extend(_words(u))
    return streams


def _words(u: PlacedUtterance) -> list[_TimedWord]:
    """Per-word timed tokens for one utterance.

    Uses ``u.words`` (absolute-time, aligner-produced) when present. SOT
    strings carry only utterance-level timestamps (design §4.7), so parsed
    utterances always have ``words=()``; approximate per-word timing there by
    splitting the utterance's span evenly across its tokens.
    """
    if u.words:
        return [_TimedWord(w.text, w.start_s, w.end_s) for w in u.words]
    tokens = u.text.split()
    if not tokens:
        return []
    span = (u.end_s - u.start_s) / len(tokens)
    return [
        _TimedWord(tok, u.start_s + i * span, u.start_s + (i + 1) * span)
        for i, tok in enumerate(tokens)
    ]


def _speaker_mappings(
    ref_streams: Mapping[str, list[_TimedWord]],
    hyp_streams: Mapping[str, list[_TimedWord]],
) -> list[dict[str | None, str | None]]:
    """Every way to pair hypothesis speakers up with reference speakers.

    Padded with ``None`` on whichever side has fewer speakers, so an unmatched
    reference speaker scores as all deletions and an unmatched hypothesis
    speaker as all insertions.
    """
    ref_ids: list[str | None] = list(ref_streams)
    hyp_ids: list[str | None] = list(hyp_streams)
    hyp_ids += [None] * max(0, len(ref_ids) - len(hyp_ids))
    ref_ids += [None] * max(0, len(hyp_ids) - len(ref_ids))
    return [dict(zip(ref_ids, perm, strict=True)) for perm in permutations(hyp_ids)]


def _permutation_errors(
    ref_streams: Mapping[str, list[_TimedWord]],
    hyp_streams: Mapping[str, list[_TimedWord]],
    mapping: Mapping[str | None, str | None],
    collar_s: float | None,
) -> int:
    total = 0
    for ref_id, hyp_id in mapping.items():
        ref_words = ref_streams.get(ref_id, []) if ref_id is not None else []
        hyp_words = hyp_streams.get(hyp_id, []) if hyp_id is not None else []
        total += _edit_distance(ref_words, hyp_words, collar_s)
    return total


def _edit_distance(
    ref: Sequence[_TimedWord], hyp: Sequence[_TimedWord], collar_s: float | None
) -> int:
    """Word-level Levenshtein distance, single row of dynamic programming."""
    prev = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        curr = [i] + [0] * len(hyp)
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if _is_match(ref_word, hyp_word, collar_s) else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[len(hyp)]


def _is_match(ref: _TimedWord, hyp: _TimedWord, collar_s: float | None) -> bool:
    """Whether ``hyp`` may be credited as a correct match for ``ref``."""
    if ref.text != hyp.text:
        return False
    if collar_s is None:
        return True
    return hyp.start_s <= ref.end_s + collar_s and hyp.end_s >= ref.start_s - collar_s
