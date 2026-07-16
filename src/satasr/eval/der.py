"""Diarization Error Rate (DER) — design §6.

DER scores predicted vs. reference speaker timelines: miss (reference speech
with no matching hypothesis speaker), false alarm (hypothesis speech with no
matching reference speaker), and speaker confusion (both sides active but
mapped to the wrong identity). This is the standard NIST/pyannote definition,
computed here directly from exact utterance intervals rather than sampled
frames, so timing is exact.

Pure function, no model dependency: it only reads
:class:`~satasr.core.models.PlacedUtterance`. Callers scoring serialized model
output must parse it first with :func:`satasr.format.parse_utterances` (the
single source of truth for the SOT grammar, CLAUDE.md rule 4) — this module
never re-parses the string itself, except in the :func:`der_from_sot`
convenience wrapper, which simply delegates to that parser.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from satasr.core.models import PlacedUtterance
from satasr.format import parse_utterances


@dataclass(frozen=True)
class DERResult:
    """The three error components plus the total reference speaker-time they
    are normalized by (design §6). ``rate`` is the DER itself."""

    miss_s: float
    false_alarm_s: float
    confusion_s: float
    total_ref_s: float

    @property
    def rate(self) -> float:
        error_s = self.miss_s + self.false_alarm_s + self.confusion_s
        if self.total_ref_s == 0:
            return 0.0 if error_s == 0 else float("inf")
        return error_s / self.total_ref_s


def der_from_sot(reference_sot: str, hypothesis_sot: str) -> DERResult:
    """Convenience wrapper: parse two SOT strings and score them.

    Uses ``satasr.format.parse_utterances`` — the single source of truth for
    the grammar — rather than growing a second parser here.
    """
    return diarization_error_rate(
        parse_utterances(reference_sot), parse_utterances(hypothesis_sot)
    )


def diarization_error_rate(
    reference: Iterable[PlacedUtterance], hypothesis: Iterable[PlacedUtterance]
) -> DERResult:
    """Compute DER of ``hypothesis`` against ``reference``."""
    ref = [u for u in reference if u.end_s > u.start_s]
    hyp = [u for u in hypothesis if u.end_s > u.start_s]
    total_ref_s = sum(u.end_s - u.start_s for u in ref)

    mapping = _optimal_mapping(_overlap_by_speaker(ref, hyp))
    miss_s, false_alarm_s, confusion_s = _score_timeline(ref, hyp, mapping)
    return DERResult(miss_s, false_alarm_s, confusion_s, total_ref_s)


def _overlap_by_speaker(
    ref: list[PlacedUtterance], hyp: list[PlacedUtterance]
) -> dict[tuple[str, str], float]:
    """Total overlap seconds for every (ref_speaker, hyp_speaker) pair,
    summed across all of their utterances."""
    overlap: dict[tuple[str, str], float] = {}
    for r, h in itertools.product(ref, hyp):
        seconds = max(0.0, min(r.end_s, h.end_s) - max(r.start_s, h.start_s))
        if seconds <= 0:
            continue
        key = (r.speaker_id, h.speaker_id)
        overlap[key] = overlap.get(key, 0.0) + seconds
    return overlap


def _optimal_mapping(overlap: dict[tuple[str, str], float]) -> dict[str, str]:
    """Hyp-speaker -> ref-speaker mapping maximizing total overlap (§6).

    Exact DER scoring needs an optimal one-to-one mapping (a rectangular
    assignment problem). The design caps concurrent speakers at a handful
    (§1: 1-4 talkers), so a full permutation search over the smaller side is
    exact and fast without a Hungarian-algorithm dependency — pyproject.toml
    is owned by a concurrent task, so no new dependency can be added here.
    """
    ref_ids = sorted({r for r, _ in overlap})
    hyp_ids = sorted({h for _, h in overlap})
    if not ref_ids or not hyp_ids:
        return {}

    if len(hyp_ids) <= len(ref_ids):
        best = _best_permutation(
            hyp_ids, ref_ids, lambda h, r: overlap.get((r, h), 0.0)
        )
        return dict(zip(hyp_ids, best, strict=True))
    best = _best_permutation(ref_ids, hyp_ids, lambda r, h: overlap.get((r, h), 0.0))
    return {h: r for r, h in zip(ref_ids, best, strict=True)}


def _best_permutation(
    keys: list[str],
    candidates: list[str],
    score_fn: Callable[[str, str], float],
) -> tuple[str, ...]:
    """The assignment of ``candidates`` (taken ``len(keys)`` at a time) to
    ``keys``, in order, maximizing ``sum(score_fn(key, candidate))``."""
    best_score, best_perm = -1.0, tuple(candidates[: len(keys)])
    for perm in itertools.permutations(candidates, len(keys)):
        score = sum(score_fn(k, c) for k, c in zip(keys, perm, strict=True))
        if score > best_score:
            best_score, best_perm = score, perm
    return best_perm


def _active_speakers(
    utterances: list[PlacedUtterance], t0: float, t1: float
) -> set[str]:
    """Speaker ids active throughout the sub-interval (t0, t1)."""
    return {u.speaker_id for u in utterances if u.start_s <= t0 and u.end_s >= t1}


def _score_timeline(
    ref: list[PlacedUtterance], hyp: list[PlacedUtterance], mapping: dict[str, str]
) -> tuple[float, float, float]:
    """Sum miss/false-alarm/confusion seconds over the timeline partition
    induced by every ref/hyp utterance boundary (§6)."""
    boundaries = sorted(
        {u.start_s for u in (*ref, *hyp)} | {u.end_s for u in (*ref, *hyp)}
    )
    miss_s = false_alarm_s = confusion_s = 0.0
    for t0, t1 in zip(boundaries, boundaries[1:], strict=False):
        duration = t1 - t0
        ref_active = _active_speakers(ref, t0, t1)
        hyp_active = _active_speakers(hyp, t0, t1)
        correct = sum(1 for h in hyp_active if mapping.get(h) in ref_active)
        miss_s += max(0, len(ref_active) - len(hyp_active)) * duration
        false_alarm_s += max(0, len(hyp_active) - len(ref_active)) * duration
        confusion_s += (min(len(ref_active), len(hyp_active)) - correct) * duration
    return miss_s, false_alarm_s, confusion_s
