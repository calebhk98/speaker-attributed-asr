"""Decide *where* each clip sits on the mixed timeline and *whether* it is cut.

The scheduler never lets the number of simultaneously-active clips exceed the
requested peak (design §4.6), and cuts interrupted speakers within the truncation
window (§4.7). It works purely on clip durations, so it is trivial to unit-test
without any audio.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from satasr.core.config import MixingConfig


@dataclass(frozen=True)
class Placement:
    """Where clip ``index`` starts, and how much of it to keep (None = all)."""

    index: int
    offset_s: float
    cut_at_s: float | None


@dataclass
class _Slot:
    """Mutable bookkeeping for a clip while scheduling."""

    offset_s: float
    duration_s: float
    cut_at_s: float | None = None

    @property
    def kept_s(self) -> float:
        return self.duration_s if self.cut_at_s is None else self.cut_at_s

    @property
    def end_s(self) -> float:
        return self.offset_s + self.kept_s


class OverlapScheduler:
    """Places clips left-to-right, weaving in overlaps up to the target peak."""

    def __init__(self, config: MixingConfig) -> None:
        self._config = config

    def schedule(
        self, durations: list[float], target_peak: int, rng: random.Random
    ) -> list[Placement]:
        slots: list[_Slot] = []
        for duration in durations:
            slots.append(self._place_one(slots, duration, target_peak, rng))
        return [Placement(i, s.offset_s, s.cut_at_s) for i, s in enumerate(slots)]

    def _place_one(
        self,
        placed: list[_Slot],
        duration: float,
        target_peak: int,
        rng: random.Random,
    ) -> _Slot:
        """Return the slot for one new clip — overlapping or sequential."""
        overlap = self._try_overlap(placed, target_peak, rng)
        if overlap is None:
            return _Slot(self._sequential_offset(placed), duration)

        start, victim = overlap
        self._maybe_truncate(victim, start, rng)
        return _Slot(start, duration)

    def _try_overlap(
        self, placed: list[_Slot], target_peak: int, rng: random.Random
    ) -> tuple[float, _Slot] | None:
        """Pick a start time that overlaps an active clip, or None to go after."""
        if not placed or target_peak <= 1:
            return None
        if rng.random() >= self._config.overlap_probability:
            return None

        victim = max(placed, key=lambda slot: slot.offset_s)
        start = rng.uniform(victim.offset_s, victim.end_s)
        if self._active_at(placed, start) >= target_peak:
            return None  # overlapping here would break the peak cap
        return start, victim

    def _maybe_truncate(self, victim: _Slot, start: float, rng: random.Random) -> None:
        """Cut the interrupted speaker short, sometimes (design §4.6)."""
        if rng.random() < self._config.uncut_probability:
            return  # this speaker talks through the interruption

        into = start - victim.offset_s
        delta = rng.uniform(
            self._config.min_truncation_s, self._config.max_truncation_s
        )
        cut = min(victim.duration_s, into + delta)
        if victim.cut_at_s is None or cut < victim.cut_at_s:
            victim.cut_at_s = cut

    @staticmethod
    def _sequential_offset(placed: list[_Slot]) -> float:
        return max((slot.end_s for slot in placed), default=0.0)

    @staticmethod
    def _active_at(placed: list[_Slot], time_s: float) -> int:
        return sum(1 for slot in placed if slot.offset_s <= time_s < slot.end_s)
