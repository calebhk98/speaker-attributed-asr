"""Deliberately balance how many speakers overlap at once (design §4.6).

Rather than letting the simultaneous-speaker count fall out randomly, we draw it
from an explicit weight table (the Sortformer-style ~1/3/6/10 idea). A weight of
zero means "never pick this count".
"""

from __future__ import annotations

import random


def sample_simultaneous(weights: dict[int, float], rng: random.Random) -> int:
    """Draw a simultaneous-speaker count according to ``weights``."""
    choices = [count for count, weight in weights.items() if weight > 0]
    if not choices:
        raise ValueError("no positive weights to sample from")

    positive = [weights[count] for count in choices]
    return rng.choices(choices, weights=positive, k=1)[0]
