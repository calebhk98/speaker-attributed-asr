"""Importing this package registers every built-in aligner as a side effect.

Each new aligner adds one import line here and nowhere else — the single
place that knows the full aligner roster (mirrors ``satasr.tts.engines``).
"""

from satasr.alignment import (
    ctc_segmentation,  # noqa: F401  (import for registration)
    mfa,  # noqa: F401  (import for registration)
    proportional,  # noqa: F401  (import for registration)
)
from satasr.alignment.registry import ALIGNERS

__all__ = ["ALIGNERS", "ctc_segmentation", "mfa", "proportional"]
