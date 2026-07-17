"""Importing this package registers every built-in augmenter as a side effect.

Each new augmenter adds one import line here and nowhere else — the single
place that knows the full augmenter roster (mirrors
``satasr.tts.engines.__init__``).
"""

from satasr.augment import gain, noise, reverb  # noqa: F401  (import for registration)
from satasr.augment.chain import AugmentChain
from satasr.augment.registry import AUGMENTERS

__all__ = ["AUGMENTERS", "AugmentChain", "gain", "noise", "reverb"]
