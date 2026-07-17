"""The single registry every acoustic augmenter plugs into.

Mirrors ``satasr.tts.registry``: augmenters register themselves by name (see
``gain.py``, ``noise.py``, ``reverb.py``) and the generation pipeline selects
and composes them purely by name, so adding a new transform never touches
orchestration code (design §4.9).
"""

from __future__ import annotations

from satasr.core.interfaces import Augmenter
from satasr.core.registry import Registry

AUGMENTERS: Registry[Augmenter] = Registry("augmenter")
