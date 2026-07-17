"""The single registry every aligner plugs into.

Mirrors ``satasr.tts.registry``: aligners register themselves by name here (see
``proportional.py``, ``ctc_segmentation.py``), so the pipeline can select an
aligner purely by name without importing any concrete engine (design §4.7).
"""

from __future__ import annotations

from satasr.core.interfaces import Aligner
from satasr.core.registry import Registry

ALIGNERS: Registry[Aligner] = Registry("aligner")
