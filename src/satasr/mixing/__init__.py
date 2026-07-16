"""Data Stage 2: splice single-speaker clips into overlapping multi-speaker
training examples (design §4.6/§4.7).

The public entry point is :class:`~satasr.mixing.mixer.OverlapMixer`, which
implements the :class:`~satasr.core.interfaces.Mixer` protocol.
"""

from satasr.mixing.mixer import OverlapMixer

__all__ = ["OverlapMixer"]
