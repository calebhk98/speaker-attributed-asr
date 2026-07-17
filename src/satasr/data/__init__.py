"""Data loaders that produce the shared training-example schema (design §4.8).

Every loader here — real-corpus (:mod:`satasr.data.real`) and, eventually,
synthetic-facing helpers — yields :class:`~satasr.core.models.MixedClip`, the
same schema the synthetic pipeline's mixer produces. That single shared shape
is what lets downstream training code stay source-agnostic: it never needs to
know whether a clip came from a TTS-mixed synthetic scene or a real corpus.
"""

from __future__ import annotations
