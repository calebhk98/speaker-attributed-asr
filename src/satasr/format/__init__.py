"""Serialized-output (SOT) transcript format — the single source of truth for
turning a :class:`~satasr.core.models.MixedClip`'s ground truth into the text
string the model trains on and back (design §4.7).

Re-exports the public API of :mod:`satasr.format.serialization`; import from
here rather than reaching into the submodule.
"""

from satasr.format.serialization import (
    DEFAULT_TRUNCATION_MARKER,
    parse_utterances,
    serialize_clip,
    serialize_utterances,
)

__all__ = [
    "DEFAULT_TRUNCATION_MARKER",
    "parse_utterances",
    "serialize_clip",
    "serialize_utterances",
]
