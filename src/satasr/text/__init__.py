"""Importing this package registers every built-in text source as a side effect.

Mirrors ``satasr.tts.engines``: each new source adds one import line here and
nowhere else — the single place that knows the full text-source roster
(design §4.5).
"""

from satasr.text import (
    llm,  # noqa: F401  (import for registration)
    plain_file,  # noqa: F401  (import for registration)
    static,  # noqa: F401  (import for registration)
    wikipedia,  # noqa: F401  (import for registration)
)
from satasr.text.registry import TEXT_SOURCES
from satasr.text.splitting import split_sentences

__all__ = [
    "TEXT_SOURCES",
    "split_sentences",
    "llm",
    "plain_file",
    "static",
    "wikipedia",
]
