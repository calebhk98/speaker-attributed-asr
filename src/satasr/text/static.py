"""A fixed, in-memory text source — for tests and small hand-built fixtures.

Not a §4.5 corpus source itself; it exists so other stages (mixing, TTS
engines, the pipeline) can be exercised end-to-end without depending on a
real text corpus.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from satasr.text.registry import TEXT_SOURCES


@TEXT_SOURCES.register("static")
class StaticTextSource:
    """Yields exactly the sentences it was constructed with, in order."""

    def __init__(self, sentences: Sequence[str]) -> None:
        self._sentences = tuple(sentences)

    def sentences(self) -> Iterator[str]:
        """Return a fresh iterator each call so callers can re-iterate."""
        return iter(self._sentences)
