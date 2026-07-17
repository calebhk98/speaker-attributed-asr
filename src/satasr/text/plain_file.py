"""A text source backed by a single UTF-8 text file on disk.

Useful for ad-hoc corpora (a pasted article, an LLM-generated passage saved
to disk) that don't warrant a dedicated source implementation. Sentence
segmentation is delegated to :func:`satasr.text.splitting.split_sentences`
(§4.5's per-sentence TTS granularity) — this module owns no splitting logic
of its own.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from satasr.text.registry import TEXT_SOURCES
from satasr.text.splitting import split_sentences


@TEXT_SOURCES.register("plain_file")
class PlainFileTextSource:
    """Reads ``path`` and yields its sentences.

    The file is read lazily inside :meth:`sentences`, not at construction, so
    building the source never touches disk and the file may change (or not
    yet exist) until it is actually consumed.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def sentences(self) -> Iterator[str]:
        """Return a fresh iterator each call so callers can re-iterate."""
        text = self._path.read_text(encoding="utf-8")
        return iter(split_sentences(text))
