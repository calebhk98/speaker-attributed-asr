"""Feed sentences from a ``TextSource`` for as long as generation needs them.

Corpora built for fixtures (e.g. ``StaticTextSource``) are documented as
returning "a fresh iterator each call so callers can re-iterate" — this module
is that caller. It re-invokes ``sentences()`` when one pass is exhausted, so a
small fixture can still drive a run toward an hour budget (§4.5). A genuinely
empty source (zero sentences on a full pass) stops instead of spinning.
"""

from __future__ import annotations

from collections.abc import Iterator

from satasr.core.interfaces import TextSource


def endless_sentences(text_source: TextSource) -> Iterator[str]:
    """Yield sentences, re-reading ``text_source`` once each pass is used up."""
    while True:
        produced = False
        for sentence in text_source.sentences():
            produced = True
            yield sentence
        if not produced:
            return
