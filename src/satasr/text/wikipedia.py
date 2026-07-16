"""Wikipedia article text — cleanly licensed, topically diverse (design §4.5).

Wikipedia is called out in the design as a preferred source: it is free of
the copyright concerns that rule out movie/TV scripts, and its topic spread
beats narrow LLM-generated or social-post corpora. This module is a STUB —
it is importable and registers under "wikipedia" so the roster is visible
and the pipeline can select it by name, but actually fetching and yielding
article text is unbuilt. It must never make network calls at import time;
only calling :meth:`sentences` fails, and it fails loudly rather than
silently returning nothing.
"""

from __future__ import annotations

from collections.abc import Iterator

from satasr.text.registry import TEXT_SOURCES

_ISSUE_NOTE = (
    "WikipediaTextSource is not implemented yet (design §4.5: clean "
    "licensing, high topic diversity). See the tracking GitHub issue for "
    "this text source in the speaker-attributed-asr repository before "
    "picking this up."
)


@TEXT_SOURCES.register("wikipedia")
class WikipediaTextSource:
    """Placeholder for a Wikipedia-backed text source. Not yet implemented."""

    def __init__(self, language: str = "en") -> None:
        # Stored, not used yet — future implementation will fetch this
        # language's Wikipedia dump/API. No network access happens here.
        self._language = language

    def sentences(self) -> Iterator[str]:
        raise NotImplementedError(_ISSUE_NOTE)
