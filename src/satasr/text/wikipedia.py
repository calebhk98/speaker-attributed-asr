"""Wikipedia article text — cleanly licensed, topically diverse (design §4.5).

Wikipedia is called out in the design as a preferred source: it is free of
the copyright concerns that rule out movie/TV scripts, and its topic spread
beats narrow LLM-generated or social-post corpora (§4.5 also notes the
accepted gap: this synthetic-adjacent text won't reliably carry natural
disfluency). Article text is pulled from the public MediaWiki API's plaintext
"extracts" (no extra dependency needed — ``urllib`` is stdlib), cleaned of
the few markup leftovers that survive as plaintext (section headings,
``[12]``-style reference markers), and handed to the one shared splitter.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence

from satasr.text.registry import TEXT_SOURCES
from satasr.text.splitting import split_sentences

_API_URL_TEMPLATE = "https://{language}.wikipedia.org/w/api.php"

# Plaintext extracts (explaintext=1) still carry a couple of markup remnants
# that split_sentences (deliberately dependency-free) shouldn't have to
# special-case: "== Heading ==" section titles and "[1]" reference markers.
_HEADING_LINE = re.compile(r"^\s*=+.*?=+\s*$", re.MULTILINE)
_REFERENCE_MARKER = re.compile(r"\[\d+\]")


def _clean_markup(raw: str) -> str:
    """Strip leftover section headings and reference markers from ``raw``."""
    without_headings = _HEADING_LINE.sub("", raw)
    return _REFERENCE_MARKER.sub("", without_headings)


def _fetch_via_api(language: str, title: str) -> str:
    """Fetch one article's plaintext extract from the Wikipedia API.

    Only called from inside :meth:`WikipediaTextSource.sentences`, so the
    network-touching imports live here rather than at module scope — the
    module and the class stay import-safe and construction-safe (§4.5 stub
    contract this class replaces).
    """
    import json
    import urllib.parse
    import urllib.request

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
    }
    url = _API_URL_TEMPLATE.format(language=language)
    url = f"{url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload: object = json.loads(response.read().decode("utf-8"))
    return _extract_from_payload(payload)


def _extract_from_payload(payload: object) -> str:
    """Pull the ``extract`` field out of a query-API response, defensively.

    The response is untyped JSON (mypy strict sees ``object``); guard clauses
    walk down to the single page's extract without assuming its shape.
    """
    query = payload.get("query") if isinstance(payload, dict) else None
    pages = query.get("pages") if isinstance(query, dict) else None
    if not isinstance(pages, dict) or not pages:
        return ""
    page = next(iter(pages.values()))
    extract = page.get("extract") if isinstance(page, dict) else None
    return extract if isinstance(extract, str) else ""


@TEXT_SOURCES.register("wikipedia")
class WikipediaTextSource:
    """Yields sentences pulled from named Wikipedia articles, in order.

    ``titles`` names the articles to draw from; picking which titles (e.g.
    for topic diversity, §4.5) is the caller's concern, not this class's.
    ``fetch_article`` is the network call, injectable so tests can supply a
    small local fixture instead of the real API (keeps CI offline). Neither
    fetching nor cleaning happens until :meth:`sentences` is iterated.
    """

    def __init__(
        self,
        titles: Sequence[str] = (),
        language: str = "en",
        fetch_article: Callable[[str, str], str] | None = None,
    ) -> None:
        self._titles = tuple(titles)
        self._language = language
        self._fetch_article = fetch_article or _fetch_via_api

    def sentences(self) -> Iterator[str]:
        """Fetch each article and yield its sentences, lazily, in order.

        A generator function: nothing runs — no fetch, no cleanup, no
        splitting — until the caller pulls the first item, and each call
        returns a fresh generator so the source can be re-iterated.
        """
        for title in self._titles:
            raw = self._fetch_article(self._language, title)
            cleaned = _clean_markup(raw)
            yield from split_sentences(cleaned)
