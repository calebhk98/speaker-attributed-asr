"""Sentence segmentation — the single source of truth (design §4.5).

Every ``TextSource`` that needs to turn raw prose into per-sentence TTS
inputs calls :func:`split_sentences` rather than re-implementing its own
splitter. Kept dependency-free (no nltk/spacy) so text sources have no heavy
runtime requirements; good enough for clean Wikipedia/LLM/social-post text,
not a general-purpose NLP sentence tokenizer.
"""

from __future__ import annotations

import re

# Split right after a run of sentence-ending punctuation (., !, ?) that is
# followed by whitespace. The punctuation stays attached to the sentence it
# ends; runs like "?!" or "..." are treated as one boundary, not several.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into stripped, non-empty sentences.

    Text with no terminal punctuation is returned as a single sentence.
    Leading/trailing whitespace and empty fragments (e.g. from runs of blank
    lines) are dropped.
    """
    stripped = text.strip()
    if not stripped:
        return []
    pieces = _BOUNDARY.split(stripped)
    return [piece.strip() for piece in pieces if piece.strip()]
