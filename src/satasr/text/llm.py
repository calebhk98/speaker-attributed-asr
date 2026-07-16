"""LLM-generated text source (design §4.5).

Wikipedia and social posts are called out alongside LLM-generated text as
clean, diverse source material; movie/TV scripts are explicitly out of scope
here (copyright flag, §4.5, open Q2). This source prompts a language model
for prose on a random everyday topic -- never dialogue/script formatting --
and segments the result with the shared :func:`split_sentences` (single
source of truth, no second splitter). Known gap per §4.5: synthetic LLM text
won't reliably contain natural disfluency; this source makes no attempt to
inject any.

The model/provider lives behind :class:`LlmProvider` (dependency inversion,
house rule 5) so a stub can stand in for tests and any real backend (an
OpenAI-compatible endpoint, a local model, ...) is swappable without touching
this module. No network call happens at import or construction -- only
:meth:`LlmTextSource.sentences` triggers generation.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from satasr.text.registry import TEXT_SOURCES
from satasr.text.splitting import split_sentences

# Config-driven default prompt: plain prose, explicitly not dialogue/script
# formatting, to keep generated text clear of the §4.5 copyright flag.
_DEFAULT_PROMPT = (
    "Write a short paragraph (3-6 sentences) of plain English prose about a "
    "random everyday topic. Do not use dialogue, scripts, or scene "
    "formatting -- narrative prose only."
)
# Default model id for the built-in OpenAI-compatible provider.
_DEFAULT_MODEL = "gpt-4o-mini"


@runtime_checkable
class LlmProvider(Protocol):
    """Turns a prompt into raw generated text.

    The one seam between :class:`LlmTextSource` and any concrete model/vendor
    SDK (§4.5 dependency inversion) -- implement this method to add a new
    backend; nothing else in this module changes.
    """

    def generate(self, prompt: str) -> str: ...


class _OpenAiCompatibleProvider:
    """Default provider: an OpenAI-compatible chat-completions endpoint.

    Mirrors the lazy-import pattern used by ``tts/engines/voxtral.py`` -- the
    ``openai`` package is a heavy optional dependency, so it is imported only
    inside :meth:`generate`, never at module top level.
    """

    def __init__(self, model: str) -> None:
        self._model = model

    def generate(self, prompt: str) -> str:
        from openai import OpenAI  # type: ignore

        client = OpenAI()
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        return content or ""


@TEXT_SOURCES.register("llm")
class LlmTextSource:
    """Yields sentences generated on demand by an :class:`LlmProvider`.

    Construction only stores config -- ``prompt``, and ``provider`` (defaults
    to the built-in OpenAI-compatible provider for ``model``) -- and never
    calls the provider. Each call to :meth:`sentences` generates fresh text
    and splits it via the shared :func:`split_sentences`.
    """

    def __init__(
        self,
        prompt: str = _DEFAULT_PROMPT,
        provider: LlmProvider | None = None,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._prompt = prompt
        self._provider = (
            provider if provider is not None else _OpenAiCompatibleProvider(model)
        )

    def sentences(self) -> Iterator[str]:
        """Generate text lazily and split it into sentences.

        Calls :meth:`LlmProvider.generate` fresh each time so re-iterating
        yields newly generated text rather than a cached first result -- no
        network/model call happens until this method runs.
        """
        text = self._provider.generate(self._prompt)
        return iter(split_sentences(text))
