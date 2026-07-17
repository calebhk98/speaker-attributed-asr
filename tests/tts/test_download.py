"""Tests for the generic model downloader.

The download flow is exercised against a throwaway registry with fake engines
and an injected fetcher, so nothing touches the network or real weights.
"""

from __future__ import annotations

from satasr.core.audio import AudioBuffer
from satasr.core.interfaces import TTSEngine, VoiceReference
from satasr.core.registry import Registry
from satasr.tts.base import BaseTTSEngine
from satasr.tts.download import download_models, planned_models


class _TwoModelEngine(BaseTTSEngine):
    name = "two"
    supports_cloning = False
    model_ids = ("org/a", "org/b")

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        raise NotImplementedError


class _NoModelEngine(BaseTTSEngine):
    name = "none"
    supports_cloning = False
    model_ids = ()

    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        raise NotImplementedError


def _registry() -> Registry[TTSEngine]:
    registry: Registry[TTSEngine] = Registry("tts engine")
    registry.register("two")(_TwoModelEngine)
    registry.register("none")(_NoModelEngine)
    return registry


def test_planned_models_reads_declared_ids() -> None:
    plan = planned_models(registry=_registry())
    assert plan == {"two": ("org/a", "org/b"), "none": ()}


def test_download_fetches_every_declared_model() -> None:
    calls: list[str] = []
    fetched = download_models(
        registry=_registry(), fetcher=lambda repo, _cache: calls.append(repo)
    )
    assert calls == ["org/a", "org/b"]  # engine with no models is skipped
    assert fetched == ["org/a", "org/b"]


def test_dry_run_downloads_nothing() -> None:
    calls: list[str] = []
    fetched = download_models(
        registry=_registry(),
        fetcher=lambda repo, _cache: calls.append(repo),
        dry_run=True,
    )
    assert calls == []
    assert fetched == ["org/a", "org/b"]


def test_one_failure_does_not_abort_the_rest() -> None:
    def flaky(repo: str, _cache: str | None) -> None:
        if repo == "org/a":
            raise RuntimeError("huggingface_hub not installed")

    fetched = download_models(registry=_registry(), fetcher=flaky)
    assert fetched == ["org/b"]  # the good one still went through


def test_slug_filter_selects_a_subset() -> None:
    calls: list[str] = []
    download_models(
        ["two"], registry=_registry(), fetcher=lambda repo, _cache: calls.append(repo)
    )
    assert calls == ["org/a", "org/b"]
