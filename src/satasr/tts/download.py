"""Fetch model weights for TTS engines ahead of time.

One generic downloader for every engine: it reads each engine's declared
``model_ids`` from the registry and pulls each repo via ``huggingface_hub``. It
knows nothing about any specific engine (dependency inversion) — an engine opts
in simply by declaring ``model_ids``. Run it as a module:

    python -m satasr.tts.download --list           # show what would be fetched
    python -m satasr.tts.download --all             # download everything
    python -m satasr.tts.download --engines xtts_v2,bark
    python -m satasr.tts.download --all --dry-run

Weights are large and gated behind optional dependencies, so nothing is
downloaded at import time and the actual fetch is only imported when needed.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from satasr.core.interfaces import TTSEngine
from satasr.core.registry import Registry
from satasr.tts.registry import TTS_ENGINES

# A fetcher takes a repo id + optional cache dir and downloads it. Injectable so
# the download flow can be tested without touching the network.
Fetcher = Callable[[str, str | None], None]


def planned_models(
    slugs: Sequence[str] | None = None,
    registry: Registry[TTSEngine] = TTS_ENGINES,
) -> dict[str, tuple[str, ...]]:
    """Map each selected engine to the model ids it declares."""
    names = list(slugs) if slugs is not None else registry.available()
    return {name: _model_ids(registry.create(name)) for name in names}


def download_models(
    slugs: Sequence[str] | None = None,
    cache_dir: str | None = None,
    *,
    dry_run: bool = False,
    registry: Registry[TTSEngine] = TTS_ENGINES,
    fetcher: Fetcher | None = None,
) -> list[str]:
    """Download every declared model for the selected engines; return the ids.

    Idempotent: ``huggingface_hub`` skips already-cached repos. Engines that
    declare no models are reported and skipped, not treated as an error.
    """
    fetch = fetcher or _hf_snapshot_download
    fetched: list[str] = []
    for name, ids in planned_models(slugs, registry).items():
        if not ids:
            print(f"[skip] {name}: no model_ids declared")
            continue
        fetched.extend(_fetch_all(name, ids, cache_dir, fetch, dry_run=dry_run))
    return fetched


def _fetch_all(
    name: str,
    ids: tuple[str, ...],
    cache_dir: str | None,
    fetch: Fetcher,
    *,
    dry_run: bool,
) -> list[str]:
    """Fetch every id for one engine, best-effort (a flat helper keeps callers
    shallow). Returns the ids that were fetched (or planned, in a dry run)."""
    done: list[str] = []
    for repo_id in ids:
        if dry_run:
            print(f"[plan] {name}: {repo_id}")
            done.append(repo_id)
            continue
        _try_fetch(name, repo_id, cache_dir, fetch, done)
    return done


def _try_fetch(
    name: str, repo_id: str, cache_dir: str | None, fetch: Fetcher, done: list[str]
) -> None:
    """Fetch one repo. A failure (uninstalled hub, non-HF id, network) is logged
    and skipped so one engine never blocks the rest — pre-fetch is best-effort."""
    try:
        fetch(repo_id, cache_dir)
    except Exception as exc:  # noqa: BLE001 - report and continue, don't abort
        print(f"[fail] {name}: {repo_id} ({exc})")
        return
    print(f"[ok  ] {name}: {repo_id}")
    done.append(repo_id)


def _model_ids(engine: TTSEngine) -> tuple[str, ...]:
    return tuple(getattr(engine, "model_ids", ()))


def _hf_snapshot_download(repo_id: str, cache_dir: str | None) -> None:
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    snapshot_download(repo_id=repo_id, cache_dir=cache_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download TTS engine model weights.")
    parser.add_argument("--all", action="store_true", help="every registered engine")
    parser.add_argument("--engines", help="comma-separated engine names")
    parser.add_argument("--cache-dir", help="Hugging Face cache directory")
    parser.add_argument(
        "--list", action="store_true", help="print the plan, fetch nothing"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="log without downloading"
    )
    return parser


def _selected(args: argparse.Namespace) -> list[str] | None:
    if args.engines:
        return [name.strip() for name in args.engines.split(",") if name.strip()]
    return None  # None means "all registered engines"


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    slugs = _selected(args)

    if args.list:
        for name, ids in planned_models(slugs).items():
            print(f"{name}: {', '.join(ids) if ids else '(none declared)'}")
        return 0

    download_models(slugs, args.cache_dir, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
