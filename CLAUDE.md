# CLAUDE.md — house rules for this repo

This file is the contract every contributor (human or agent) follows. It is
deliberately short; the point is that the rules are few and non-negotiable.

## What this project is

A data + training pipeline for **multi-talker, speaker-attributed ASR with
serialized output** — audio with 1-N overlapping speakers in, a per-speaker
timestamped transcript out. See `README.md` for the goal and `docs/design.md`
for the full design document (the single source of truth for *what* and *why*).

## The seven rules

1. **TDD, red/green.** Write a failing test first and commit it (`test(...): ...
   (red)`), then make it pass and commit (`feat(...): ... (green)`). A red
   commit may use `git commit --no-verify` to record the failing spec; the
   paired green commit must pass every hook. Never write production code that
   has no test.

2. **Never nesting.** Prefer guard clauses and early returns to nested blocks.
   Extract a helper before you reach a third level of indentation. Ruff enforces
   this (`max-nested-blocks = 3`, capped branches/returns in `pyproject.toml`).

3. **Short files.** Aim for < ~120 lines per module. A file that outgrows this
   is two files. One clear responsibility per file.

4. **Single source of truth. No repetition.** Every concept is defined once:
   data shapes in `core/models.py`, audio in `core/audio.py`, plugin lookup via
   the one generic `core/registry.py`, tunables in `core/config.py`. If you are
   about to copy code, extract it instead.

5. **Depend on abstractions (dependency inversion).** Orchestration depends on
   the Protocols in `core/interfaces.py` — never on a concrete engine. New
   back-ends are *added* by implementing an interface and registering it; no
   existing code changes. This is what lets the 20 TTS engines be built in
   parallel.

6. **Comments explain why.** Every module and public function has a docstring.
   Comments cite the design section they implement (e.g. `# §4.7`). Explain
   intent and non-obvious decisions, not the syntax.

7. **Pre-commit is the gate.** `pre-commit install` once. Every commit runs
   ruff (lint + format), mypy (strict), and the fast test suite. Green locally
   before you push.

## Commands

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pre-commit install          # wire up the commit gate
.venv/bin/python -m pytest            # all fast tests
.venv/bin/python -m pytest -m slow    # weight-downloading / GPU tests (opt-in)
.venv/bin/ruff check src tests        # lint
.venv/bin/mypy                        # type-check (strict)
```

## How to add a TTS engine (the parallel work)

Each engine is one self-contained GitHub issue and touches only new files:

1. Create `src/satasr/tts/engines/<engine>.py`.
2. Subclass `BaseTTSEngine`, set `name` and `supports_cloning`, implement the
   single `_render(text, voice) -> AudioBuffer` method. Everything else
   (validation, the voice contract, clip packaging) is inherited — do not
   reimplement it.
3. Decorate the class with `@TTS_ENGINES.register("<engine>")`. The file is
   **auto-discovered** — you do not edit `engines/__init__.py` or any other
   shared file. This is what keeps engines conflict-free in parallel.
4. Keep the heavy third-party library import **inside** `_render` (never at
   module top level) so discovery stays cheap and weight-free.
5. Add `tests/tts/engines/test_<engine>.py`. Mark tests that download weights or
   need a GPU with `@pytest.mark.slow` so CI stays fast; add a lightweight test
   (shape/contract) that runs without weights.

Use `src/satasr/tts/engines/sine.py` as the reference template. The same
register-and-implement pattern applies to aligners, augmenters, and text
sources (each has its own registry).

## Boundaries for agents

- Touch only the files your task owns. Do not edit another module's files.
- Do not commit weights, audio, or datasets (`.gitignore` blocks them).
- Keep `docs/design.md` authoritative; if code and design disagree, raise it,
  don't silently diverge.
