# speaker-attributed-asr

Multi-talker, **speaker-attributed ASR with serialized output** — take audio
with 1-N people talking (including overlap) and produce a per-speaker,
timestamped transcript:

```
Speaker A: (0:00-0:15) "So, when we go to Africa we need some protection from lions"
Speaker B: (0:05-0:12) "Wait, we are going to Africa?"
Speaker C: (0:00-0:15) "While those two are talking about their trip, how was the salon?"
```

This is not diarization and ASR bolted together — it is the joint problem
(target-speaker / speaker-attributed ASR with Serialized Output Training). The
reference architecture we fine-tune from is **DiCoW / SA-DiCoW**
(diarization-conditioned Whisper-large-v3-turbo, ~918M params).

The full rationale, scope decisions, data strategy and research references live
in **[`docs/design.md`](docs/design.md)** — the single source of truth. Read it
before changing direction.

## Why the code is shaped like this

The heavy lifting (10k hours of synthetic data across ~20 TTS engines, then
fine-tuning) is embarrassingly parallel. The codebase is built so each piece is
an **independent plugin against a stable interface**, so 20 people/agents can
implement 20 engines at once without stepping on each other:

- `core/interfaces.py` defines Protocols (`TTSEngine`, `Aligner`, `Augmenter`,
  `Mixer`, `TextSource`). Everything else depends on these, not on concretes
  (dependency inversion).
- A single generic `core/registry.py` powers name-based discovery for every
  extension point. Adding an engine changes **no existing orchestration code**.
- `core/models.py` / `core/audio.py` are the one definition of every data shape
  that flows through the pipeline (single source of truth).

See **[`docs/architecture.md`](docs/architecture.md)** for the module map and
**[`CLAUDE.md`](CLAUDE.md)** for the house rules (TDD red/green, never-nesting,
short files, pre-commit, comments, no repetition).

## Pipeline at a glance

```
TextSource → TTSEngine → SpeakerClip → Aligner (word timestamps on clean audio)
   → Mixer (overlap + interruption truncation) → MixedClip → Augmenter (noise/reverb)
   → training example (audio + serialized, speaker-tagged, timestamped transcript)
```

Two independent speaker counts are tracked per clip (design §4.6): how many
distinct speakers are *present*, and how many ever speak *simultaneously*.

## Quickstart

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pre-commit install
.venv/bin/python -m pytest        # fast suite, no weights needed
```

## One command to run the whole pipeline

Everything is driven by a single TOML config and one entry point:

```bash
satasr run --config config.example.toml            # execute the configured stages
satasr run --config config.example.toml --dry-run  # just print the stage plan
```

`config.example.toml` is a complete, annotated config. As shipped it runs the
`build_dataset` stage end-to-end on the dependency-free `sine` engine — text →
TTS → alignment → overlap mixing → augmentation → an on-disk dataset (audio +
`manifest.jsonl` with serialized, speaker-tagged, timestamped transcripts, split
into train/val/test) — so it works anywhere, no weights or GPU. The config
selects the text source, engines + weights, augmentation chain, mixing knobs,
and which `stages` to run (`build_dataset`, `phase1`, `phase2`, `evaluate`). The
training/eval stages need the model backend + a checkpoint and are enabled on
the GPU host. See [`config.example.toml`](config.example.toml) and
`src/satasr/run/`.

The dependency-free `sine` reference engine lets you run the whole pipeline in
CI without downloading any model weights. Real engines are added one GitHub
issue at a time — see the issue tracker, each is self-contained and parallel.

## Status

Foundation is in place and fully tested: core domain model, plugin interfaces +
registry, config, the overlap/truncation mixer, and the TTS engine base +
reference engine. Aligners, augmenters, text sources, the 20 real TTS engines,
generation orchestration, training (Phase 1 full fine-tune, Phase 2 LoRA on real
audio) and evaluation (DER, tcpWER/cpWER) are tracked as issues.

## License

MIT (project code). Individual TTS engines and datasets carry their own licenses
— several are non-commercial; see `docs/design.md` §4.3 and §4.8 before use.
