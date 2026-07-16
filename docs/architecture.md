# Architecture

The project is layered so that everything depends *inward* on `core`. `core`
depends only on numpy. Nothing in `core` imports a back-end (torch, a TTS model,
an aligner). This is what keeps the extension points independent and testable.

```
              ┌────────────────────────────────────────────┐
              │  core  (the stable centre)                  │
              │  audio · models · interfaces · registry ·   │
              │  config                                     │
              └────────────────────────────────────────────┘
                    ▲          ▲          ▲          ▲
        implements  │          │          │          │  implements
              ┌─────┴───┐ ┌────┴────┐ ┌───┴────┐ ┌───┴──────┐
              │  tts    │ │alignment│ │ augment│ │  text     │
              │ engines │ │ aligners│ │ers     │ │  sources  │
              └─────────┘ └─────────┘ └────────┘ └───────────┘
                    ▲          ▲          ▲          ▲
                    └──────────┴────┬─────┴──────────┘
                                    │  compose (never imported by core)
                           ┌────────┴─────────┐
                           │ mixing · pipeline │
                           │ training · eval   │
                           └───────────────────┘
```

## The data path

Every stage speaks in the value objects from `core/models.py`, so stages compose
without adapters:

| Stage        | Interface (`core/interfaces.py`) | Input → Output                          |
|--------------|----------------------------------|-----------------------------------------|
| Text         | `TextSource`                     | — → sentences (`str`)                    |
| Synthesis    | `TTSEngine`                      | text + `VoiceReference` → `SpeakerClip`  |
| Alignment    | `Aligner`                        | clean `AudioBuffer` + text → `Word`s     |
| Mixing       | `Mixer`                          | `SpeakerClip`s → `MixedClip`             |
| Augmentation | `Augmenter`                      | `AudioBuffer` → `AudioBuffer`            |

`MixedClip` is the finished training example: mixed `AudioBuffer` + a tuple of
`PlacedUtterance` (the serialized transcript) + `SpeakerCounts` (the two
independent counts from design §4.6).

## Extension points

Each has exactly one registry (an instance of the single generic
`core/registry.Registry`) so discovery is uniform:

| Registry          | Module                       | Add one by…                     |
|-------------------|------------------------------|---------------------------------|
| `TTS_ENGINES`     | `tts/registry.py`            | subclassing `BaseTTSEngine`     |
| `ALIGNERS`        | `alignment/registry.py`      | implementing `Aligner`          |
| `AUGMENTERS`      | `augment/registry.py`        | implementing `Augmenter`        |
| `TEXT_SOURCES`    | `text/registry.py`           | implementing `TextSource`       |

A plugin registers with a decorator and is imported once in its package
`__init__`. Nothing else changes — that is the whole point.

## Why this enables parallelism

Because a new engine (a) depends only on `BaseTTSEngine` + the value objects and
(b) lives in its own file with its own test file, twenty engines are twenty
non-overlapping diffs. The reference `sine` engine proves the contract and keeps
CI weight-free; real engines replace nothing, they just add themselves.

## Configuration

All tunables live in `core/config.py` (max simultaneous speakers, truncation
window, overlap probability, per-engine hour allocation). Change a number in one
place; generation, mixing and tests all see it.
