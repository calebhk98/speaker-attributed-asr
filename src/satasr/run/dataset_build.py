"""Build a dataset of ``MixedClip``s end-to-end — the runnable data pipeline.

Ties every data-side stage together through their interfaces (design §4):
text source -> TTS engine (weighted) -> aligner -> overlap mixer -> augment
chain -> dataset writer. Runs fully offline on the ``sine`` reference engine, so
it is the part of a run this environment can actually execute.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, replace

from satasr.alignment import ALIGNERS
from satasr.augment import AUGMENTERS, AugmentChain
from satasr.core.interfaces import Aligner, Augmenter, TTSEngine
from satasr.core.models import SpeakerClip, SpeakerCounts
from satasr.mixing import OverlapMixer
from satasr.pipeline.balancing import (
    DEFAULT_TOTAL_WEIGHTS,
    BalancingConfig,
    plan_clip_counts,
)
from satasr.pipeline.dataset_writer import DatasetWriter
from satasr.pipeline.text_feed import endless_sentences
from satasr.run.config import AugmentSpec, PipelineConfig
from satasr.run.engine_pool import EnginePool
from satasr.run.voices import VoiceProvider, build_voice_provider
from satasr.text import TEXT_SOURCES


@dataclass(frozen=True)
class BuildSummary:
    """What one ``build_dataset`` run produced."""

    num_examples: int
    output_dir: str
    splits: dict[str, int]


def build_dataset(
    config: PipelineConfig, rng: random.Random | None = None
) -> BuildSummary:
    """Generate, mix, augment and write ``config.dataset.num_examples`` clips."""
    rng = rng if rng is not None else random.Random(config.seed)
    dataset = config.dataset
    text = endless_sentences(
        TEXT_SOURCES.create(dataset.text_source, **dataset.text_options)
    )
    aligner = ALIGNERS.create(dataset.aligner)
    augment = _chain(dataset.augment, rng)
    writer = DatasetWriter(dataset.output_dir)
    engines = EnginePool(dataset.engine_weights)
    voices = build_voice_provider(dataset, config.seed)
    mixer = OverlapMixer(config.mixing, rng)

    plan = plan_clip_counts(dataset.num_examples, _balancing(config, voices), rng)
    splits: Counter[str] = Counter()
    for index, counts in enumerate(plan):
        clips = _example_clips(counts, text, engines, aligner, voices, rng)
        mixed = mixer.mix(clips, counts)
        row = writer.write(
            f"clip{index:06d}", mixed.with_audio(augment.apply(mixed.audio))
        )
        splits[row.split] += 1
    return BuildSummary(len(plan), dataset.output_dir, dict(splits))


def _balancing(config: PipelineConfig, voices: VoiceProvider) -> BalancingConfig:
    """Cap the draws to the run's ceiling (§4.6) and the available voice count.

    Simultaneous overlap can never exceed ``max_simultaneous_speakers``; the
    total distinct speakers per clip can never exceed how many voices exist
    (a bank of N speakers cannot fill a larger clip).
    """
    ceiling = config.mixing.max_simultaneous_speakers
    simultaneous = _cap(config.mixing.simultaneous_weights, ceiling)
    capacity = voices.capacity()
    if capacity is None:
        return BalancingConfig(simultaneous_weights=simultaneous)
    return BalancingConfig(
        total_weights=_cap(DEFAULT_TOTAL_WEIGHTS, capacity),
        simultaneous_weights=simultaneous,
    )


def _cap(weights: dict[int, float], ceiling: int) -> dict[int, float]:
    """Keep only speaker counts that fit within ``ceiling`` (>= 1 guaranteed)."""
    kept = {count: weight for count, weight in weights.items() if count <= ceiling}
    return kept if kept else {1: 1.0}


def _example_clips(
    counts: SpeakerCounts,
    text: Iterator[str],
    engines: EnginePool,
    aligner: Aligner,
    voices: VoiceProvider,
    rng: random.Random,
) -> list[SpeakerClip]:
    return [
        _one_clip(speaker_id, next(text), engines.pick(rng), voices, aligner)
        for speaker_id in voices.speaker_ids(counts.total, rng)
    ]


def _one_clip(
    speaker_id: str,
    sentence: str,
    engine: TTSEngine,
    voices: VoiceProvider,
    aligner: Aligner,
) -> SpeakerClip:
    voice = voices.reference(speaker_id, supports_cloning=engine.supports_cloning)
    clip = engine.synthesize(sentence, voice)
    return replace(clip, words=aligner.align(clip.audio, clip.text))


def _chain(specs: tuple[AugmentSpec, ...], rng: random.Random) -> AugmentChain:
    return AugmentChain([_augmenter(spec, rng) for spec in specs])


def _augmenter(spec: AugmentSpec, rng: random.Random) -> Augmenter:
    """Build one augmenter, injecting a seeded rng for those that take one."""
    options = dict(spec.options)
    try:
        return AUGMENTERS.create(spec.name, **options)
    except TypeError as exc:
        if "rng" not in str(exc):
            raise
        return AUGMENTERS.create(spec.name, rng=rng, **options)
