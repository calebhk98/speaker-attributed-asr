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
from satasr.core.interfaces import Aligner, Augmenter, TTSEngine, VoiceReference
from satasr.core.models import SpeakerClip, SpeakerCounts
from satasr.mixing import OverlapMixer
from satasr.pipeline.balancing import BalancingConfig, plan_clip_counts
from satasr.pipeline.dataset_writer import DatasetWriter
from satasr.pipeline.text_feed import endless_sentences
from satasr.run.config import AugmentSpec, PipelineConfig
from satasr.run.engine_pool import EnginePool
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
    mixer = OverlapMixer(config.mixing, rng)

    plan = plan_clip_counts(dataset.num_examples, _balancing(config), rng)
    splits: Counter[str] = Counter()
    for index, counts in enumerate(plan):
        clips = _example_clips(counts, text, engines, aligner, rng)
        mixed = mixer.mix(clips, counts)
        row = writer.write(
            f"clip{index:06d}", mixed.with_audio(augment.apply(mixed.audio))
        )
        splits[row.split] += 1
    return BuildSummary(len(plan), dataset.output_dir, dict(splits))


def _balancing(config: PipelineConfig) -> BalancingConfig:
    """Cap the simultaneous-speaker draw at the run's configured ceiling (§4.6)."""
    ceiling = config.mixing.max_simultaneous_speakers
    weights = {
        count: weight
        for count, weight in config.mixing.simultaneous_weights.items()
        if count <= ceiling
    }
    return BalancingConfig(simultaneous_weights=weights)


def _example_clips(
    counts: SpeakerCounts,
    text: Iterator[str],
    engines: EnginePool,
    aligner: Aligner,
    rng: random.Random,
) -> list[SpeakerClip]:
    return [
        _one_clip(f"S{speaker + 1}", next(text), engines.pick(rng), aligner)
        for speaker in range(counts.total)
    ]


def _one_clip(
    speaker_id: str, sentence: str, engine: TTSEngine, aligner: Aligner
) -> SpeakerClip:
    clip = engine.synthesize(sentence, _voice(speaker_id, engine))
    return replace(clip, words=aligner.align(clip.audio, clip.text))


def _voice(speaker_id: str, engine: TTSEngine) -> VoiceReference:
    if engine.supports_cloning:
        raise ValueError(
            f"engine {engine.name!r} clones voices; wire a VoiceBank (issue #31) "
            "before using cloning engines in a run"
        )
    return VoiceReference(speaker_id=speaker_id, preset=speaker_id)


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
