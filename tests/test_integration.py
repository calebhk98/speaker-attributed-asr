"""End-to-end smoke test: the whole pipeline composed through its interfaces.

Proves the slice fits together with only registries + Protocols — no stage
imports another stage's concrete type (dependency inversion). Runs entirely on
the dependency-free reference engine, so it is fast and weight-free.
"""

from __future__ import annotations

import random
from dataclasses import replace

from satasr.alignment import ALIGNERS
from satasr.augment import AUGMENTERS, AugmentChain
from satasr.core.config import MixingConfig
from satasr.core.interfaces import VoiceReference
from satasr.core.models import SpeakerCounts
from satasr.mixing import OverlapMixer
from satasr.text import TEXT_SOURCES
from satasr.tts import TTS_ENGINES


def test_text_to_mixed_training_example() -> None:
    source = TEXT_SOURCES.create(
        "static", sentences=["Hello there friend.", "How are you today?"]
    )
    engine = TTS_ENGINES.create("sine")
    aligner = ALIGNERS.create("proportional")

    clips = []
    for index, sentence in enumerate(source.sentences()):
        voice = VoiceReference(speaker_id=chr(ord("A") + index), preset=f"voice{index}")
        clip = engine.synthesize(sentence, voice)
        clips.append(replace(clip, words=aligner.align(clip.audio, clip.text)))

    mixer = OverlapMixer(MixingConfig(), random.Random(0))
    mixed = mixer.mix(clips, SpeakerCounts(total=2, max_simultaneous=2))

    augment = AugmentChain(
        [
            AUGMENTERS.create("gain", gain_db=-3.0),
            AUGMENTERS.create("white_noise", snr_db=20.0, rng=random.Random(1)),
            AUGMENTERS.create("reverb"),
        ]
    )
    final = mixed.with_audio(augment.apply(mixed.audio))

    assert len(final.utterances) == 2
    assert final.counts.total == 2
    assert final.counts.max_simultaneous <= 2
    assert final.audio.duration_s > 0
    assert {u.speaker_id for u in final.utterances} == {"A", "B"}
    # every utterance carries word timestamps from the aligner
    assert all(u.words for u in final.utterances)
