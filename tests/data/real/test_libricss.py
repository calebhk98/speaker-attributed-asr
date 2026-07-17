"""Fast tests for the LibriCSS loader (design §4.8)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from satasr.data.real.libricss import load_libricss_corpus


def test_loads_one_mixed_clip_per_recording(
    make_kaldi_corpus: Callable[[], Path],
) -> None:
    corpus_dir = make_kaldi_corpus()

    clips = load_libricss_corpus(corpus_dir)

    assert len(clips) == 1
    clip = clips[0]
    assert clip.metadata["source"] == "libricss"
    assert clip.metadata["recording_id"] == "rec1"
    assert {u.speaker_id for u in clip.utterances} == {"spk_a", "spk_b"}
    assert clip.counts.total == 2
    assert clip.counts.max_simultaneous == 2  # utt1/utt2 overlap 1.0-2.0s


def test_utterance_times_match_segments(
    make_kaldi_corpus: Callable[[], Path],
) -> None:
    corpus_dir = make_kaldi_corpus()

    clip = load_libricss_corpus(corpus_dir)[0]

    by_text = {u.text: u for u in clip.utterances}
    utt1 = by_text["hello there"]
    assert utt1.start_s == 0.0
    assert utt1.end_s == 2.0
