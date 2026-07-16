"""Tests for the CTC-segmentation aligner.

The fast tests below need no torch/transformers install: they check
registration, the empty-text guard clause (which runs before any heavy
import), and the pure helper functions that turn raw
``ctc_segmentation`` output into the ``Aligner`` contract (monotonic,
non-overlapping, clip-covering words). The ``slow`` test drives the real
wav2vec2 CTC model and is skipped unless torch and ``ctc_segmentation``
are actually installed.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from satasr.alignment import ctc_segmentation as ctc_module
from satasr.alignment.registry import ALIGNERS
from satasr.core.audio import SAMPLE_RATE, AudioBuffer

_HEAVY_DEPS_MISSING = (
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("transformers") is None
    or importlib.util.find_spec("ctc_segmentation") is None
)


def _audio(duration_s: float = 2.0) -> AudioBuffer:
    samples = np.zeros(round(duration_s * SAMPLE_RATE), dtype=np.float32)
    return AudioBuffer(samples, SAMPLE_RATE)


def test_ctc_segmentation_is_registered() -> None:
    assert "ctc_segmentation" in ALIGNERS.available()


def test_align_rejects_empty_text() -> None:
    aligner = ALIGNERS.create("ctc_segmentation")
    with pytest.raises(ValueError, match="empty"):
        aligner.align(_audio(), "   ")


def test_normalize_for_vocab_strips_punctuation_and_uppercases() -> None:
    assert ctc_module._normalize_for_vocab("Hello,") == "HELLO"
    assert ctc_module._normalize_for_vocab("don't") == "DON'T"


def test_normalize_for_vocab_rejects_unalignable_token() -> None:
    with pytest.raises(ValueError, match="no characters alignable"):
        ctc_module._normalize_for_vocab("...")


def test_words_from_segments_preserves_token_order_and_text() -> None:
    segments = [(0.0, 0.5, 0.9), (0.5, 1.0, 0.8)]
    words = ctc_module._words_from_segments(["hello", "world"], segments, 1.0)
    assert [w.text for w in words] == ["hello", "world"]


def test_words_from_segments_clamps_overlap_to_stay_monotonic() -> None:
    """Raw segments that overlap (a model quirk) must not produce overlapping
    words: each word's start is clamped to the previous word's end.
    """
    segments = [(0.0, 0.6, 0.9), (0.4, 1.0, 0.8)]
    words = ctc_module._words_from_segments(["a", "b"], segments, 1.0)
    for earlier, later in zip(words, words[1:], strict=False):
        assert earlier.end_s <= later.start_s
        assert earlier.start_s <= earlier.end_s


def test_words_from_segments_clamps_to_clip_duration() -> None:
    """A segment end past the clip's actual duration is snapped back to it."""
    segments = [(0.0, 0.5, 0.9), (0.5, 5.0, 0.8)]
    words = ctc_module._words_from_segments(["a", "b"], segments, 1.0)
    assert words[0].start_s == pytest.approx(0.0)
    assert words[-1].end_s == pytest.approx(1.0)


def test_words_from_segments_covers_the_full_duration_and_is_well_formed() -> None:
    segments = [(0.0, 0.3, 0.9), (0.3, 0.7, 0.8), (0.7, 1.0, 0.85)]
    tokens = ["the", "quick", "fox"]
    words = ctc_module._words_from_segments(tokens, segments, 1.0)
    assert words[0].start_s == pytest.approx(0.0)
    assert words[-1].end_s == pytest.approx(1.0)
    for word in words:
        assert word.start_s <= word.end_s
    for earlier, later in zip(words, words[1:], strict=False):
        assert earlier.end_s <= later.start_s


@pytest.mark.slow
@pytest.mark.skipif(_HEAVY_DEPS_MISSING, reason="torch/transformers/ctc-seg missing")
def test_align_real_audio_with_wav2vec2() -> None:
    """Real-weights smoke test: needs torch, transformers, ctc_segmentation,
    and a network/cache to fetch ``facebook/wav2vec2-base-960h`` — none are
    available in this environment, so this is expected to be exercised in
    CI / on a GPU box, not here.
    """
    aligner = ALIGNERS.create("ctc_segmentation")
    words = aligner.align(_audio(2.0), "hello world")
    assert [w.text for w in words] == ["hello", "world"]
    assert words[0].start_s == pytest.approx(0.0)
    assert words[-1].end_s == pytest.approx(2.0)
    for earlier, later in zip(words, words[1:], strict=False):
        assert earlier.end_s <= later.start_s
