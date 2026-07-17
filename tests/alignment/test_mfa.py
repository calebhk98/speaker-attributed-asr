"""Tests for the Montreal Forced Aligner (MFA) aligner.

The fast tests below need no MFA install: they check registration, the pure
Python TextGrid parser against a hand-written fixture, and that ``align``
fails with an actionable message when the ``mfa`` CLI is absent (true of
this CI environment). The ``slow`` test drives the real ``mfa`` binary and
is skipped unless it is actually on PATH.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from satasr.alignment import mfa as mfa_module
from satasr.alignment.registry import ALIGNERS
from satasr.core.audio import SAMPLE_RATE, AudioBuffer

_SAMPLE_TEXTGRID = """
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.2
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.2
        intervals: size = 3
        intervals [1]:
            xmin = 0.0
            xmax = 0.2
            text = ""
        intervals [2]:
            xmin = 0.2
            xmax = 0.7
            text = "hello"
        intervals [3]:
            xmin = 0.7
            xmax = 1.2
            text = "world"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.2
        intervals: size = 1
        intervals [1]:
            xmin = 0.0
            xmax = 1.2
            text = "sil"
"""


def _audio(duration_s: float = 1.0) -> AudioBuffer:
    samples = np.zeros(round(duration_s * SAMPLE_RATE), dtype=np.float32)
    return AudioBuffer(samples, SAMPLE_RATE)


def test_mfa_is_registered() -> None:
    assert "mfa" in ALIGNERS.available()


def test_parse_textgrid_words_skips_silence_and_reads_words_tier() -> None:
    words = mfa_module._parse_textgrid_words(_SAMPLE_TEXTGRID)
    assert [w.text for w in words] == ["hello", "world"]


def test_parse_textgrid_words_is_monotonic_and_well_formed() -> None:
    words = mfa_module._parse_textgrid_words(_SAMPLE_TEXTGRID)
    for word in words:
        assert word.start_s < word.end_s
    for earlier, later in zip(words, words[1:], strict=False):
        assert earlier.end_s <= later.start_s


def test_parse_textgrid_words_rejects_missing_tier() -> None:
    with pytest.raises(RuntimeError, match="no tier named"):
        mfa_module._parse_textgrid_words(
            'item []:\n    item [1]:\n        name = "phones"\n'
        )


def test_parse_textgrid_words_rejects_all_silence_tier() -> None:
    silent = _SAMPLE_TEXTGRID.replace('text = "hello"', 'text = "sp"').replace(
        'text = "world"', 'text = "sil"'
    )
    with pytest.raises(RuntimeError, match="no aligned words"):
        mfa_module._parse_textgrid_words(silent)


def test_align_rejects_empty_text() -> None:
    aligner = ALIGNERS.create("mfa")
    with pytest.raises(ValueError, match="empty"):
        aligner.align(_audio(), "   ")


def test_align_reports_missing_mfa_binary() -> None:
    aligner = ALIGNERS.create("mfa", mfa_binary="definitely-not-a-real-mfa-binary")
    with pytest.raises(RuntimeError, match="not found on PATH"):
        aligner.align(_audio(), "hello world")


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("mfa") is None, reason="mfa CLI not installed")
def test_align_real_audio_with_installed_mfa() -> None:
    aligner = ALIGNERS.create("mfa")
    words = aligner.align(_audio(2.0), "hello world")
    assert [w.text for w in words]
    for earlier, later in zip(words, words[1:], strict=False):
        assert earlier.end_s <= later.start_s
