"""Shared fixture for Kaldi-style corpus loader tests (design §4.8).

LibriCSS and AMI are both commonly re-exported as Kaldi data dirs, so their
loader tests (test_libricss.py, test_ami.py) share one tiny fixture corpus
instead of each re-implementing the same wav.scp/segments/text/utt2spk setup.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def make_kaldi_corpus(
    tmp_path: Path, write_wav: Callable[..., None]
) -> Callable[[], Path]:
    """Build a one-recording, two-speaker, one-overlap Kaldi-style corpus dir."""

    def _make() -> Path:
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        write_wav(corpus_dir / "rec1.wav", duration_s=3.0)
        (corpus_dir / "wav.scp").write_text("rec1 rec1.wav\n")
        (corpus_dir / "segments").write_text("utt1 rec1 0.0 2.0\nutt2 rec1 1.0 3.0\n")
        (corpus_dir / "text").write_text("utt1 hello there\nutt2 hi friend\n")
        (corpus_dir / "utt2spk").write_text("utt1 spk_a\nutt2 spk_b\n")
        return corpus_dir

    return _make
