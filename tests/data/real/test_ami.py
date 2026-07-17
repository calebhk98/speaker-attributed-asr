"""Fast tests for the AMI loader (design §4.8).

CAVEAT: the AMI Meeting Corpus license has not been verified for use in this
project (§4.8, open question 3) — see ``satasr.data.real.ami.AMI_LICENSE_CAVEAT``.
These tests only ever exercise tiny synthetic fixtures, never real AMI audio.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from satasr.data.real.ami import AMI_LICENSE_CAVEAT, load_ami_corpus


def test_loads_one_mixed_clip_per_recording(
    make_kaldi_corpus: Callable[[], Path],
) -> None:
    corpus_dir = make_kaldi_corpus()

    clips = load_ami_corpus(corpus_dir)

    assert len(clips) == 1
    assert clips[0].metadata["source"] == "ami"
    assert clips[0].counts.total == 2
    assert clips[0].counts.max_simultaneous == 2


def test_license_caveat_is_documented() -> None:
    assert "license" in AMI_LICENSE_CAVEAT.lower()
    assert "4.8" in AMI_LICENSE_CAVEAT
