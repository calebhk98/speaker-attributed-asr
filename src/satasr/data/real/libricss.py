"""LibriCSS loader (design §4.8).

LibriCSS re-plays concatenated LibriSpeech utterances through loudspeakers and
records them with a microphone array, so — unlike LibriSpeechMix's constructed
mixtures — the overlap timing here is genuine acoustic overlap, even though the
source speech is the same public-domain LibriVox material. Distributions and
recipes commonly re-export it as a Kaldi data dir, which
:mod:`satasr.data.real.kaldi_format` already knows how to read.
"""

from __future__ import annotations

from pathlib import Path

from satasr.core.models import MixedClip
from satasr.data.real.kaldi_format import load_corpus


def load_libricss_corpus(corpus_dir: Path) -> tuple[MixedClip, ...]:
    """Load a Kaldi-style LibriCSS export (wav.scp/segments/text/utt2spk)."""
    return load_corpus(corpus_dir, source="libricss")
