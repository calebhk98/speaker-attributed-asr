"""AMI Meeting Corpus loader (design §4.8) — genuine spontaneous overlap.

AMI is the strongest source of real, spontaneous turn-taking dynamics (as
opposed to LibriSpeechMix's constructed mixtures or LibriCSS's scripted
re-play), which is exactly what Phase 2 LoRA training wants. Standard recipes
commonly re-export it as a Kaldi data dir, reusing the same reader as LibriCSS.

CAVEAT: the AMI Meeting Corpus license has not yet been verified for use in
this project (design §4.8, open question 3). This loader exists so real AMI
data is one config change away the moment that verification lands — do not
point it at real AMI audio before then.
"""

from __future__ import annotations

from pathlib import Path

from satasr.core.models import MixedClip
from satasr.data.real.kaldi_format import load_corpus

AMI_LICENSE_CAVEAT = (
    "AMI Meeting Corpus license must be verified before use "
    "(design doc section 4.8, open question 3)."
)


def load_ami_corpus(corpus_dir: Path) -> tuple[MixedClip, ...]:
    """Load a Kaldi-style AMI export (wav.scp/segments/text/utt2spk) dir."""
    return load_corpus(corpus_dir, source="ami")
