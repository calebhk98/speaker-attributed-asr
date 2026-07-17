"""Loaders for real overlapping-speech corpora (design §4.8).

Covers LibriSpeechMix (:mod:`satasr.data.real.librispeech_mix`), LibriCSS
(:mod:`satasr.data.real.libricss`) and AMI (:mod:`satasr.data.real.ami`). Every
loader returns the same :class:`~satasr.core.models.MixedClip` schema the
synthetic pipeline uses, so training code never has to branch on source.

CAVEAT: the AMI Meeting Corpus license has not been verified for use in this
project (§4.8, open question 3) — see
:data:`satasr.data.real.ami.AMI_LICENSE_CAVEAT` before ingesting real AMI audio.
"""

from __future__ import annotations
