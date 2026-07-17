"""Montreal Forced Aligner (MFA) — classic HMM forced alignment, the design's
pick for *real recorded speech* (design §4.7 / §8): reliable on natural
audio, less so on TTS output, where ``CtcSegmentationAligner`` is preferred
instead. Used mainly in Training Phase 2 (real data), per design §1 (#1).

MFA ships as a separate Kaldi-based CLI, not a pip package with weights.
Install once, outside this repo, then download a language's models:

    conda create -n aligner -c conda-forge montreal-forced-aligner
    conda activate aligner
    mfa model download acoustic english_us_arpa
    mfa model download dictionary english_us_arpa

This module only shells out to the already-installed ``mfa`` binary and
writes/reads plain files, so the subprocess/tempfile/wave imports needed to
do that stay inside :meth:`MfaAligner.align` — importing this module (e.g.
for registry discovery) never requires MFA, Kaldi, or a model download.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from satasr.alignment.registry import ALIGNERS
from satasr.core.audio import AudioBuffer
from satasr.core.models import Word

# Labels MFA emits for non-speech intervals in the "words" tier.
_SILENCE_LABELS = {"", "sil", "sp", "spn"}

_INSTALL_HINT = (
    "MfaAligner requires the external 'mfa' CLI (Montreal Forced Aligner), "
    "which was not found on PATH. Install it with e.g. "
    "'conda create -n aligner -c conda-forge montreal-forced-aligner', then "
    "'mfa model download acoustic <name>' and 'mfa model download "
    "dictionary <name>' for the language you need."
)

_ITEM_RE = re.compile(r"item \[\d+\]:(.*?)(?=item \[\d+\]:|\Z)", re.S)
_TIER_NAME_RE = re.compile(r'name = "([^"]*)"')
_INTERVAL_RE = re.compile(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"')


@ALIGNERS.register("mfa")
class MfaAligner:
    """Forced alignment via the Montreal Forced Aligner CLI (design §4.7).

    ``dictionary`` and ``acoustic_model`` name pretrained MFA models (e.g.
    "english_us_arpa") that must already be downloaded via
    ``mfa model download``; this class never fetches them itself.
    """

    def __init__(
        self,
        dictionary: str = "english_us_arpa",
        acoustic_model: str = "english_us_arpa",
        mfa_binary: str = "mfa",
    ) -> None:
        self._dictionary = dictionary
        self._acoustic_model = acoustic_model
        self._mfa_binary = mfa_binary

    def align(self, audio: AudioBuffer, text: str) -> tuple[Word, ...]:
        if not text.split():
            raise ValueError("cannot align empty text")

        import shutil

        if shutil.which(self._mfa_binary) is None:
            raise RuntimeError(_INSTALL_HINT)

        import tempfile

        with tempfile.TemporaryDirectory() as workdir:
            corpus_dir, output_dir = _prepare_corpus(Path(workdir), audio, text)
            self._run_mfa(corpus_dir, output_dir)
            grid_path = output_dir / "utterance.TextGrid"
            return _parse_textgrid_words(grid_path.read_text(encoding="utf-8"))

    def _run_mfa(self, corpus_dir: Path, output_dir: Path) -> None:
        """Invoke ``mfa align`` on a one-utterance, single-speaker corpus."""
        import subprocess

        command = [
            self._mfa_binary,
            "align",
            "--clean",
            "--single_speaker",
            str(corpus_dir),
            self._dictionary,
            self._acoustic_model,
            str(output_dir),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"mfa align failed (exit {result.returncode}): {result.stderr}"
            )


def _prepare_corpus(root: Path, audio: AudioBuffer, text: str) -> tuple[Path, Path]:
    """Lay out the one-file corpus MFA's CLI expects, plus an empty output dir."""
    corpus_dir, output_dir = root / "corpus", root / "output"
    corpus_dir.mkdir()
    output_dir.mkdir()
    _write_wav(corpus_dir / "utterance.wav", audio)
    (corpus_dir / "utterance.lab").write_text(text, encoding="utf-8")
    return corpus_dir, output_dir


def _write_wav(path: Path, audio: AudioBuffer) -> None:
    """Write ``audio`` as 16-bit PCM mono — the format MFA's corpus reader wants."""
    import wave

    pcm16 = (np.clip(audio.samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(audio.sample_rate)
        handle.writeframes(pcm16.tobytes())


def _parse_textgrid_words(text: str) -> tuple[Word, ...]:
    """Pull the "words" interval tier out of an MFA long-format TextGrid.

    A minimal hand-written parser rather than a third-party TextGrid library,
    since MFA's output format is a small, stable grammar and this keeps the
    module's only real dependency on the ``mfa`` CLI itself.
    """
    tier = _find_tier(text, "words")
    words = [
        Word(label, float(xmin), float(xmax))
        for xmin, xmax, label in _INTERVAL_RE.findall(tier)
        if label.strip().lower() not in _SILENCE_LABELS
    ]
    if not words:
        raise RuntimeError("mfa produced no aligned words in the 'words' tier")
    return tuple(words)


def _find_tier(text: str, name: str) -> str:
    blocks: list[str] = _ITEM_RE.findall(text)
    for block in blocks:
        match = _TIER_NAME_RE.search(block)
        if match is not None and match.group(1) == name:
            return block
    raise RuntimeError(f"TextGrid has no tier named {name!r}")
