"""CTC-segmentation aligner — design §4.7's preferred approach for the
synthetic (TTS) audio this pipeline mostly generates.

CTC-segmentation (Kuerzinger et al., 2020) aligns text to audio using a
CTC-trained ASR model's per-frame character posteriors and a Viterbi-style
search, giving much tighter word timestamps than the dependency-free
``ProportionalAligner``. Torch, transformers, and the ``ctc-segmentation``
package are heavy, optional dependencies, so every import of them stays
inside a method body — importing this module for registry discovery never
requires them.
"""

from __future__ import annotations

import re

import numpy as np
from numpy.typing import NDArray

from satasr.alignment.registry import ALIGNERS
from satasr.core.audio import AudioBuffer
from satasr.core.models import Word

#: Default wav2vec2 CTC checkpoint: English, 16 kHz, upper-case + "|"
#: word-delimiter vocabulary.
DEFAULT_MODEL_NAME = "facebook/wav2vec2-base-960h"

#: (start_s, end_s, confidence), one per word — the shape
#: ``ctc_segmentation.determine_utterance_segments`` returns.
_Segment = tuple[float, float, float]


@ALIGNERS.register("ctc_segmentation")
class CtcSegmentationAligner:
    """Forced alignment via CTC-segmentation against a wav2vec2 CTC model.

    Each word in ``text`` is passed to the ``ctc-segmentation`` library as
    its own "utterance", so the library's search returns word-level
    start/end times rather than one span for the whole sentence.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name

    def align(self, audio: AudioBuffer, text: str) -> tuple[Word, ...]:
        """Return word-level timestamps for ``text`` against ``audio``.

        The empty-text guard runs before any heavy import, so it (and the
        registration test) can be exercised without torch installed.
        """
        tokens = text.split()
        if not tokens:
            raise ValueError("cannot align empty text")

        log_probs, char_list = self._log_probs(audio)
        normalized = [_normalize_for_vocab(token) for token in tokens]
        segments = self._segments(log_probs, normalized, char_list, audio)
        return _words_from_segments(tokens, segments, audio.duration_s)

    def _log_probs(self, audio: AudioBuffer) -> tuple[NDArray[np.float32], list[str]]:
        """Run the wav2vec2 CTC model once; return frame log-posteriors and
        its character vocabulary (ordered by index, for ``ctc_segmentation``).
        """
        import torch  # type: ignore
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # type: ignore

        processor = Wav2Vec2Processor.from_pretrained(self._model_name)
        model = Wav2Vec2ForCTC.from_pretrained(self._model_name)
        model.eval()

        inputs = processor(
            audio.samples, sampling_rate=audio.sample_rate, return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(inputs.input_values).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1).numpy()

        vocab = processor.tokenizer.get_vocab()
        char_list = [str(tok) for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
        return np.asarray(log_probs, dtype=np.float32), char_list

    @staticmethod
    def _segments(
        log_probs: NDArray[np.float32],
        normalized: list[str],
        char_list: list[str],
        audio: AudioBuffer,
    ) -> list[_Segment]:
        """Run ctc-segmentation's Viterbi-style search over ``log_probs``,
        treating each entry of ``normalized`` as its own utterance so the
        result is one (start, end, confidence) segment per word.
        """
        import ctc_segmentation as ctcseg  # type: ignore

        config = ctcseg.CtcSegmentationParameters(char_list=char_list)
        frame_count = log_probs.shape[0]
        config.index_duration = audio.num_samples / frame_count / audio.sample_rate

        ground_truth, utt_begin_indices = ctcseg.prepare_text(config, normalized)
        timings, char_probs, _ = ctcseg.ctc_segmentation(
            config, log_probs, ground_truth
        )
        raw = ctcseg.determine_utterance_segments(
            config, utt_begin_indices, char_probs, timings, normalized
        )
        return [(float(start), float(end), float(score)) for start, end, score in raw]


def _normalize_for_vocab(token: str) -> str:
    """Upper-case ``token`` and drop characters outside wav2vec2-base-960h's
    vocabulary (A-Z and apostrophe), so ``prepare_text`` can map every
    character onto one of the model's CTC output classes.
    """
    cleaned = re.sub(r"[^A-Za-z']", "", token).upper()
    if not cleaned:
        raise ValueError(
            f"token {token!r} has no characters alignable by {DEFAULT_MODEL_NAME}"
        )
    return cleaned


def _words_from_segments(
    tokens: list[str], segments: list[_Segment], duration_s: float
) -> tuple[Word, ...]:
    """Clamp raw CTC-segmentation output to the ``Aligner`` contract: words
    stay monotonic, non-overlapping, and within ``[0, duration_s]`` even if
    the model's segmentation drifts slightly past a clip boundary.
    """
    words: list[Word] = []
    cursor = 0.0
    for token, (start, end, _score) in zip(tokens, segments, strict=True):
        clamped_start = min(max(start, cursor), duration_s)
        clamped_end = min(max(end, clamped_start), duration_s)
        words.append(Word(token, clamped_start, clamped_end))
        cursor = clamped_end
    return tuple(words)
