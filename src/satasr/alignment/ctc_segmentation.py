"""CTC-segmentation aligner — design §4.7's preferred approach for the
synthetic (TTS) audio this pipeline mostly generates.

CTC-segmentation aligns text to audio using a CTC-trained ASR model's frame
posteriors, giving much tighter word timestamps than the dependency-free
``ProportionalAligner``. This module is a STUB: the real implementation
needs torch and a wav2vec2-style CTC model, which are optional, heavy
dependencies (see ``pyproject.toml``'s extras). Importing this module must
stay cheap so registry discovery never requires them, so those imports
belong inside a method body, not at module scope, once implemented.
"""

from __future__ import annotations

from satasr.alignment.registry import ALIGNERS
from satasr.core.audio import AudioBuffer
from satasr.core.models import Word

_STUB_MESSAGE = (
    "CtcSegmentationAligner is not implemented yet. Forced alignment via "
    "CTC-segmentation needs torch plus a wav2vec2-style CTC model; tracked "
    "in the project's GitHub issue 'Implement CTC-segmentation aligner "
    "(design §4.7)'. Use the 'proportional' aligner in the meantime."
)


@ALIGNERS.register("ctc_segmentation")
class CtcSegmentationAligner:
    """Forced alignment via CTC-segmentation. Not yet implemented.

    Heavy dependencies (torch, a CTC/wav2vec2 model) belong inside
    :meth:`align`, not at module import time, so merely importing this
    module — e.g. for registry discovery — never requires them.
    """

    def align(self, audio: AudioBuffer, text: str) -> tuple[Word, ...]:
        # Real implementation will import torch + a wav2vec2 CTC model here,
        # run CTC-segmentation between `text` and `audio`, and return Words.
        raise NotImplementedError(_STUB_MESSAGE)
