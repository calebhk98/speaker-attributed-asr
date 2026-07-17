"""DiCoW / SA-DiCoW wrapper: diarization-conditioned Whisper-large-v3-turbo,
fine-tuned for multi-talker, speaker-attributed ASR with serialized output
(design §2). Implements the :class:`~satasr.model.interfaces.Model` Protocol
so training/eval code depends only on that abstraction, never on this
concrete checkpoint (CLAUDE.md rule 5).

Checkpoint: published by BUT Speech@FIT (inference code at
https://github.com/BUTSpeechFIT/DiCoW) on the Hugging Face Hub as
``BUT-FIT/DiCoW_v3_MLC``, a ~918M-parameter fine-tune of
``openai/whisper-large-v3-turbo``. Loading and inference need ``torch`` and
``transformers`` (with ``trust_remote_code=True`` — the repo ships its own
``modeling_dicow.py``), both heavy, GPU-hungry optional dependencies not
installed by default in this environment. Per CLAUDE.md rule 5 and the house
style every TTS engine already follows (see ``tts/engines/kokoro.py``), those
imports live strictly inside the methods that need them, and weights load
lazily on first :meth:`DiCoWModel.predict` rather than in ``__init__`` — so
constructing this object, and importing this module, stays fast and
dependency-free.
"""

from __future__ import annotations

from typing import Any

from satasr.core.audio import AudioBuffer
from satasr.core.models import PlacedUtterance
from satasr.format import parse_utterances

#: Published checkpoint on the Hugging Face Hub (see module docstring).
DICOW_MODEL_ID = "BUT-FIT/DiCoW_v3_MLC"


class DiCoWModel:
    """`Model`-protocol wrapper around the DiCoW checkpoint (design §2)."""

    name = "dicow"

    def __init__(self, model_id: str = DICOW_MODEL_ID, device: str = "cpu") -> None:
        self.model_id = model_id
        self.device = device
        self._processor: Any = None
        self._model: Any = None

    def predict(self, audio: AudioBuffer) -> tuple[PlacedUtterance, ...]:
        """Run DiCoW on ``audio`` and decode its serialized-output string.

        Reuses :func:`satasr.format.parse_utterances` — the one place the SOT
        line grammar is defined (CLAUDE.md rule 4) — so the returned tuple is
        exactly the type :class:`~satasr.core.models.MixedClip.utterances`
        carries, never a hand-rolled re-parse of the transcript.
        """
        raw_text = self._generate(audio)
        return parse_utterances(raw_text)

    def _generate(self, audio: AudioBuffer) -> str:
        """Load the checkpoint if needed, then decode ``audio`` to raw text."""
        self._load()
        import torch  # type: ignore[import-not-found]

        inputs = self._processor(
            audio.samples, sampling_rate=audio.sample_rate, return_tensors="pt"
        ).to(self.device)
        stno_mask = self._default_stno_mask(inputs.input_features.shape[-1])
        with torch.no_grad():
            generated_ids = self._model.generate(
                input_features=inputs.input_features, stno_mask=stno_mask
            )
        decoded: str = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return decoded

    def _default_stno_mask(self, num_frames: int) -> Any:
        """A permissive placeholder diarization-conditioning mask.

        DiCoW conditions on a per-frame Silence/Target/Non-target/Overlap
        (STNO) mask produced by an upstream diarizer (see ``encoder.py``'s
        ``FDDT`` module in the published checkpoint). This wrapper has no
        diarizer wired in yet (out of scope for this issue, tracked under the
        parent epic #1), so it defaults to "always Target speaker" for the
        whole clip — the correct behaviour for a single-speaker utterance and
        a documented simplification for overlapping audio.
        """
        import torch

        # Class index 1 == "Target" in the S/T/N/O ordering used by FDDT.
        mask = torch.zeros(1, num_frames, 4)
        mask[..., 1] = 1.0
        return mask

    def _load(self) -> None:
        """Lazily fetch the processor and model from the Hub (idempotent)."""
        if self._model is not None:
            return
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
        )

        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, trust_remote_code=True
        ).to(self.device)
        self._model.eval()
