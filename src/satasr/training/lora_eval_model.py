"""``Model``-protocol wrapper around a loaded base+LoRA-adapter pair (§5).

Split out of :mod:`satasr.training.lora_backend` so each file stays under the
house's short-file guideline (CLAUDE.md rule 3). Mirrors
``model/dicow.py``'s ``predict`` shape: decode, then hand the raw text to the
one canonical SOT parser (rule 4) rather than re-rolling another parser here.
"""

from __future__ import annotations

from typing import Any

from satasr.core.audio import AudioBuffer
from satasr.core.models import PlacedUtterance
from satasr.format import parse_utterances


class LoraEvalModel:
    """Wraps an already-loaded (base + adapter) generation pair for eval."""

    def __init__(self, model: Any, processor: Any, device: str) -> None:
        self._model = model
        self._processor = processor
        self._device = device

    def predict(self, audio: AudioBuffer) -> tuple[PlacedUtterance, ...]:
        """Generate the SOT string for ``audio`` and parse it (§4.7)."""
        import torch  # type: ignore[import-not-found]

        inputs = self._processor(
            audio.samples, sampling_rate=audio.sample_rate, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            generated_ids = self._model.generate(input_features=inputs.input_features)
        raw_text: str = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return parse_utterances(raw_text)
