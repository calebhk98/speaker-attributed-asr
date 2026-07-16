"""Concrete LoRA mechanics for :mod:`satasr.training.phase2` (design §5).

``LoraTrainStep`` is what "LoRA fine-tune (not full)" actually means in code:
constructing it wraps a checkpoint's weights in small trainable low-rank
adapters via ``peft`` and freezes every base parameter, then each call is one
teacher-forced optimizer step over the (still-frozen) base. It satisfies both
``TrainStep`` (``__call__``) and ``Checkpointer`` (``save``) from
``satasr.training.interfaces`` — one object plugs into both slots of
``run_phase2`` (rule 4: no need for a second class just to save).

Every torch/peft/transformers import stays inside a method body, per the
house style already used by ``model/dicow.py`` and
``alignment/ctc_segmentation.py`` — importing this module for its dataclass
or for :func:`load_lora_model`'s signature never requires those heavy,
optional, weight-downloading dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from satasr.core.audio import AudioBuffer
from satasr.model.interfaces import Model
from satasr.training.lora_eval_model import LoraEvalModel


@dataclass(frozen=True)
class LoraHyperparams:
    """LoRA adapter shape + optimization knobs (§5). Conservative defaults
    keep the trained adapter tiny relative to the ~918M frozen base."""

    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    learning_rate: float = 1e-4


class LoraTrainStep:
    """Wraps ``checkpoint_id`` in LoRA on construction; each call performs
    one optimizer step for a single clip's audio against its own SOT
    transcript (reusing the one serializer, rule 4, rather than a caller
    hand-building the target string)."""

    def __init__(
        self, checkpoint_id: str, hyperparams: LoraHyperparams, device: str = "cpu"
    ) -> None:
        from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
        )

        self._device = device
        self._processor = AutoProcessor.from_pretrained(
            checkpoint_id, trust_remote_code=True
        )
        base = AutoModelForSpeechSeq2Seq.from_pretrained(
            checkpoint_id, trust_remote_code=True
        ).to(device)
        peft_config = LoraConfig(
            r=hyperparams.rank,
            lora_alpha=hyperparams.alpha,
            lora_dropout=hyperparams.dropout,
            target_modules=list(hyperparams.target_modules),
        )
        # get_peft_model freezes every base parameter and marks only the
        # injected adapter matrices trainable — the "not full" of §5.
        self.model = get_peft_model(base, peft_config)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self._optimizer = self._optim(trainable, hyperparams.learning_rate)

    def __call__(self, audio: AudioBuffer, transcript: str) -> float:
        self._optimizer.zero_grad()
        inputs = self._processor(
            audio.samples, sampling_rate=audio.sample_rate, return_tensors="pt"
        ).to(self._device)
        labels = self._processor.tokenizer(
            transcript, return_tensors="pt"
        ).input_ids.to(self._device)
        loss = self.model(input_features=inputs.input_features, labels=labels).loss
        loss.backward()
        self._optimizer.step()
        return float(loss.item())

    def save(self, path: Path) -> None:
        """Persist only the trained adapter weights (never the frozen base)."""
        self.model.save_pretrained(str(path))

    @staticmethod
    def _optim(trainable: list[Any], learning_rate: float) -> Any:
        import torch  # type: ignore[import-not-found]

        return torch.optim.AdamW(trainable, lr=learning_rate)


def load_lora_model(
    checkpoint_id: str, adapter_dir: Path, device: str = "cpu"
) -> Model:
    """Reload a saved adapter onto a fresh copy of its base for evaluation —
    the save/load round trip a real Phase-2 run relies on."""
    from peft import PeftModel
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint_id, trust_remote_code=True)
    base = AutoModelForSpeechSeq2Seq.from_pretrained(
        checkpoint_id, trust_remote_code=True
    ).to(device)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).to(device)
    merged.eval()
    return LoraEvalModel(merged, processor, device)
