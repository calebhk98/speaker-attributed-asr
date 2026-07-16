"""Training Phase 2 — LoRA fine-tune on real human speech (design §5, §4.8).

Phase 1 (full fine-tune on synthetic audio, #37) is a sibling module written
concurrently; this one never imports it. Mirroring ``phase1.py``'s
dependency-inversion shape (CLAUDE.md rule 5), :func:`run_phase2` depends only
on ``satasr.model.Model`` for prediction and the two Protocols in
``satasr.training.interfaces`` (``TrainStep``, ``Checkpointer``) for the
optimizer step and adapter persistence — never on torch/PEFT directly, so this
module stays weight-free to import. The concrete LoRA mechanics (wrap in
low-rank adapters, freeze every base weight, teacher-forced step) live in
:mod:`satasr.training.lora_backend`, whose ``LoraTrainStep`` satisfies both
Protocols and is what a real run plugs in here.

**Split discipline (§4.1), reused not reinvented:** real clips are bucketed
into LoRA-train vs. held-out-test with the exact same deterministic hash
:func:`~satasr.pipeline.dataset_writer.assign_split` uses for the synthetic
manifest, so a real clip id always resolves to the same split and the two
buckets can never overlap — no separate leakage check is needed.

**The two required evaluations (§5):** ``held_out_real_metric`` scores the
held-out real split never touched during training; ``synthetic_regression_
metric`` re-scores the *same* trained model against the Phase-1 synthetic
test set, to confirm adapting to the smaller real sample didn't regress
synthetic performance. Both use ``satasr.eval.cpwer.cpwer`` by default (the
project's one multi-speaker metric, rule 4).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from satasr.core.models import MixedClip
from satasr.eval.cpwer import cpwer
from satasr.format import serialize_clip
from satasr.model.interfaces import Model
from satasr.pipeline.dataset_writer import SplitRatios, assign_split
from satasr.training.config import MetricFn
from satasr.training.interfaces import Checkpointer, TrainStep

_DEFAULT_SPLIT_RATIOS = SplitRatios()


@dataclass(frozen=True)
class Phase2Config:
    """Tunables for one Phase 2 run (§5) — the one place these knobs live."""

    adapter_dir: Path
    epochs: int = 1
    metric: MetricFn = field(default=cpwer)


@dataclass(frozen=True)
class Phase2Result:
    """What a completed Phase 2 run produced: where the adapter landed and
    the two evaluation numbers §5 requires."""

    adapter_dir: Path
    held_out_real_metric: float
    synthetic_regression_metric: float
    num_train_clips: int
    num_held_out_clips: int


def run_phase2(
    model: Model,
    train_step: TrainStep,
    checkpointer: Checkpointer,
    real_clips: Mapping[str, MixedClip],
    synthetic_test: Sequence[MixedClip],
    config: Phase2Config,
    split_ratios: SplitRatios = _DEFAULT_SPLIT_RATIOS,
) -> Phase2Result:
    """LoRA fine-tune loop: train on real clips, save the adapter, then score
    it on the held-out real split and the Phase-1 synthetic test set (§5).

    ``model`` must reflect ``train_step``'s live weights (the same object a
    caller wired both around, per ``phase1.py``'s established pattern) so
    evaluation sees whatever the training loop just updated.
    """
    train_clips, held_out_clips = _split_real_clips(real_clips, split_ratios)
    if not train_clips:
        raise ValueError("no real clips assigned to the LoRA train split")
    if not held_out_clips:
        raise ValueError("no real clips assigned to the held-out real split")

    for _ in range(config.epochs):
        for clip in train_clips:
            train_step(clip.audio, serialize_clip(clip))
    checkpointer.save(config.adapter_dir)

    return Phase2Result(
        adapter_dir=config.adapter_dir,
        held_out_real_metric=_evaluate(model, held_out_clips, config.metric),
        synthetic_regression_metric=_evaluate(model, synthetic_test, config.metric),
        num_train_clips=len(train_clips),
        num_held_out_clips=len(held_out_clips),
    )


def _split_real_clips(
    real_clips: Mapping[str, MixedClip], ratios: SplitRatios
) -> tuple[list[MixedClip], list[MixedClip]]:
    """Bucket real clips by ``assign_split`` (§4.1): "test" is held out, every
    other bucket ("train"/"val") is trained on — Phase 2 has no separate
    validation step, unlike Phase 1's checkpoint-selection loop."""
    train: list[MixedClip] = []
    held_out: list[MixedClip] = []
    for clip_id, clip in real_clips.items():
        bucket = held_out if assign_split(clip_id, ratios) == "test" else train
        bucket.append(clip)
    return train, held_out


def _evaluate(model: Model, clips: Sequence[MixedClip], metric: MetricFn) -> float:
    """Mean ``metric`` over ``clips``, predicting with ``model`` (never trains)."""
    if not clips:
        raise ValueError("cannot evaluate over an empty clip set")
    scores = [metric(clip.utterances, model.predict(clip.audio)) for clip in clips]
    return sum(scores) / len(scores)
