"""Training Phase 1 — full fine-tune on synthetic data (design §5, §4.1).

Trains from a DiCoW/SA-DiCoW-style checkpoint on the synthetic dataset a
:class:`~satasr.pipeline.dataset_writer.DatasetWriter` wrote. Per CLAUDE.md
rule 5, this loop depends only on the ``Model`` Protocol
(:mod:`satasr.model.interfaces`) for validation — never on the concrete DiCoW
class — plus the two Protocols in :mod:`satasr.training.interfaces` that keep
the optimizer step and checkpoint format out of this module entirely: the
caller supplies a ``TrainStep`` closure (forward + loss + backward + update,
including any multi-GPU placement across the dual 3090s, §3) and a
``Checkpointer``. This module never imports torch.

**Split discipline (§4.1):** :func:`run_phase1` reads only ``train`` and
``val`` manifest rows — ``test`` rows are filtered out in :func:`_split_rows`
before the loop starts and are never read again here. :func:`evaluate_test_split`
scores the held-out test split separately, meant to be called exactly once,
after training has finished and a checkpoint has been chosen from validation
alone.

**Overlap caveat (§7):** synthetic overlap intensity is a mixing-time property
(``MixingConfig``), not something this loop controls — don't chase a weak
validation metric by re-mixing toward heavier overlap; it helps ASR but
measurably hurts diarization.
"""

from __future__ import annotations

import math
from pathlib import Path

from satasr.eval.cpwer import cpwer
from satasr.format import parse_utterances
from satasr.model import Model
from satasr.pipeline.dataset_writer import ManifestRow, read_audio, read_manifest
from satasr.training.config import EpochStats, MetricFn, Phase1Config, Phase1Result
from satasr.training.interfaces import Checkpointer, TrainStep


def run_phase1(
    model: Model,
    train_step: TrainStep,
    checkpointer: Checkpointer,
    dataset_root: Path | str,
    config: Phase1Config,
) -> Phase1Result:
    """Full fine-tune loop: train on ``train``, pick checkpoints on ``val``.

    Never reads ``test`` rows (§4.1). Early-stops once ``val_metric`` hasn't
    improved by at least ``config.min_delta`` for ``config.patience``
    consecutive epochs; otherwise runs the full ``config.max_epochs``.
    """
    train_rows, val_rows = _split_rows(dataset_root)
    if not train_rows:
        raise ValueError("no train rows in manifest")
    if not val_rows:
        raise ValueError("no val rows in manifest")

    history: list[EpochStats] = []
    best_metric = math.inf
    best_checkpoint: Path | None = None
    stall = 0
    for epoch in range(1, config.max_epochs + 1):
        train_loss = _train_one_epoch(train_step, dataset_root, train_rows)
        val_metric = _evaluate(model, dataset_root, val_rows, config.metric)
        history.append(EpochStats(epoch, train_loss, val_metric))
        improved = val_metric < best_metric - config.min_delta
        if not improved:
            stall += 1
            if stall >= config.patience:
                return Phase1Result(best_checkpoint, best_metric, tuple(history), True)
            continue
        best_metric = val_metric
        best_checkpoint = config.checkpoint_dir / f"epoch-{epoch:03d}.ckpt"
        checkpointer.save(best_checkpoint)
        stall = 0
    return Phase1Result(best_checkpoint, best_metric, tuple(history), False)


def evaluate_test_split(
    model: Model, dataset_root: Path | str, metric: MetricFn = cpwer
) -> float:
    """Score the held-out synthetic test split — call this exactly once (§4.1).

    Deliberately separate from :func:`run_phase1`: the test split must never
    influence training or checkpoint selection, only the final reported score.
    """
    rows = tuple(r for r in read_manifest(dataset_root) if r.split == "test")
    if not rows:
        raise ValueError("no test rows in manifest")
    return _evaluate(model, dataset_root, rows, metric)


def _split_rows(
    dataset_root: Path | str,
) -> tuple[tuple[ManifestRow, ...], tuple[ManifestRow, ...]]:
    """Train/val rows only — ``test`` rows are dropped right here (§4.1)."""
    rows = read_manifest(dataset_root)
    train = tuple(r for r in rows if r.split == "train")
    val = tuple(r for r in rows if r.split == "val")
    return train, val


def _train_one_epoch(
    train_step: TrainStep, dataset_root: Path | str, rows: tuple[ManifestRow, ...]
) -> float:
    """Run one optimizer step per training row; return the mean loss."""
    losses = [train_step(read_audio(dataset_root, row), row.transcript) for row in rows]
    return sum(losses) / len(losses)


def _evaluate(
    model: Model,
    dataset_root: Path | str,
    rows: tuple[ManifestRow, ...],
    metric: MetricFn,
) -> float:
    """Mean ``metric`` over ``rows``, predicting with ``model`` (never trains)."""
    scores = [_score_row(model, dataset_root, row, metric) for row in rows]
    return sum(scores) / len(scores)


def _score_row(
    model: Model, dataset_root: Path | str, row: ManifestRow, metric: MetricFn
) -> float:
    audio = read_audio(dataset_root, row)
    reference = parse_utterances(row.transcript)
    hypothesis = model.predict(audio)
    return metric(reference, hypothesis)
