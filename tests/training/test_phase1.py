"""Fast tests for the Phase 1 training loop (design §5, §4.1).

Drives the loop with a tiny fake ``Model``/``TrainStep``/``Checkpointer`` — no
torch, no GPU, no real weights — against a real tmp dataset written by
:class:`~satasr.pipeline.dataset_writer.DatasetWriter`. Any real-weights
training belongs in a ``@pytest.mark.slow`` test elsewhere, not here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from satasr.core.audio import AudioBuffer
from satasr.core.models import MixedClip, PlacedUtterance, SpeakerCounts
from satasr.format import parse_utterances
from satasr.model import Model
from satasr.pipeline.dataset_writer import (
    DatasetWriter,
    SplitRatios,
    read_audio,
    read_manifest,
)
from satasr.training.config import Phase1Config
from satasr.training.interfaces import Checkpointer, TrainStep
from satasr.training.phase1 import evaluate_test_split, run_phase1

_SAMPLE_RATE = 16_000
_ALL_TRAIN = SplitRatios(train=1.0, val=0.0, test=0.0)
_ALL_VAL = SplitRatios(train=0.0, val=1.0, test=0.0)
_ALL_TEST = SplitRatios(train=0.0, val=0.0, test=1.0)
_TEXT = "hello world"


def _silence_clip(text: str = _TEXT) -> MixedClip:
    samples = np.zeros(_SAMPLE_RATE // 4, dtype=np.float32)
    utterances = (PlacedUtterance("a", 0.0, 1.0, text),)
    return MixedClip(
        audio=AudioBuffer(samples, _SAMPLE_RATE),
        utterances=utterances,
        counts=SpeakerCounts(total=1, max_simultaneous=1),
    )


def _build_dataset(root: Path, n_train: int = 2, with_val: bool = True) -> None:
    for i in range(n_train):
        DatasetWriter(root, _ALL_TRAIN).write(f"train-{i}", _silence_clip())
    if with_val:
        DatasetWriter(root, _ALL_VAL).write("val-0", _silence_clip())
    DatasetWriter(root, _ALL_TEST).write("test-0", _silence_clip())


class _LearningModel:
    """Predicts ``wrong`` until ``calls['n']`` reaches ``threshold``, then
    predicts ``correct`` — simulates a model that improves as training
    (tracked externally, via the shared ``calls`` counter) progresses."""

    def __init__(
        self,
        calls: dict[str, int],
        threshold: int,
        correct: tuple[PlacedUtterance, ...],
        wrong: tuple[PlacedUtterance, ...],
    ) -> None:
        self._calls = calls
        self._threshold = threshold
        self._correct = correct
        self._wrong = wrong

    def predict(self, audio: AudioBuffer) -> tuple[PlacedUtterance, ...]:
        if self._calls["n"] >= self._threshold:
            return self._correct
        return self._wrong


class _RecordingCheckpointer:
    """Records every path it was asked to save, instead of writing anything."""

    def __init__(self) -> None:
        self.saved: list[Path] = []

    def save(self, path: Path) -> None:
        self.saved.append(path)


def _counting_train_step(calls: dict[str, int]) -> TrainStep:
    def train_step(audio: AudioBuffer, transcript: str) -> float:
        calls["n"] += 1
        return 1.0

    return train_step


def _reference() -> tuple[PlacedUtterance, ...]:
    """The exact utterance every ``_silence_clip()`` transcript parses back to."""
    return (PlacedUtterance("a", 0.0, 1.0, _TEXT),)


_WRONG = (PlacedUtterance("a", 0.0, 1.0, "nope nope"),)


def test_fakes_satisfy_the_protocols_the_loop_depends_on() -> None:
    calls = {"n": 0}
    model = _LearningModel(calls, threshold=1, correct=_reference(), wrong=_WRONG)

    assert isinstance(model, Model)
    assert isinstance(_counting_train_step(calls), TrainStep)
    assert isinstance(_RecordingCheckpointer(), Checkpointer)


def test_run_phase1_iterates_checkpoints_and_early_stops_on_val(
    tmp_path: Path,
) -> None:
    _build_dataset(tmp_path, n_train=2)
    calls = {"n": 0}
    # Two train rows/epoch; needs two full epochs before predictions turn correct.
    model = _LearningModel(calls, threshold=4, correct=_reference(), wrong=_WRONG)
    checkpointer = _RecordingCheckpointer()
    config = Phase1Config(checkpoint_dir=tmp_path / "ckpt", max_epochs=10, patience=2)

    result = run_phase1(
        model, _counting_train_step(calls), checkpointer, tmp_path, config
    )

    # Epoch 1: wrong (cpwer 1.0, still an improvement over +inf) -> checkpoint.
    # Epoch 2: correct (cpwer 0.0, improved further) -> checkpoint.
    # Epochs 3-4: no further improvement -> patience=2 exhausted, stop early.
    assert [e.val_metric for e in result.history] == [1.0, 0.0, 0.0, 0.0]
    assert result.stopped_early is True
    assert result.best_val_metric == 0.0
    assert result.best_checkpoint == tmp_path / "ckpt" / "epoch-002.ckpt"
    assert checkpointer.saved == [
        tmp_path / "ckpt" / "epoch-001.ckpt",
        tmp_path / "ckpt" / "epoch-002.ckpt",
    ]


def test_run_phase1_runs_full_max_epochs_when_never_early_stopping(
    tmp_path: Path,
) -> None:
    _build_dataset(tmp_path, n_train=1)
    calls = {"n": 0}
    model = _LearningModel(calls, threshold=0, correct=_reference(), wrong=_WRONG)
    config = Phase1Config(checkpoint_dir=tmp_path / "ckpt", max_epochs=3, patience=99)

    result = run_phase1(
        model, _counting_train_step(calls), _RecordingCheckpointer(), tmp_path, config
    )

    assert len(result.history) == 3
    assert result.stopped_early is False


def test_run_phase1_never_reads_the_test_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_dataset(tmp_path, n_train=1)
    calls = {"n": 0}
    model = _LearningModel(calls, threshold=0, correct=_reference(), wrong=_WRONG)

    def guarded_read_audio(root: Path, row: object) -> AudioBuffer:
        assert row.split != "test"  # type: ignore[attr-defined]
        return read_audio(root, row)  # type: ignore[arg-type]

    monkeypatch.setattr("satasr.training.phase1.read_audio", guarded_read_audio)
    config = Phase1Config(checkpoint_dir=tmp_path / "ckpt", max_epochs=2, patience=99)

    run_phase1(
        model, _counting_train_step(calls), _RecordingCheckpointer(), tmp_path, config
    )


def test_run_phase1_requires_train_rows(tmp_path: Path) -> None:
    DatasetWriter(tmp_path, _ALL_VAL).write("val-0", _silence_clip())
    DatasetWriter(tmp_path, _ALL_TEST).write("test-0", _silence_clip())
    model = _LearningModel({"n": 0}, threshold=0, correct=_reference(), wrong=_WRONG)
    config = Phase1Config(checkpoint_dir=tmp_path / "ckpt")

    with pytest.raises(ValueError, match="no train rows"):
        run_phase1(model, lambda a, t: 0.0, _RecordingCheckpointer(), tmp_path, config)


def test_run_phase1_requires_val_rows(tmp_path: Path) -> None:
    DatasetWriter(tmp_path, _ALL_TRAIN).write("train-0", _silence_clip())
    DatasetWriter(tmp_path, _ALL_TEST).write("test-0", _silence_clip())
    model = _LearningModel({"n": 0}, threshold=0, correct=_reference(), wrong=_WRONG)
    config = Phase1Config(checkpoint_dir=tmp_path / "ckpt")

    with pytest.raises(ValueError, match="no val rows"):
        run_phase1(model, lambda a, t: 0.0, _RecordingCheckpointer(), tmp_path, config)


def test_evaluate_test_split_scores_only_the_test_rows(tmp_path: Path) -> None:
    _build_dataset(tmp_path, n_train=1)
    model = _LearningModel({"n": 999}, threshold=0, correct=_reference(), wrong=_WRONG)

    score = evaluate_test_split(model, tmp_path)

    assert score == 0.0


def test_evaluate_test_split_requires_test_rows(tmp_path: Path) -> None:
    DatasetWriter(tmp_path, _ALL_TRAIN).write("train-0", _silence_clip())
    model = _LearningModel({"n": 0}, threshold=0, correct=_reference(), wrong=_WRONG)

    with pytest.raises(ValueError, match="no test rows"):
        evaluate_test_split(model, tmp_path)


def test_reference_fixture_matches_what_the_manifest_actually_writes(
    tmp_path: Path,
) -> None:
    # Sanity check that the manifest's transcript really does round-trip to
    # the same reference the fakes above assume, so the cpwer=0.0/1.0
    # assertions aren't relying on an untested assumption.
    DatasetWriter(tmp_path, _ALL_VAL).write("val-0", _silence_clip())
    (written,) = read_manifest(tmp_path)

    assert parse_utterances(written.transcript) == _reference()
