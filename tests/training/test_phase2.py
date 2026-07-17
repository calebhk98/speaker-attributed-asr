"""Fast tests for the Phase 2 LoRA training loop (design §5, §4.8).

Drives the loop with a tiny fake ``TrainStep``/``Checkpointer``/``Model`` — no
torch, no GPU, no real weights — against real clips loaded through the actual
real-audio corpus loader added in #34
(:func:`satasr.data.real.kaldi_format.load_corpus`) plus one directly-built
synthetic clip for the regression recheck. Real-weights LoRA training is a
``@pytest.mark.slow`` test at the bottom and self-skips here.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from satasr.core.audio import AudioBuffer
from satasr.core.models import MixedClip, PlacedUtterance, SpeakerCounts
from satasr.data.real.kaldi_format import load_corpus
from satasr.format import serialize_clip
from satasr.model.interfaces import Model
from satasr.pipeline.dataset_writer import SplitRatios
from satasr.training.interfaces import Checkpointer, TrainStep
from satasr.training.phase2 import Phase2Config, _split_real_clips, run_phase2

_TEXT = "hello world"
_SPEAKER = "speaker-a"

# Computed via satasr.pipeline.dataset_writer.assign_split with default
# SplitRatios: these ids are a fixed, deterministic sample of the hash space,
# two landing in "train" and two in "test" — picked so the fixture below
# exercises both buckets without relying on chance.
_TRAIN_IDS = ["real-0", "real-1"]
_HELD_OUT_IDS = ["real-24", "real-51"]

_SYNTHETIC_DURATION_S = 2.0


def _reference() -> tuple[PlacedUtterance, ...]:
    return (PlacedUtterance(_SPEAKER, 0.0, 1.0, _TEXT),)


_WRONG = (PlacedUtterance(_SPEAKER, 0.0, 1.0, "nope nope"),)


def _write_silent_wav(path: Path, duration_s: float, sample_rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * round(duration_s * sample_rate))


def _build_real_corpus(root: Path, recording_ids: list[str]) -> Path:
    """A tiny Kaldi-style corpus dir, one one-second recording per id."""
    corpus_dir = root / "real_corpus"
    corpus_dir.mkdir()
    scp, segments, text, utt2spk = [], [], [], []
    for rid in recording_ids:
        _write_silent_wav(corpus_dir / f"{rid}.wav", duration_s=1.0)
        scp.append(f"{rid} {rid}.wav")
        segments.append(f"{rid}-utt1 {rid} 0.0 1.0")
        text.append(f"{rid}-utt1 {_TEXT}")
        utt2spk.append(f"{rid}-utt1 {_SPEAKER}")
    (corpus_dir / "wav.scp").write_text("\n".join(scp) + "\n")
    (corpus_dir / "segments").write_text("\n".join(segments) + "\n")
    (corpus_dir / "text").write_text("\n".join(text) + "\n")
    (corpus_dir / "utt2spk").write_text("\n".join(utt2spk) + "\n")
    return corpus_dir


def _real_clips(tmp_path: Path) -> dict[str, MixedClip]:
    corpus_dir = _build_real_corpus(tmp_path, _TRAIN_IDS + _HELD_OUT_IDS)
    clips = load_corpus(corpus_dir, source="real_fixture")
    return {clip.metadata["recording_id"]: clip for clip in clips}


def _synthetic_clip() -> MixedClip:
    samples = np.zeros(round(_SYNTHETIC_DURATION_S * 16_000), dtype=np.float32)
    utterances = (PlacedUtterance(_SPEAKER, 0.0, _SYNTHETIC_DURATION_S, _TEXT),)
    return MixedClip(
        audio=AudioBuffer(samples, 16_000),
        utterances=utterances,
        counts=SpeakerCounts(total=1, max_simultaneous=1),
        metadata={"source": "synthetic"},
    )


class _FakeLoraState:
    """Shared state a fake ``TrainStep``/``Checkpointer``/``Model`` triple
    coordinate through, standing in for a real peft-wrapped model's frozen
    base and trainable adapter weights."""

    def __init__(self) -> None:
        self.base_weight = "frozen-base"  # must never be mutated by a step
        self.lora_weight = 0
        self.train_calls: list[str] = []


class _FakeTrainStep:
    """Fake ``TrainStep``: bumps only the adapter weight, never the base."""

    def __init__(self, state: _FakeLoraState) -> None:
        self._state = state

    def __call__(self, audio: AudioBuffer, transcript: str) -> float:
        assert self._state.base_weight == "frozen-base"
        self._state.lora_weight += 1
        self._state.train_calls.append(transcript)
        return 1.0


class _FakeCheckpointer:
    """Fake ``Checkpointer``: writes the adapter state to disk as JSON."""

    def __init__(self, state: _FakeLoraState) -> None:
        self._state = state

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        saved = {
            "lora_weight": self._state.lora_weight,
            "base_weight": self._state.base_weight,
        }
        path.write_text(json.dumps(saved))


class _FakeEvalModel:
    """Fake ``Model``: "learns" once ``lora_weight`` reaches ``threshold``,
    but always answers wrong on synthetic-length audio — so the two required
    evaluations (§5) are provably scored independently, not conflated."""

    def __init__(
        self,
        state: _FakeLoraState,
        threshold: int,
        correct: tuple[PlacedUtterance, ...],
        wrong: tuple[PlacedUtterance, ...],
    ) -> None:
        self._state = state
        self._threshold = threshold
        self._correct = correct
        self._wrong = wrong

    def predict(self, audio: AudioBuffer) -> tuple[PlacedUtterance, ...]:
        if audio.duration_s == _SYNTHETIC_DURATION_S:
            return self._wrong
        if self._state.lora_weight >= self._threshold:
            return self._correct
        return self._wrong


def _load_saved_adapter(path: Path) -> dict[str, object]:
    saved: dict[str, object] = json.loads(path.read_text())
    return saved


def test_fakes_satisfy_the_protocols_the_loop_depends_on() -> None:
    state = _FakeLoraState()
    model = _FakeEvalModel(state, threshold=0, correct=_reference(), wrong=_WRONG)

    assert isinstance(model, Model)
    assert isinstance(_FakeTrainStep(state), TrainStep)
    assert isinstance(_FakeCheckpointer(state), Checkpointer)


def test_split_real_clips_buckets_deterministically_and_disjointly(
    tmp_path: Path,
) -> None:
    real_clips = _real_clips(tmp_path)

    train, held_out = _split_real_clips(real_clips, SplitRatios())

    assert {c.metadata["recording_id"] for c in train} == set(_TRAIN_IDS)
    assert {c.metadata["recording_id"] for c in held_out} == set(_HELD_OUT_IDS)


def test_run_phase2_runs_the_lora_loop_and_freezes_base_weights(
    tmp_path: Path,
) -> None:
    real_clips = _real_clips(tmp_path)
    state = _FakeLoraState()
    model = _FakeEvalModel(
        state, threshold=len(_TRAIN_IDS), correct=_reference(), wrong=_WRONG
    )
    config = Phase2Config(adapter_dir=tmp_path / "adapter", epochs=1)

    result = run_phase2(
        model,
        _FakeTrainStep(state),
        _FakeCheckpointer(state),
        real_clips,
        [_synthetic_clip()],
        config,
    )

    # One optimizer step per real train clip; the base sentinel never moved.
    expected_transcripts = {serialize_clip(real_clips[i]) for i in _TRAIN_IDS}
    assert state.base_weight == "frozen-base"
    assert state.lora_weight == len(_TRAIN_IDS)
    assert set(state.train_calls) == expected_transcripts
    assert len(state.train_calls) == len(_TRAIN_IDS)
    assert result.num_train_clips == len(_TRAIN_IDS)
    assert result.num_held_out_clips == len(_HELD_OUT_IDS)
    # lora_weight reached threshold -> held-out real predictions are correct.
    assert result.held_out_real_metric == pytest.approx(0.0)
    # ...but the synthetic-length clip is deliberately always wrong (every
    # word missed -> cpwer 1.0), proving the two metrics aren't conflated.
    assert result.synthetic_regression_metric == pytest.approx(1.0)


def test_run_phase2_saves_an_adapter_that_reloads_with_the_trained_weight(
    tmp_path: Path,
) -> None:
    real_clips = _real_clips(tmp_path)
    state = _FakeLoraState()
    model = _FakeEvalModel(state, threshold=1, correct=_reference(), wrong=_WRONG)
    adapter_dir = tmp_path / "adapter"
    config = Phase2Config(adapter_dir=adapter_dir, epochs=1)

    run_phase2(
        model,
        _FakeTrainStep(state),
        _FakeCheckpointer(state),
        real_clips,
        [_synthetic_clip()],
        config,
    )

    # Reload from disk into a plain dict, independent of the live `state`
    # object, to prove the adapter genuinely persisted (not just an
    # in-memory reference the test happens to still hold).
    saved = _load_saved_adapter(adapter_dir)
    assert saved == {"lora_weight": len(_TRAIN_IDS), "base_weight": "frozen-base"}


def test_run_phase2_requires_a_nonempty_train_split(tmp_path: Path) -> None:
    real_clips = _real_clips(tmp_path)
    state = _FakeLoraState()
    model = _FakeEvalModel(state, threshold=0, correct=_reference(), wrong=_WRONG)
    config = Phase2Config(adapter_dir=tmp_path / "adapter")
    all_held_out = SplitRatios(train=0.0, val=0.0, test=1.0)

    with pytest.raises(ValueError, match="LoRA train split"):
        run_phase2(
            model,
            _FakeTrainStep(state),
            _FakeCheckpointer(state),
            real_clips,
            [_synthetic_clip()],
            config,
            split_ratios=all_held_out,
        )


def test_run_phase2_requires_a_nonempty_held_out_split(tmp_path: Path) -> None:
    real_clips = _real_clips(tmp_path)
    state = _FakeLoraState()
    model = _FakeEvalModel(state, threshold=0, correct=_reference(), wrong=_WRONG)
    config = Phase2Config(adapter_dir=tmp_path / "adapter")
    all_train = SplitRatios(train=1.0, val=0.0, test=0.0)

    with pytest.raises(ValueError, match="held-out real split"):
        run_phase2(
            model,
            _FakeTrainStep(state),
            _FakeCheckpointer(state),
            real_clips,
            [_synthetic_clip()],
            config,
            split_ratios=all_train,
        )


def test_run_phase2_requires_a_nonempty_synthetic_test_set(tmp_path: Path) -> None:
    real_clips = _real_clips(tmp_path)
    state = _FakeLoraState()
    model = _FakeEvalModel(state, threshold=0, correct=_reference(), wrong=_WRONG)
    config = Phase2Config(adapter_dir=tmp_path / "adapter")

    with pytest.raises(ValueError, match="empty clip set"):
        run_phase2(
            model,
            _FakeTrainStep(state),
            _FakeCheckpointer(state),
            real_clips,
            [],
            config,
        )


@pytest.mark.slow
def test_lora_train_step_and_load_lora_model_real_weights(tmp_path: Path) -> None:
    """Real LoRA wrap + optimizer step + save/reload: needs torch, peft, and
    transformers, all absent here, so this self-skips."""
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    from satasr.training.lora_backend import (
        LoraHyperparams,
        LoraTrainStep,
        load_lora_model,
    )

    train_step = LoraTrainStep("openai/whisper-tiny", LoraHyperparams())
    train_step(_synthetic_clip().audio, _TEXT)
    train_step.save(tmp_path / "adapter")

    eval_model = load_lora_model("openai/whisper-tiny", tmp_path / "adapter")
    assert isinstance(eval_model.predict(_synthetic_clip().audio), tuple)
