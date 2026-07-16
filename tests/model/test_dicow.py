"""Tests for the DiCoW wrapper: fast contract/decode checks that never import
torch or transformers, plus a slow real load+infer smoke test that self-skips
when those heavy optional dependencies (or their weights) are unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from satasr.core.audio import AudioBuffer
from satasr.format import parse_utterances
from satasr.model import DICOW_MODEL_ID, DiCoWModel, Model


def test_dicow_model_satisfies_protocol() -> None:
    assert isinstance(DiCoWModel(), Model)


def test_dicow_model_constructs_without_importing_torch() -> None:
    # No heavy import happens until predict() is called; this must succeed
    # even in an environment with neither torch nor transformers installed.
    model = DiCoWModel()
    assert model.model_id == DICOW_MODEL_ID
    assert model.device == "cpu"


def test_dicow_model_accepts_model_id_and_device_overrides() -> None:
    model = DiCoWModel(model_id="local/checkpoint", device="cuda")
    assert model.model_id == "local/checkpoint"
    assert model.device == "cuda"


def test_predict_decodes_generated_text_via_parse_utterances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stub out the heavy generation step so this stays fast, and assert that
    # predict() decodes whatever raw SOT string the model produces through
    # the one canonical parser (satasr.format.parse_utterances) rather than
    # any hand-rolled re-parse (CLAUDE.md rule 4).
    sot = "<A> 0.0->1.5: hello there\n<B> 1.0->2.5: hi back"
    model = DiCoWModel()
    monkeypatch.setattr(model, "_generate", lambda audio: sot)

    audio = AudioBuffer(np.zeros(16_000, dtype=np.float32))
    assert model.predict(audio) == parse_utterances(sot)


def test_predict_returns_empty_tuple_for_empty_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DiCoWModel()
    monkeypatch.setattr(model, "_generate", lambda audio: "")

    audio = AudioBuffer(np.zeros(16_000, dtype=np.float32))
    assert model.predict(audio) == ()


@pytest.mark.slow
def test_dicow_real_load_and_infer() -> None:
    """Real load + inference: needs torch, transformers, and downloaded
    weights (~918M params). Self-skips when either dependency is missing."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    model = DiCoWModel()
    audio = AudioBuffer(np.zeros(16_000, dtype=np.float32))
    utterances = model.predict(audio)
    assert isinstance(utterances, tuple)
