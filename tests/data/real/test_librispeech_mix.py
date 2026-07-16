"""Fast tests for the LibriSpeechMix loader (design §4.8)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from satasr.data.real.librispeech_mix import load_librispeech_mix_manifest


def test_builds_one_mixed_clip_from_delayed_clean_utterances(
    tmp_path: Path, write_wav: Callable[..., None]
) -> None:
    write_wav(tmp_path / "a.wav", duration_s=2.0, tone_hz=200.0)
    write_wav(tmp_path / "b.wav", duration_s=2.0, tone_hz=400.0)
    manifest = tmp_path / "mix.jsonl"
    record = {
        "utterances": [
            {
                "speaker_id": "spk_a",
                "wav": "a.wav",
                "text": "hello there",
                "start_s": 0.0,
            },
            {
                "speaker_id": "spk_b",
                "wav": "b.wav",
                "text": "hi friend",
                "start_s": 1.0,
            },
        ]
    }
    manifest.write_text(json.dumps(record) + "\n")

    clips = load_librispeech_mix_manifest(manifest)

    assert len(clips) == 1
    clip = clips[0]
    assert clip.metadata["source"] == "librispeech_mix"
    assert clip.counts.total == 2
    assert clip.counts.max_simultaneous == 2  # 1.0-2.0s overlap
    assert abs(clip.audio.duration_s - 3.0) < 0.01  # b ends at 1.0 + 2.0
    speakers = {u.speaker_id for u in clip.utterances}
    assert speakers == {"spk_a", "spk_b"}
