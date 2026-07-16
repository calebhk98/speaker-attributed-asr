"""LibriSpeechMix loader (design §4.8).

LibriSpeechMix does not ship pre-mixed audio: a JSON-lines manifest lists, per
mixture, the clean single-speaker LibriSpeech source utterances and the delay
each is started at, and the mixture is built by summing them on that shared
timeline. That overlap is artificially constructed — unlike LibriCSS/AMI's
genuine acoustic overlap (§4.8) — but the underlying acoustics/conditions are
still real, recorded LibriVox speech. This loader performs the construction so
callers get back a single MixedClip either way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from satasr.core.audio import AudioBuffer
from satasr.core.models import MixedClip, PlacedUtterance
from satasr.data.counts import speaker_counts
from satasr.data.wav_io import load_wav_mono


@dataclass(frozen=True)
class _SourceUtterance:
    speaker_id: str
    wav_path: Path
    text: str
    start_s: float


def load_librispeech_mix_manifest(manifest_path: Path) -> tuple[MixedClip, ...]:
    """Load every mixture described in a LibriSpeechMix JSON-lines manifest."""
    base_dir = manifest_path.parent
    lines = manifest_path.read_text().splitlines()
    return tuple(
        _build_clip(_parse_mixture(json.loads(line), base_dir))
        for line in lines
        if line.strip()
    )


def _parse_mixture(record: dict[str, Any], base_dir: Path) -> list[_SourceUtterance]:
    return [
        _SourceUtterance(
            speaker_id=item["speaker_id"],
            wav_path=base_dir / item["wav"],
            text=item["text"],
            start_s=float(item["start_s"]),
        )
        for item in record["utterances"]
    ]


def _build_clip(sources: list[_SourceUtterance]) -> MixedClip:
    clean = [(src, load_wav_mono(src.wav_path)) for src in sources]
    audio = _render(clean)
    utterances = tuple(
        PlacedUtterance(
            src.speaker_id, src.start_s, src.start_s + buf.duration_s, src.text
        )
        for src, buf in clean
    )
    return MixedClip(
        audio, utterances, speaker_counts(utterances), {"source": "librispeech_mix"}
    )


def _render(clean: list[tuple[_SourceUtterance, AudioBuffer]]) -> AudioBuffer:
    sample_rate = clean[0][1].sample_rate
    total = max(
        round(src.start_s * sample_rate) + buf.num_samples for src, buf in clean
    )
    mix = np.zeros(total, dtype=np.float32)
    for src, buf in clean:
        start = round(src.start_s * sample_rate)
        mix[start : start + buf.num_samples] += buf.samples
    np.clip(mix, -1.0, 1.0, out=mix)
    return AudioBuffer(mix, sample_rate)
