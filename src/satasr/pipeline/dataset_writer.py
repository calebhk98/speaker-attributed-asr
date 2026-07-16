"""Persist ``MixedClip``s as training examples: audio + manifest row (§4.1/§4.7).

**On-disk schema (single source of truth — document changes here, nowhere
else)**::

    <root>/<split>/audio/<clip_id>.wav   16-bit PCM mono WAV (§4.7's audio side)
    <root>/manifest.jsonl                 one JSON object per written clip:
        {
          "clip_id": str,
          "split": "train" | "val" | "test",
          "audio_path": str,             # relative to <root>
          "transcript": str,             # SOT text, see satasr.format.serialize_clip
          "duration_s": float,
          "speaker_count_total": int,             # SpeakerCounts.total
          "speaker_count_max_simultaneous": int,  # SpeakerCounts.max_simultaneous
          "metadata": dict[str, str],
        }

The transcript is produced by :func:`satasr.format.serialization.serialize_clip`
— the one and only SOT serializer (CLAUDE.md rule 4) — never re-implemented
here. ``SpeakerCounts`` (§4.6) is written as its two independent fields rather
than collapsed into one number, per its own docstring.

**Split discipline (§4.1):** train/val/test must be assigned deterministically
and must never leak a clip across splits. :func:`assign_split` hashes the
clip id into a stable fraction and buckets it against cumulative ratios, so
the same id always lands in the same split (across processes and re-runs)
and the buckets are disjoint by construction — there is no shared state to
drift out of sync.
"""

from __future__ import annotations

import hashlib
import json
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from satasr.core.audio import AudioBuffer
from satasr.core.models import MixedClip
from satasr.format.serialization import serialize_clip

_MANIFEST_NAME = "manifest.jsonl"
_AUDIO_SUBDIR = "audio"
_PCM16_ENCODE_SCALE = 32767.0
_PCM16_DECODE_SCALE = 32768.0
_PCM16_WIDTH_BYTES = 2
_HASH_HEX_DIGITS = 15  # 60 bits: ample precision for a uniform [0, 1) fraction
_RATIO_SUM_TOLERANCE = 1e-9


@dataclass(frozen=True)
class SplitRatios:
    """Target train/val/test proportions (§4.1); must sum to 1.0."""

    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > _RATIO_SUM_TOLERANCE:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")

    def boundaries(self) -> tuple[float, float]:
        """Cumulative (train_end, val_end) fractions; test is the remainder."""
        return self.train, self.train + self.val


_DEFAULT_SPLIT_RATIOS = SplitRatios()


@dataclass(frozen=True)
class ManifestRow:
    """One manifest line — the schema documented at module top."""

    clip_id: str
    split: str
    audio_path: str
    transcript: str
    duration_s: float
    speaker_count_total: int
    speaker_count_max_simultaneous: int
    metadata: dict[str, str]


def assign_split(clip_id: str, ratios: SplitRatios = _DEFAULT_SPLIT_RATIOS) -> str:
    """Deterministic, disjoint train/val/test assignment for ``clip_id`` (§4.1).

    A stable hash of the id (not a seeded RNG advanced by call order) is
    mapped into ``[0, 1)`` and bucketed against ``ratios``' cumulative
    boundaries, so the same id always resolves to the same split no matter
    the process, machine, or write order — and the three buckets partition
    the space, so no id can ever fall in two splits at once.
    """
    digest = hashlib.sha256(clip_id.encode("utf-8")).hexdigest()
    fraction = int(digest[:_HASH_HEX_DIGITS], 16) / float(16**_HASH_HEX_DIGITS)
    train_end, val_end = ratios.boundaries()
    if fraction < train_end:
        return "train"
    if fraction < val_end:
        return "val"
    return "test"


class DatasetWriter:
    """Writes ``MixedClip``s under ``root`` per the schema above (§4.1/§4.7)."""

    def __init__(
        self, root: Path | str, ratios: SplitRatios = _DEFAULT_SPLIT_RATIOS
    ) -> None:
        self._root = Path(root)
        self._ratios = ratios

    def write(self, clip_id: str, clip: MixedClip) -> ManifestRow:
        """Write one clip's audio + manifest row; returns the row written."""
        split = assign_split(clip_id, self._ratios)
        audio_rel = Path(split) / _AUDIO_SUBDIR / f"{clip_id}.wav"
        _write_wav(self._root / audio_rel, clip.audio)
        row = ManifestRow(
            clip_id=clip_id,
            split=split,
            audio_path=str(audio_rel),
            transcript=serialize_clip(clip),
            duration_s=clip.audio.duration_s,
            speaker_count_total=clip.counts.total,
            speaker_count_max_simultaneous=clip.counts.max_simultaneous,
            metadata=dict(clip.metadata),
        )
        self._append_manifest(row)
        return row

    def _append_manifest(self, row: ManifestRow) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(row))
        with (self._root / _MANIFEST_NAME).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def read_manifest(root: Path | str) -> tuple[ManifestRow, ...]:
    """Read every manifest row back, in the order it was written."""
    path = Path(root) / _MANIFEST_NAME
    if not path.exists():
        return ()
    lines = (line for line in path.read_text(encoding="utf-8").splitlines() if line)
    return tuple(ManifestRow(**json.loads(line)) for line in lines)


def read_audio(root: Path | str, row: ManifestRow) -> AudioBuffer:
    """Re-read the audio a manifest row points at, undoing ``_write_wav``."""
    with wave.open(str(Path(root) / row.audio_path), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
        sample_rate = handle.getframerate()
    pcm16 = np.frombuffer(raw, dtype=np.int16)
    samples = (pcm16.astype(np.float32) / _PCM16_DECODE_SCALE).astype(np.float32)
    return AudioBuffer(samples, sample_rate)


def _write_wav(path: Path, audio: AudioBuffer) -> None:
    """Write ``audio`` as 16-bit PCM mono (mirrors ``augment/reverb.py``'s WAV IO)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(audio.samples, -1.0, 1.0) * _PCM16_ENCODE_SCALE).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(_PCM16_WIDTH_BYTES)
        handle.setframerate(audio.sample_rate)
        handle.writeframes(pcm16.tobytes())
