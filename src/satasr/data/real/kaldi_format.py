"""Shared Kaldi-style directory reader for LibriCSS and AMI (design §4.8).

Both corpora are commonly distributed, or re-exported by standard recipes, as
Kaldi data directories: ``wav.scp`` maps a recording id to its audio file,
``segments`` gives each utterance's start/end time within that recording,
``text`` gives the transcript per utterance id, and ``utt2spk`` gives the
speaker id per utterance id. Parsing this layout once here means the two
per-corpus loaders are a handful of lines each instead of two divergent
parsers (rule 4, no repetition).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from satasr.core.models import MixedClip, PlacedUtterance
from satasr.data.counts import speaker_counts
from satasr.data.wav_io import load_wav_mono


@dataclass(frozen=True)
class _Segment:
    utt_id: str
    recording_id: str
    start_s: float
    end_s: float


def load_corpus(corpus_dir: Path, *, source: str) -> tuple[MixedClip, ...]:
    """Load every recording in a Kaldi-style corpus dir into MixedClips."""
    wav_paths = _read_wav_scp(corpus_dir)
    texts = _read_kv_file(corpus_dir / "text")
    speakers = _read_kv_file(corpus_dir / "utt2spk")
    by_recording = _group_by_recording(_read_segments(corpus_dir))
    return tuple(
        _build_clip(
            recording_id, segments, wav_paths[recording_id], texts, speakers, source
        )
        for recording_id, segments in by_recording.items()
    )


def _build_clip(
    recording_id: str,
    segments: list[_Segment],
    wav_path: Path,
    texts: dict[str, str],
    speakers: dict[str, str],
    source: str,
) -> MixedClip:
    audio = load_wav_mono(wav_path)
    utterances = tuple(
        PlacedUtterance(
            speaker_id=speakers[segment.utt_id],
            start_s=segment.start_s,
            end_s=segment.end_s,
            text=texts.get(segment.utt_id, ""),
        )
        for segment in segments
    )
    metadata = {"source": source, "recording_id": recording_id}
    return MixedClip(audio, utterances, speaker_counts(utterances), metadata)


def _group_by_recording(segments: list[_Segment]) -> dict[str, list[_Segment]]:
    by_recording: dict[str, list[_Segment]] = {}
    for segment in segments:
        by_recording.setdefault(segment.recording_id, []).append(segment)
    return by_recording


def _read_wav_scp(corpus_dir: Path) -> dict[str, Path]:
    pairs = _split_lines(corpus_dir / "wav.scp")
    return {recording_id: corpus_dir / rel_path for recording_id, rel_path in pairs}


def _read_segments(corpus_dir: Path) -> list[_Segment]:
    lines = (corpus_dir / "segments").read_text().splitlines()
    return [_parse_segment(line) for line in lines if line.strip()]


def _parse_segment(line: str) -> _Segment:
    utt_id, recording_id, start_s, end_s = line.split()
    return _Segment(utt_id, recording_id, float(start_s), float(end_s))


def _read_kv_file(path: Path) -> dict[str, str]:
    return dict(_split_lines(path))


def _split_lines(path: Path) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        key, value = line.split(maxsplit=1)
        result.append((key, value))
    return result
