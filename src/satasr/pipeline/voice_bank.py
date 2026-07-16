"""Voice bank: hands out ``VoiceReference``s for real recorded speakers (§4.4).

Cloning engines need a real reference clip (LibriVox / Common Voice, CC0 —
§4.8) rather than a synthetic voice. :class:`VoiceBank` indexes clips per
speaker under a configured directory (one subdirectory per speaker id) and
hands out ``VoiceReference(speaker_id, reference_audio=...)`` for engines
with ``supports_cloning=True``, or ``VoiceReference(speaker_id, preset=...)``
for the rest (e.g. Kokoro's fixed bank) — one bank drives every engine
instead of each sampling its own small built-in set.

Nothing touches disk until a reference is requested: the speaker index is
built lazily on first access and cached; a clip's audio loads only when
sampled (mirrors ``augment/reverb.py``'s lazy corpus sampling).
"""

from __future__ import annotations

import random
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.interfaces import VoiceReference

_CLIP_GLOB = "*.wav"
_PCM16_FULL_SCALE = 32768.0


@dataclass(frozen=True)
class VoiceBankConfig:
    """Where reference clips live (one subdirectory per speaker id), plus the
    fallback preset bank for non-cloning engines (§4.4/§4.8).
    """

    source_dir: str
    preset_names: tuple[str, ...] = ()
    seed: int = 0


def _load_wav(path: Path, sample_rate: int) -> NDArray[np.float32]:
    """Decode a 16-bit PCM WAV file to float32 samples at ``sample_rate``.

    Stdlib ``wave`` (mirrors ``augment/reverb.py``'s RIR loader) — no extra
    dependency beyond numpy.
    """
    with wave.open(str(path), "rb") as wav_file:
        native_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        raw = wav_file.readframes(wav_file.getnframes())
    pcm16 = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        pcm16 = pcm16.reshape(-1, channels).mean(axis=1)
    samples = (pcm16.astype(np.float32) / _PCM16_FULL_SCALE).astype(np.float32)
    return _resample(samples, native_rate, sample_rate)


def _resample(
    samples: NDArray[np.float32], src_rate: int, dst_rate: int
) -> NDArray[np.float32]:
    """Plain-numpy linear resample (no scipy), mirroring the TTS engines' helper."""
    if samples.size == 0 or src_rate == dst_rate:
        return samples
    duration_s = samples.shape[0] / src_rate
    dst_count = max(1, round(duration_s * dst_rate))
    src_times = np.arange(samples.shape[0], dtype=np.float64) / src_rate
    dst_times = np.arange(dst_count, dtype=np.float64) / dst_rate
    resampled: NDArray[np.float32] = np.interp(dst_times, src_times, samples).astype(
        np.float32
    )
    return resampled


class VoiceBank:
    """Indexes real reference clips per speaker and hands out ``VoiceReference``s.

    Cloning engines get a real clip sampled deterministically from that
    speaker's subdirectory (§4.4); others get a preset drawn the same way
    from ``config.preset_names``. Both are seeded from ``config.seed`` +
    speaker id (not one shared RNG advanced by call order), so a fresh bank
    built from the same config reproduces a prior run's assignments exactly.
    """

    def __init__(self, config: VoiceBankConfig) -> None:
        self._config = config
        self._clips_by_speaker: dict[str, tuple[Path, ...]] | None = None

    def speakers(self) -> tuple[str, ...]:
        """Sorted speaker ids discovered under ``config.source_dir`` (lazy)."""
        return tuple(sorted(self._index()))

    def reference(self, speaker_id: str, *, supports_cloning: bool) -> VoiceReference:
        """The ``VoiceReference`` to hand a TTS engine for ``speaker_id``.

        Pass the target engine's own ``supports_cloning`` flag (see
        ``core/interfaces.TTSEngine``) — the single place this decision is
        made, so no engine needs its own voice-sampling logic.
        """
        if supports_cloning:
            clip_path = self._sample_clip(speaker_id)
            return VoiceReference(
                speaker_id=speaker_id, reference_audio=self._load(clip_path)
            )
        return VoiceReference(
            speaker_id=speaker_id, preset=self._sample_preset(speaker_id)
        )

    def _index(self) -> dict[str, tuple[Path, ...]]:
        """Build (once) the speaker -> clip-paths map from ``source_dir``."""
        if self._clips_by_speaker is not None:
            return self._clips_by_speaker
        root = Path(self._config.source_dir)
        index: dict[str, tuple[Path, ...]] = {}
        for speaker_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            clips = tuple(sorted(speaker_dir.glob(_CLIP_GLOB)))
            if clips:
                index[speaker_dir.name] = clips
        self._clips_by_speaker = index
        return index

    def _sample_clip(self, speaker_id: str) -> Path:
        clips = self._index().get(speaker_id)
        if not clips:
            raise KeyError(f"no reference clips indexed for speaker {speaker_id!r}")
        return self._rng_for(speaker_id).choice(clips)

    def _sample_preset(self, speaker_id: str) -> str:
        if not self._config.preset_names:
            raise ValueError(
                "VoiceBankConfig.preset_names is empty; cannot assign a preset"
            )
        return self._rng_for(speaker_id).choice(self._config.preset_names)

    def _rng_for(self, speaker_id: str) -> random.Random:
        """Deterministic per-speaker RNG, seeded from ``config.seed`` + id."""
        return random.Random(f"{self._config.seed}:{speaker_id}")

    def _load(self, path: Path) -> AudioBuffer:
        return AudioBuffer(_load_wav(path, SAMPLE_RATE), SAMPLE_RATE)
