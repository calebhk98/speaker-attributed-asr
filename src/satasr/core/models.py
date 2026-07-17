"""Domain data model — the single source of truth for what flows through the
pipeline.

Nothing here imports back-end libraries (torch, TTS engines, ...). These are
plain, immutable value objects so that every stage speaks the same language:

    TextSource -> TTSEngine -> SpeakerClip -> Aligner -> (words) -> Mixer
        -> MixedClip (the training example: audio + serialized transcript)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from satasr.core.audio import AudioBuffer


@dataclass(frozen=True)
class Word:
    """One aligned word with start/end times in seconds, relative to its clip."""

    text: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class SpeakerClip:
    """A single sentence spoken by a single speaker, as produced by a TTSEngine.

    ``words`` is empty until an :class:`~satasr.core.interfaces.Aligner` fills
    in word-level timestamps on the clean, uncut audio (see design §4.7).
    """

    audio: AudioBuffer
    text: str
    speaker_id: str
    words: tuple[Word, ...] = ()

    @property
    def is_aligned(self) -> bool:
        return len(self.words) > 0


@dataclass(frozen=True)
class PlacedUtterance:
    """A speaker's utterance positioned on the mixed timeline.

    Times are absolute (seconds from the start of the mixed clip). ``truncated``
    marks an utterance the mixer cut short to simulate an interruption.
    """

    speaker_id: str
    start_s: float
    end_s: float
    text: str
    words: tuple[Word, ...] = ()
    truncated: bool = False


@dataclass(frozen=True)
class SpeakerCounts:
    """The two independent speaker counts the design insists on tracking (§4.6).

    ``total`` = distinct identities present in the clip.
    ``max_simultaneous`` = most speakers ever overlapping at one instant.
    They are not the same number and must be recorded separately.
    """

    total: int
    max_simultaneous: int


@dataclass(frozen=True)
class MixedClip:
    """A finished training example: mixed audio plus its serialized transcript."""

    audio: AudioBuffer
    utterances: tuple[PlacedUtterance, ...]
    counts: SpeakerCounts
    metadata: dict[str, str] = field(default_factory=dict)

    def with_audio(self, audio: AudioBuffer) -> MixedClip:
        """Return a copy after an augmenter reshapes the audio; labels unchanged."""
        return replace(self, audio=audio)
