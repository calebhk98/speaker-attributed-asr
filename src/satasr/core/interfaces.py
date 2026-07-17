"""Abstract boundaries between pipeline stages (dependency inversion).

High-level orchestration code depends only on these Protocols, never on a
concrete engine. A new TTS engine, aligner, or augmenter is added by
*implementing* one of these — no existing code changes. This is what lets the
20 TTS engines be built in parallel against a stable contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from satasr.core.audio import AudioBuffer
from satasr.core.models import MixedClip, SpeakerClip, SpeakerCounts, Word


@dataclass(frozen=True)
class VoiceReference:
    """A voice to speak in. ``reference_audio`` is set for cloning engines
    (design §4.4: clone real recorded speakers); ``preset`` names a built-in
    voice for engines without cloning (e.g. Kokoro).
    """

    speaker_id: str
    reference_audio: AudioBuffer | None = None
    preset: str | None = None


@runtime_checkable
class TextSource(Protocol):
    """Yields sentences to be spoken (Wikipedia, LLM text, ...). See §4.5."""

    def sentences(self) -> Iterator[str]: ...


@runtime_checkable
class TTSEngine(Protocol):
    """Turns text + a voice into a single-speaker clip.

    Implementations register themselves in ``satasr.tts.registry`` so they can
    be selected by name and weighted for the 10k-hour generation budget (§4.2).
    """

    name: str
    supports_cloning: bool

    def synthesize(self, text: str, voice: VoiceReference) -> SpeakerClip: ...


@runtime_checkable
class Aligner(Protocol):
    """Produces word-level timestamps for clean, uncut audio (§4.7)."""

    def align(self, audio: AudioBuffer, text: str) -> tuple[Word, ...]: ...


@runtime_checkable
class Augmenter(Protocol):
    """Applies an acoustic transform (noise, reverb) to audio (§4.9)."""

    def apply(self, audio: AudioBuffer) -> AudioBuffer: ...


@runtime_checkable
class Mixer(Protocol):
    """Splices single-speaker clips into one overlapping multi-speaker clip."""

    def mix(self, clips: Iterable[SpeakerClip], counts: SpeakerCounts) -> MixedClip: ...
