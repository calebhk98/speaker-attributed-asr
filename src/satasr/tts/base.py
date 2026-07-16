"""Shared scaffolding for every TTS engine — written once, reused by all.

A concrete engine subclasses :class:`BaseTTSEngine`, sets ``name`` /
``supports_cloning``, and implements the single :meth:`_render` method. Input
validation, the voice contract, and packaging the result into a
:class:`~satasr.core.models.SpeakerClip` live here so no engine repeats them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from satasr.core.audio import AudioBuffer
from satasr.core.interfaces import VoiceReference
from satasr.core.models import SpeakerClip


class BaseTTSEngine(ABC):
    """Template-method base: engines only supply ``_render``."""

    #: Unique registry name, e.g. "xtts_v2". Set by each subclass.
    name: str = ""
    #: True if the engine clones a voice from reference audio (design §4.4).
    supports_cloning: bool = False
    #: Hugging Face repo id(s) / model identifiers this engine needs at runtime.
    #: Declared here (not buried in ``_render``) so the model downloader can
    #: fetch weights ahead of time without importing the heavy library. Empty
    #: means "no pre-downloadable weights declared".
    model_ids: tuple[str, ...] = ()

    def synthesize(self, text: str, voice: VoiceReference) -> SpeakerClip:
        """Validate, render, and package one single-speaker clip."""
        if not text.strip():
            raise ValueError("cannot synthesize empty text")
        self._check_voice(voice)

        audio = self._render(text, voice)
        return SpeakerClip(audio=audio, text=text, speaker_id=voice.speaker_id)

    def _check_voice(self, voice: VoiceReference) -> None:
        """Enforce the voice contract implied by ``supports_cloning``."""
        if self.supports_cloning and voice.reference_audio is None:
            raise ValueError(
                f"{self.name} clones voices; set VoiceReference.reference_audio"
            )
        if not self.supports_cloning and voice.preset is None:
            raise ValueError(
                f"{self.name} uses preset voices; set VoiceReference.preset"
            )

    @abstractmethod
    def _render(self, text: str, voice: VoiceReference) -> AudioBuffer:
        """Produce raw audio for ``text`` in ``voice``. Implemented per engine."""
        raise NotImplementedError
