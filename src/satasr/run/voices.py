"""Assign a ``VoiceReference`` to each speaker in a generated clip (design §4.4).

Two strategies behind one Protocol (dependency inversion), chosen by config:

- :class:`PresetVoiceProvider` — synthetic per-speaker presets. The zero-config
  default; works only for preset engines (cloning engines raise a clear error).
- :class:`BankVoiceProvider` — draws distinct real speakers from a
  :class:`~satasr.pipeline.voice_bank.VoiceBank`, so cloning engines get real
  reference audio and thus work in a run.

``capacity`` lets the builder cap a clip's total-speaker count to the number of
voices that actually exist (a bank of N speakers cannot fill a 20-speaker clip).
"""

from __future__ import annotations

import random
from typing import Protocol, runtime_checkable

from satasr.core.interfaces import VoiceReference
from satasr.pipeline.voice_bank import VoiceBank, VoiceBankConfig
from satasr.run.config import DatasetStageConfig


@runtime_checkable
class VoiceProvider(Protocol):
    """Supplies distinct speaker ids and a voice for each (per clip)."""

    def capacity(self) -> int | None:
        """Max distinct speakers available, or ``None`` if unbounded."""

    def speaker_ids(self, count: int, rng: random.Random) -> list[str]:
        """Pick ``count`` distinct speaker ids for one clip."""

    def reference(self, speaker_id: str, *, supports_cloning: bool) -> VoiceReference:
        """The voice to hand an engine, per its ``supports_cloning`` flag."""


class PresetVoiceProvider:
    """Synthetic presets (``S1``, ``S2``, ...); cloning engines need a bank."""

    def capacity(self) -> int | None:
        return None

    def speaker_ids(self, count: int, rng: random.Random) -> list[str]:
        return [f"S{index + 1}" for index in range(count)]

    def reference(self, speaker_id: str, *, supports_cloning: bool) -> VoiceReference:
        if supports_cloning:
            raise ValueError(
                "cloning engines need reference audio; set [dataset].voice_source_dir "
                "to a VoiceBank directory (design §4.4) to use them in a run"
            )
        return VoiceReference(speaker_id=speaker_id, preset=speaker_id)


class BankVoiceProvider:
    """Draws distinct real speakers from a :class:`VoiceBank` (§4.4)."""

    def __init__(self, bank: VoiceBank) -> None:
        self._bank = bank

    def capacity(self) -> int | None:
        return len(self._bank.speakers())

    def speaker_ids(self, count: int, rng: random.Random) -> list[str]:
        available = list(self._bank.speakers())
        if len(available) < count:
            raise ValueError(
                f"voice bank has {len(available)} speakers but a clip needs {count} "
                "distinct — add more reference speakers or lower the speaker count"
            )
        return rng.sample(available, count)

    def reference(self, speaker_id: str, *, supports_cloning: bool) -> VoiceReference:
        return self._bank.reference(speaker_id, supports_cloning=supports_cloning)


def build_voice_provider(dataset: DatasetStageConfig, seed: int) -> VoiceProvider:
    """Pick the provider a run's config asks for (bank if a source dir is set)."""
    if dataset.voice_source_dir is None:
        return PresetVoiceProvider()
    config = VoiceBankConfig(
        source_dir=dataset.voice_source_dir,
        preset_names=dataset.voice_presets,
        seed=seed,
    )
    return BankVoiceProvider(VoiceBank(config))
