"""Core: the stable centre of the project.

Everything here is back-end agnostic — value objects, Protocols, the plugin
registry and configuration. Other packages depend *inward* on core; core
depends on nothing but numpy. Re-exported here for convenient importing.
"""

from satasr.core.audio import SAMPLE_RATE, AudioBuffer
from satasr.core.config import (
    Config,
    GenerationConfig,
    MixingConfig,
    default_config,
)
from satasr.core.interfaces import (
    Aligner,
    Augmenter,
    Mixer,
    TextSource,
    TTSEngine,
    VoiceReference,
)
from satasr.core.models import (
    MixedClip,
    PlacedUtterance,
    SpeakerClip,
    SpeakerCounts,
    Word,
)
from satasr.core.registry import Registry

__all__ = [
    "SAMPLE_RATE",
    "AudioBuffer",
    "Config",
    "GenerationConfig",
    "MixingConfig",
    "default_config",
    "Aligner",
    "Augmenter",
    "Mixer",
    "TextSource",
    "TTSEngine",
    "VoiceReference",
    "MixedClip",
    "PlacedUtterance",
    "SpeakerClip",
    "SpeakerCounts",
    "Word",
    "Registry",
]
