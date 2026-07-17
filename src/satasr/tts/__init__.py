"""TTS package: the engine registry plus all registered engines.

Importing ``satasr.tts`` guarantees the built-in engines have registered, so
``TTS_ENGINES.available()`` is populated.
"""

from satasr.tts import engines  # noqa: F401  (triggers engine registration)
from satasr.tts.base import BaseTTSEngine
from satasr.tts.registry import TTS_ENGINES

__all__ = ["BaseTTSEngine", "TTS_ENGINES"]
