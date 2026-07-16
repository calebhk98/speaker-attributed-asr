"""The single registry every TTS engine plugs into.

Engines register themselves by name here (see ``engines/``). The generation
pipeline selects and weights engines purely by name, so adding an engine never
touches orchestration code — this is what makes the 20 engines independent work
items (design §4.2/§4.3).
"""

from __future__ import annotations

from satasr.core.interfaces import TTSEngine
from satasr.core.registry import Registry

TTS_ENGINES: Registry[TTSEngine] = Registry("tts engine")
