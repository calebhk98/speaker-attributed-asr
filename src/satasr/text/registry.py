"""The single registry every text source plugs into.

Mirrors ``satasr.tts.registry`` / ``satasr.alignment.registry``: text sources
register themselves by name here (see ``static.py``, ``plain_file.py``,
``wikipedia.py``), so the generation pipeline can pick a source of sentences
purely by name without importing any concrete implementation (design §4.5).
"""

from __future__ import annotations

from satasr.core.interfaces import TextSource
from satasr.core.registry import Registry

TEXT_SOURCES: Registry[TextSource] = Registry("text source")
