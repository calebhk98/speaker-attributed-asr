"""Importing this package registers every built-in engine as a side effect.

Each new engine adds one import line here and nowhere else — the single place
that knows the full engine roster. Real engines (design §4.3) are added as they
are implemented against their GitHub issues.
"""

from satasr.tts.engines import sine  # noqa: F401  (import for registration)

__all__ = ["sine"]
