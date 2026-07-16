"""Auto-discovers and registers every engine module in this package.

Dropping a new ``<engine>.py`` file into this directory is all that is needed —
it is imported automatically here, so 20+ engines can be added as independent,
non-conflicting files with **no shared edit** (this is what lets them be built
in parallel). An engine whose optional heavy dependency is missing raises on
import and is skipped; its own test asserts registration, so a real breakage is
caught there, not hidden.
"""

from __future__ import annotations

import contextlib
import importlib
import pkgutil

for _module in pkgutil.iter_modules(__path__):
    if _module.name.startswith("_"):
        continue
    # A broken/uninstalled engine must not take down discovery of the rest.
    with contextlib.suppress(Exception):
        importlib.import_module(f"{__name__}.{_module.name}")
