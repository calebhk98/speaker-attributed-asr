"""Speaker-attributed ASR model abstraction (design §2).

Re-exports the public API of :mod:`satasr.model.interfaces` and
:mod:`satasr.model.dicow`; import from here rather than reaching into the
submodules (mirrors :mod:`satasr.format`'s package layout).
"""

from satasr.model.dicow import DICOW_MODEL_ID, DiCoWModel
from satasr.model.interfaces import Model

__all__ = [
    "DICOW_MODEL_ID",
    "DiCoWModel",
    "Model",
]
