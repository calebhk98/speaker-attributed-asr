"""A single, generic plugin registry — reused for every extension point.

TTS engines, aligners, augmenters and text sources are all discovered the same
way through instances of :class:`Registry`. Writing this once (instead of a
bespoke lookup per stage) is the "single source of truth, no repetition" rule
applied to plugin wiring.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps a string name to a factory that builds a plugin of type ``T``."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator that records a factory under ``name``."""

        def decorate(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._factories:
                raise ValueError(f"{self._kind} {name!r} is already registered")
            self._factories[name] = factory
            return factory

        return decorate

    def create(self, name: str, /, **kwargs: object) -> T:
        """Instantiate the plugin registered under ``name``.

        ``name`` is positional-only so factories are free to declare their own
        ``name`` keyword argument without colliding with this lookup key.
        """
        factory = self._factories.get(name)
        if factory is None:
            known = self.available()
            raise KeyError(f"unknown {self._kind} {name!r}; available: {known}")
        return factory(**kwargs)

    def available(self) -> list[str]:
        """Sorted list of registered names — a stable single source of truth."""
        return sorted(self._factories)
