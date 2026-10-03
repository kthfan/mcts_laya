"""Name -> factory registries, so configs can select components without code changes.

New components register themselves with a decorator:

    @ENVIRONMENTS.register("countdown")
    class CountdownEnv(...): ...

and are built from config with `ENVIRONMENTS.build("countdown", **params)`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def deco(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._items:
                raise KeyError(f"{self.kind} {name!r} is already registered")
            self._items[name] = factory
            return factory

        return deco

    def build(self, name: str, *args: Any, **kwargs: Any) -> T:
        _import_builtins()
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} {name!r}; known: {sorted(self._items)}")
        return self._items[name](*args, **kwargs)

    def names(self):
        _import_builtins()
        return sorted(self._items)


ENVIRONMENTS: Registry = Registry("environment")
EVALUATORS: Registry = Registry("evaluator")
SEARCHERS: Registry = Registry("searcher")
TEACHERS: Registry = Registry("teacher")

_BUILTINS_IMPORTED = False


def _import_builtins() -> None:
    """Import the modules whose decorators fill the registries (once, lazily)."""
    global _BUILTINS_IMPORTED
    if _BUILTINS_IMPORTED:
        return
    _BUILTINS_IMPORTED = True
    from . import envs, evaluators, search, teachers  # noqa: F401
