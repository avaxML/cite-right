from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...
