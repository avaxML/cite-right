"""Interfaces for the embedding models used in the citation alignment pipeline."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Interface for embedding text strings.

    Methods:
        encode(texts): Encodes a list of text strings into a list of float vectors.

    Example:
        >>> embedder: Embedder
        >>> result = embedder.encode(["Hello, world!", "Goodbye, world!"])
    """

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a list of text strings into a list of float vectors.

        Args:
            texts (Sequence[str]): The text strings to encode.

        Returns:
            list[list[float]]: List of float vectors for each input text.
        """
        ...
