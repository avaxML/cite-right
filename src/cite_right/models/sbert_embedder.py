"""SentenceTransformer embedder for the citation alignment pipeline."""

from __future__ import annotations

from typing import Sequence


class SentenceTransformerEmbedder:
    """SentenceTransformer embedder for the citation alignment pipeline."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the SentenceTransformerEmbedder.

        Args:
            model_name (str): The name of the SentenceTransformer model to use.

        Raises:
            RuntimeError: If sentence-transformers is not installed.
        """
        try:
            from sentence_transformers import (  # pyright: ignore[reportMissingImports]
                SentenceTransformer,
            )
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install with 'cite-right[embeddings]'."
            ) from exc

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a list of text strings into a list of float vectors.

        Args:
            texts (Sequence[str]): The text strings to encode.

        Returns:
            list[list[float]]: List of float vectors for each input text.
        """
        embeddings = self._model.encode(list(texts))
        return embeddings.tolist()
