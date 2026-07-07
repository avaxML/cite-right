"""SentenceTransformer embedder for the citation alignment pipeline."""

from __future__ import annotations

from typing import Sequence

_EMBEDDING_CACHE: dict[tuple[str, str], list[float]] = {}


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
        self.model_name = model_name

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a list of text strings into a list of float vectors.

        Args:
            texts (Sequence[str]): The text strings to encode.

        Returns:
            list[list[float]]: List of float vectors for each input text.
        """
        results: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for i, text in enumerate(texts):
            key = (text, self.model_name)
            if key in _EMBEDDING_CACHE:
                results[i] = _EMBEDDING_CACHE[key]
            else:
                missing_indices.append(i)
                missing_texts.append(text)

        if missing_texts:
            encoded = self._model.encode(missing_texts)
            encoded_list = encoded.tolist()
            for i, idx in enumerate(missing_indices):
                vector = encoded_list[i]
                _EMBEDDING_CACHE[(missing_texts[i], self.model_name)] = vector
                results[idx] = vector

        return results  # type: ignore
