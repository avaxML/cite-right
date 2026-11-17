"""Index for fast similarity search over embedding vectors."""

from __future__ import annotations

from math import isfinite
from typing import Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from cite_right.models.base import Embedder


class EmbeddingIndex(BaseModel):
    """An index for fast similarity search over embedding vectors.

    Attributes:
        vectors (npt.NDArray[np.float32]): Matrix of embedding vectors, one per text.
        norms (npt.NDArray[np.float32]): Precomputed L2 norms for each embedding vector.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    vectors: npt.NDArray[np.float32]
    norms: npt.NDArray[np.float32]

    @classmethod
    def build(cls, embedder: Embedder, texts: Sequence[str]) -> "EmbeddingIndex":
        """Build an EmbeddingIndex from a set of texts using the given embedder.

        Args:
            embedder (Embedder): Embedder to encode the texts as vectors.
            texts (Sequence[str]): The texts to index.

        Returns:
            EmbeddingIndex: Index containing the embeddings and their norms.
        """
        vectors = _embedding_matrix(embedder.encode(texts), len(texts))
        norms = np.linalg.norm(vectors, axis=1).astype(np.float32)
        return cls(vectors=vectors, norms=norms)

    def _query(
        self, query_vector: list[float]
    ) -> tuple[npt.NDArray[np.float32], float] | None:
        try:
            query = np.asarray(query_vector, dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            return None
        if query.shape != (self.vectors.shape[1],):
            return None
        norm = float(np.linalg.norm(query))
        return (query, norm) if isfinite(norm) and norm > 0.0 else None

    def similarity(self, index: int, query_vector: list[float]) -> float | None:
        """Return a finite cosine for one cached passage and query."""
        if not 0 <= index < len(self.vectors):
            return None
        norm = float(self.norms[index])
        if not isfinite(norm) or norm <= 0.0:
            return None
        prepared = self._query(query_vector)
        if prepared is None:
            return None
        query, query_norm = prepared
        similarity = float(np.dot(self.vectors[index], query)) / (norm * query_norm)
        return max(-1.0, min(1.0, similarity)) if isfinite(similarity) else None

    def top_k(self, query_vector: list[float], k: int) -> list[tuple[int, float]]:
        """Find the top-k most similar vectors in the index to a query vector.

        Similarity is computed using cosine similarity. Returns pairs of
        (index, score), sorted by descending similarity score.

        Args:
            query_vector (list[float]): The embedding vector to query with.
            k (int): The maximum number of top matches to return.

        Returns:
            list[tuple[int, float]]: List of (index, similarity score) sorted descending.
                The score is a float in [-1, 1], where 1 is most similar.
        """
        if k <= 0:
            return []

        prepared = self._query(query_vector)
        if prepared is None:
            return []
        query, query_norm = prepared
        dots = np.dot(self.vectors, query)
        valid_mask = np.isfinite(self.norms) & (self.norms > 0)
        scores = np.zeros_like(dots)
        scores[valid_mask] = dots[valid_mask] / (self.norms[valid_mask] * query_norm)

        n = len(scores)
        idx = np.arange(n)
        sorted_indices = np.lexsort((idx, -scores))

        results: list[tuple[int, float]] = []
        for idx_val in sorted_indices[:k]:
            if valid_mask[idx_val] and isfinite(float(scores[idx_val])):
                results.append((int(idx_val), float(scores[idx_val])))
        return results


def _embedding_matrix(
    rows: list[list[float]], expected_count: int
) -> npt.NDArray[np.float32]:
    try:
        vectors = np.asarray(rows, dtype=np.float32)
    except (TypeError, ValueError, OverflowError):
        return np.zeros((expected_count, 0), dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != expected_count:
        return np.zeros((expected_count, 0), dtype=np.float32)
    return vectors
