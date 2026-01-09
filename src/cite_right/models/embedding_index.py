"""Index for fast similarity search over embedding vectors."""

from __future__ import annotations

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
        raw_vectors = embedder.encode(texts)
        vectors = np.array(raw_vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1).astype(np.float32)
        return cls(vectors=vectors, norms=norms)

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

        query = np.array(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0.0:
            return []

        dots = np.dot(self.vectors, query)
        valid_mask = self.norms > 0
        scores = np.zeros_like(dots)
        scores[valid_mask] = dots[valid_mask] / (self.norms[valid_mask] * query_norm)

        sort_keys = list(enumerate(scores))
        sort_keys.sort(key=lambda item: (-item[1], item[0]))

        results: list[tuple[int, float]] = []
        for idx, score in sort_keys[:k]:
            if self.norms[idx] > 0:
                results.append((idx, float(score)))
        return results
