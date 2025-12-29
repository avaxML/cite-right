from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from cite_right.models.base import Embedder


@dataclass(frozen=True, slots=True)
class EmbeddingIndex:
    vectors: npt.NDArray[np.float32]
    norms: npt.NDArray[np.float32]

    @classmethod
    def build(cls, embedder: Embedder, texts: Sequence[str]) -> "EmbeddingIndex":
        raw_vectors = embedder.encode(texts)
        vectors = np.array(raw_vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1).astype(np.float32)
        return cls(vectors=vectors, norms=norms)

    def top_k(self, query_vector: list[float], k: int) -> list[tuple[int, float]]:
        if k <= 0:
            return []

        query = np.array(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0.0:
            return []

        # Vectorized dot product and cosine similarity
        dots = np.dot(self.vectors, query)
        # Avoid division by zero for zero-norm vectors
        valid_mask = self.norms > 0
        scores = np.zeros_like(dots)
        scores[valid_mask] = dots[valid_mask] / (self.norms[valid_mask] * query_norm)

        # Get indices sorted by score descending, then by index ascending for ties
        # Use negative scores for descending sort, indices for ascending tie-break
        sort_keys = list(enumerate(scores))
        sort_keys.sort(key=lambda item: (-item[1], item[0]))

        results: list[tuple[int, float]] = []
        for idx, score in sort_keys[:k]:
            if self.norms[idx] > 0:
                results.append((idx, float(score)))
        return results
