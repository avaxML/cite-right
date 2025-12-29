from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from cite_right.models.base import Embedder

# Try to import numpy for faster operations (available when embeddings extra is installed)
try:
    import numpy as _np  # type: ignore[import-not-found]

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


@dataclass(frozen=True, slots=True)
class EmbeddingIndex:
    vectors: list[list[float]]
    norms: list[float]
    _np_vectors: Any  # numpy.ndarray when numpy is available, None otherwise
    _np_norms: Any  # numpy.ndarray when numpy is available, None otherwise

    @classmethod
    def build(cls, embedder: Embedder, texts: Sequence[str]) -> "EmbeddingIndex":
        vectors = embedder.encode(texts)
        if _HAS_NUMPY:
            np_vectors = _np.array(vectors, dtype=_np.float32)
            np_norms = _np.linalg.norm(np_vectors, axis=1).astype(_np.float32)
            norms = np_norms.tolist()
            return cls(
                vectors=vectors,
                norms=norms,
                _np_vectors=np_vectors,
                _np_norms=np_norms,
            )
        norms = [_l2_norm(vector) for vector in vectors]
        return cls(vectors=vectors, norms=norms, _np_vectors=None, _np_norms=None)

    def top_k(self, query_vector: list[float], k: int) -> list[tuple[int, float]]:
        if k <= 0:
            return []

        if _HAS_NUMPY and self._np_vectors is not None and self._np_norms is not None:
            return self._top_k_numpy(query_vector, k)

        return self._top_k_python(query_vector, k)

    def _top_k_numpy(
        self, query_vector: list[float], k: int
    ) -> list[tuple[int, float]]:
        query = _np.array(query_vector, dtype=_np.float32)
        query_norm = _np.linalg.norm(query)
        if query_norm == 0.0:
            return []

        # Vectorized dot product and cosine similarity
        dots = _np.dot(self._np_vectors, query)
        # Avoid division by zero for zero-norm vectors
        valid_mask = self._np_norms > 0
        scores = _np.zeros_like(dots)
        scores[valid_mask] = dots[valid_mask] / (
            self._np_norms[valid_mask] * query_norm
        )

        # Get indices sorted by score descending, then by index ascending for ties
        # Use negative scores for descending sort, indices for ascending tie-break
        sort_keys = list(enumerate(scores))
        sort_keys.sort(key=lambda item: (-item[1], item[0]))

        results: list[tuple[int, float]] = []
        for idx, score in sort_keys[:k]:
            if self._np_norms[idx] > 0:
                results.append((idx, float(score)))
        return results

    def _top_k_python(
        self, query_vector: list[float], k: int
    ) -> list[tuple[int, float]]:
        query_norm = _l2_norm(query_vector)
        if query_norm == 0.0:
            return []

        scores: list[tuple[int, float]] = []
        for idx, (vec, norm) in enumerate(zip(self.vectors, self.norms, strict=False)):
            if norm == 0.0:
                continue
            score = _dot(query_vector, vec) / (query_norm * norm)
            scores.append((idx, score))

        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[:k]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))


def _l2_norm(vector: Sequence[float]) -> float:
    return sum(x * x for x in vector) ** 0.5
