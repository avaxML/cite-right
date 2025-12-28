from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from cite_right.models.base import Embedder


@dataclass(frozen=True)
class EmbeddingIndex:
    vectors: list[list[float]]
    norms: list[float]

    @classmethod
    def build(cls, embedder: Embedder, texts: Sequence[str]) -> "EmbeddingIndex":
        vectors = embedder.encode(texts)
        norms = [_l2_norm(vector) for vector in vectors]
        return cls(vectors=vectors, norms=norms)

    def top_k(self, query_vector: list[float], k: int) -> list[tuple[int, float]]:
        if k <= 0:
            return []
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
    return math.sqrt(sum(x * x for x in vector))
