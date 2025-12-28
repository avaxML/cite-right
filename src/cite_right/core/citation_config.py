from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CitationWeights:
    alignment: float = 1.0
    answer_coverage: float = 1.0
    evidence_coverage: float = 0.0
    lexical: float = 0.5
    embedding: float = 0.5


@dataclass(frozen=True)
class CitationConfig:
    top_k: int = 3
    min_final_score: float = 0.0
    min_alignment_score: int = 0
    min_answer_coverage: float = 0.2
    supported_answer_coverage: float = 0.6
    allow_embedding_only: bool = False
    min_embedding_similarity: float = 0.3
    supported_embedding_similarity: float = 0.6

    window_size_sentences: int = 1
    window_stride_sentences: int = 1

    max_candidates_lexical: int = 200
    max_candidates_embedding: int = 200
    max_candidates_total: int = 400

    max_citations_per_source: int = 2

    weights: CitationWeights = field(default_factory=CitationWeights)

    match_score: int = 2
    mismatch_score: int = -1
    gap_score: int = -1

    prefer_source_order: bool = True
