from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CitationWeights:
    alignment: float = 1.0
    answer_coverage: float = 1.0
    evidence_coverage: float = 0.0
    lexical: float = 0.5
    embedding: float = 0.5


@dataclass(frozen=True, slots=True)
class CitationConfig:
    """Configuration for `cite_right.align_citations`.

    Attributes:
        multi_span_evidence: If True, attempt to return non-contiguous evidence via
            `Citation.evidence_spans` when alignment indicates multiple disjoint match
            regions. The legacy `Citation.char_start/char_end/evidence` fields remain
            a single contiguous (enclosing) span for backward compatibility.
        multi_span_merge_gap_chars: Merge neighboring evidence spans when the gap
            between them is <= this many characters in the source document.
        multi_span_max_spans: Maximum number of evidence spans to return per
            citation after merging. If exceeded, the citation falls back to a single
            contiguous evidence span.
    """

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

    multi_span_evidence: bool = False
    multi_span_merge_gap_chars: int = 16
    multi_span_max_spans: int = 5
