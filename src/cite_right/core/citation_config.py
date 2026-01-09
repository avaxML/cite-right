"""Configuration for the citation alignment pipeline."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    pass


class CitationWeights(BaseModel):
    """Weights for the citation score components."""
    model_config = ConfigDict(frozen=True)

    alignment: float = 1.0
    answer_coverage: float = 1.0
    evidence_coverage: float = 0.0
    lexical: float = 0.5
    embedding: float = 0.5


class CitationConfig(BaseModel):
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

    Presets:
        Use class methods for common configurations:

        - :meth:`strict`: High-precision mode requiring strong evidence.
        - :meth:`permissive`: Lenient mode for paraphrased content.
        - :meth:`fast`: Speed-optimized with reduced candidates.
        - :meth:`balanced`: Default balanced configuration.

    Example:
        >>> from cite_right import CitationConfig, align_citations
        >>> config = CitationConfig.strict()
        >>> results = align_citations(answer, sources, config=config)
    """

    model_config = ConfigDict(frozen=True)

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

    weights: CitationWeights = Field(default_factory=CitationWeights)

    match_score: int = 2
    mismatch_score: int = -1
    gap_score: int = -1

    prefer_source_order: bool = True

    multi_span_evidence: bool = False
    multi_span_merge_gap_chars: int = 16
    multi_span_max_spans: int = 5

    @classmethod
    def strict(cls) -> "CitationConfig":
        """High-precision configuration requiring strong evidence.

        Use this when you want to minimize false positives and only accept
        citations with high answer coverage. Good for fact-checking and
        high-stakes applications.

        Returns:
            CitationConfig with strict thresholds:

            - Higher ``min_answer_coverage`` (0.4)
            - Higher ``supported_answer_coverage`` (0.7)
            - Lower ``top_k`` (2)

        Example:
            >>> config = CitationConfig.strict()
            >>> results = align_citations(answer, sources, config=config)
        """
        return cls(
            top_k=2,
            min_answer_coverage=0.4,
            supported_answer_coverage=0.7,
            min_final_score=0.3,
            max_citations_per_source=1,
        )

    @classmethod
    def permissive(cls) -> "CitationConfig":
        """Lenient configuration for paraphrased or summarized content.

        Use this when the answer may be significantly paraphrased from sources.
        Accepts lower coverage thresholds and is more tolerant of partial matches.

        Returns:
            CitationConfig with permissive thresholds:

            - Lower ``min_answer_coverage`` (0.15)
            - Lower ``supported_answer_coverage`` (0.4)
            - Higher ``top_k`` (5)
            - Embedding-only citations allowed

        Example:
            >>> config = CitationConfig.permissive()
            >>> results = align_citations(answer, sources, config=config)
        """
        return cls(
            top_k=5,
            min_answer_coverage=0.15,
            supported_answer_coverage=0.4,
            min_final_score=0.0,
            allow_embedding_only=True,
            min_embedding_similarity=0.25,
            supported_embedding_similarity=0.5,
            max_citations_per_source=3,
        )

    @classmethod
    def fast(cls) -> "CitationConfig":
        """Speed-optimized configuration with reduced candidate evaluation.

        Use this when processing large volumes and speed is critical.
        Reduces the number of candidates evaluated per span.

        Returns:
            CitationConfig with reduced candidate limits:

            - Lower ``max_candidates_lexical`` (50)
            - Lower ``max_candidates_total`` (100)
            - ``top_k`` of 1

        Example:
            >>> config = CitationConfig.fast()
            >>> results = align_citations(answer, sources, config=config)
        """
        return cls(
            top_k=1,
            max_candidates_lexical=50,
            max_candidates_embedding=50,
            max_candidates_total=100,
            max_citations_per_source=1,
        )

    @classmethod
    def balanced(cls) -> "CitationConfig":
        """Balanced default configuration (same as default constructor).

        This is equivalent to ``CitationConfig()`` but provides a named
        alternative for explicit code readability.

        Returns:
            CitationConfig with default values.

        Example:
            >>> config = CitationConfig.balanced()
            >>> results = align_citations(answer, sources, config=config)
        """
        return cls()
