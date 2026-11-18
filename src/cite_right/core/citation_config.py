"""Configuration for the citation alignment pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from cite_right.contradiction import is_protected_alias
from cite_right.text.term_aliases import is_single_word, normalize_alias


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
    min_embedding_similarity: float = 0.3

    window_size_sentences: int = 1
    window_stride_sentences: int = 1

    max_candidates_lexical: int = 200
    max_candidates_embedding: int = 200
    max_candidates_total: int = 400

    max_citations_per_source: int = 2
    max_retrieval_support: int = 3
    require_all_answer_tokens_in_evidence: bool = False

    weights: CitationWeights = Field(default_factory=CitationWeights)

    match_score: int = 2
    mismatch_score: int = -1
    gap_score: int = -1

    prefer_source_order: bool = True

    multi_span_evidence: bool = False
    multi_span_merge_gap_chars: int = 16
    multi_span_max_spans: int = 5

    term_aliases: tuple[tuple[str, ...], ...] = Field(
        default=(),
        description=(
            "Groups of equivalent single words. Declaring aliases also requires "
            "every content word of the answer to be matched. Unmatched content "
            "words keep the span `partial` when a citation is available."
        ),
    )

    @field_validator("term_aliases")
    @classmethod
    def _validate_term_aliases(
        cls, groups: tuple[tuple[str, ...], ...]
    ) -> tuple[tuple[str, ...], ...]:
        seen: set[str] = set()
        normalized_groups = []
        for group in groups:
            if len(group) < 2:
                raise ValueError(
                    f"alias group {group!r} needs at least two distinct terms"
                )
            normalized = []
            for term in group:
                word = normalize_alias(term)
                if (
                    term.isnumeric()
                    or not is_single_word(word)
                    or is_protected_alias(word)
                ):
                    raise ValueError(
                        f"invalid alias term {term!r}; use a single non-numeric word without negation or polarity markers"
                    )
                if word in seen:
                    raise ValueError(
                        f"duplicate alias term {term!r} after normalization"
                    )
                seen.add(word)
                normalized.append(word)
            normalized_groups.append(tuple(sorted(normalized)))
        return tuple(sorted(normalized_groups))

    def model_copy(
        self, *, update: Mapping[str, object] | None = None, deep: bool = False
    ) -> Self:
        if update and "term_aliases" in update:
            aliases = CitationConfig.model_validate(
                {"term_aliases": update["term_aliases"]}
            ).term_aliases
            update = {**update, "term_aliases": aliases}
        return super().model_copy(update=update, deep=deep)

    @model_serializer(mode="wrap")
    # A return annotation would replace Pydantic's generated field schema.
    def _serialize_config(self, handler: SerializerFunctionWrapHandler):  # noqa: ANN202
        payload = handler(self)
        if not self.term_aliases:
            payload.pop("term_aliases", None)
        return payload

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_embedding_only_options(cls, data: object) -> object:
        if isinstance(data, dict) and {
            "allow_embedding_only",
            "supported_embedding_similarity",
        }.intersection(data):
            raise ValueError(
                "allow_embedding_only and supported_embedding_similarity were "
                "removed; inspect SpanCitations.retrieval_support for semantic "
                "retrieval signals"
            )
        return data

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
            max_retrieval_support=2,
            require_all_answer_tokens_in_evidence=True,
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
            - Lower embedding threshold for retrieval candidate expansion

        Example:
            >>> config = CitationConfig.permissive()
            >>> results = align_citations(answer, sources, config=config)
        """
        return cls(
            top_k=5,
            min_answer_coverage=0.15,
            supported_answer_coverage=0.4,
            min_final_score=0.0,
            min_embedding_similarity=0.25,
            max_citations_per_source=3,
            max_retrieval_support=5,
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
            max_retrieval_support=1,
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
