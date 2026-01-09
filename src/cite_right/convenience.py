"""Convenience functions for common RAG post-processing workflows.

This module provides high-level helper functions that simplify common tasks
when using cite-right as a post-processing step in RAG pipelines.
"""

from __future__ import annotations

from typing import Literal, Sequence

from cite_right.citations import align_citations
from cite_right.core.citation_config import CitationConfig
from cite_right.core.interfaces import AnswerSegmenter, Segmenter, Tokenizer
from cite_right.core.results import SourceChunk, SourceDocument, SpanCitations
from cite_right.hallucination import (
    HallucinationConfig,
    HallucinationMetrics,
    compute_hallucination_metrics,
)
from cite_right.models.base import Embedder


def is_grounded(
    answer: str,
    sources: Sequence[str | SourceDocument | SourceChunk],
    *,
    threshold: float = 0.5,
    config: CitationConfig | None = None,
    hallucination_config: HallucinationConfig | None = None,
    tokenizer: Tokenizer | None = None,
    embedder: Embedder | None = None,
    backend: Literal["auto", "python", "rust"] = "auto",
) -> bool:
    """Check if an answer is sufficiently grounded in source documents.

    This is a convenience function for RAG quality gates that returns a simple
    boolean indicating whether the answer meets a groundedness threshold.

    Args:
        answer: The answer text to check.
        sources: Source documents or text strings to verify against.
        threshold: Minimum groundedness score (0-1) to consider the answer
            grounded. Default 0.5 (50% grounded).
        config: Citation configuration options.
        hallucination_config: Configuration for hallucination metrics.
        tokenizer: Custom tokenizer (default: SimpleTokenizer).
        embedder: Optional embedder for semantic matching.
        backend: Alignment backend ("auto", "python", or "rust").

    Returns:
        True if the answer's groundedness score >= threshold, False otherwise.

    Example:
        >>> from cite_right import is_grounded
        >>> answer = "Revenue grew 15% in Q4."
        >>> sources = ["Annual report: Revenue grew 15% in Q4 2024."]
        >>> if is_grounded(answer, sources, threshold=0.6):
        ...     print("Answer is well-grounded!")
        ... else:
        ...     print("Answer may contain hallucinations")
    """
    results = align_citations(
        answer,
        sources,
        config=config,
        tokenizer=tokenizer,
        embedder=embedder,
        backend=backend,
    )
    metrics = compute_hallucination_metrics(results, config=hallucination_config)
    return metrics.groundedness_score >= threshold


def is_hallucinated(
    answer: str,
    sources: Sequence[str | SourceDocument | SourceChunk],
    *,
    threshold: float = 0.5,
    config: CitationConfig | None = None,
    hallucination_config: HallucinationConfig | None = None,
    tokenizer: Tokenizer | None = None,
    embedder: Embedder | None = None,
    backend: Literal["auto", "python", "rust"] = "auto",
) -> bool:
    """Check if an answer contains significant hallucinations.

    This is a convenience function that returns True if the hallucination rate
    exceeds the given threshold.

    Args:
        answer: The answer text to check.
        sources: Source documents or text strings to verify against.
        threshold: Maximum hallucination rate (0-1) to consider acceptable.
            Default 0.5 (50% hallucination rate).
        config: Citation configuration options.
        hallucination_config: Configuration for hallucination metrics.
        tokenizer: Custom tokenizer (default: SimpleTokenizer).
        embedder: Optional embedder for semantic matching.
        backend: Alignment backend ("auto", "python", or "rust").

    Returns:
        True if the answer's hallucination rate > threshold, False otherwise.

    Example:
        >>> from cite_right import is_hallucinated
        >>> answer = "The company announced plans to colonize Mars."
        >>> sources = ["Annual report: Revenue grew 15% in Q4 2024."]
        >>> if is_hallucinated(answer, sources, threshold=0.3):
        ...     print("Warning: Answer may contain hallucinations!")
    """
    results = align_citations(
        answer,
        sources,
        config=config,
        tokenizer=tokenizer,
        embedder=embedder,
        backend=backend,
    )
    metrics = compute_hallucination_metrics(results, config=hallucination_config)
    return metrics.hallucination_rate > threshold


def check_groundedness(
    answer: str,
    sources: Sequence[str | SourceDocument | SourceChunk],
    *,
    config: CitationConfig | None = None,
    hallucination_config: HallucinationConfig | None = None,
    tokenizer: Tokenizer | None = None,
    answer_segmenter: AnswerSegmenter | None = None,
    source_segmenter: Segmenter | None = None,
    embedder: Embedder | None = None,
    backend: Literal["auto", "python", "rust"] = "auto",
) -> HallucinationMetrics:
    """Compute groundedness metrics for an answer in one call.

    This is a convenience function that combines align_citations() and
    compute_hallucination_metrics() into a single call.

    Args:
        answer: The answer text to check.
        sources: Source documents or text strings to verify against.
        config: Citation configuration options.
        hallucination_config: Configuration for hallucination metrics.
        tokenizer: Custom tokenizer (default: SimpleTokenizer).
        answer_segmenter: Custom answer segmenter.
        source_segmenter: Custom source segmenter.
        embedder: Optional embedder for semantic matching.
        backend: Alignment backend ("auto", "python", or "rust").

    Returns:
        HallucinationMetrics with groundedness score, hallucination rate,
        and per-span details.

    Example:
        >>> from cite_right import check_groundedness
        >>> answer = "Revenue grew 15%. Profits doubled."
        >>> sources = ["Annual report: Revenue grew 15% in Q4."]
        >>> metrics = check_groundedness(answer, sources)
        >>> print(f"Groundedness: {metrics.groundedness_score:.1%}")
        >>> print(f"Unsupported: {[s.text for s in metrics.unsupported_spans]}")
    """
    results = align_citations(
        answer,
        sources,
        config=config,
        tokenizer=tokenizer,
        answer_segmenter=answer_segmenter,
        source_segmenter=source_segmenter,
        embedder=embedder,
        backend=backend,
    )
    return compute_hallucination_metrics(results, config=hallucination_config)


def annotate_answer(
    answer: str,
    sources: Sequence[str | SourceDocument | SourceChunk],
    *,
    config: CitationConfig | None = None,
    tokenizer: Tokenizer | None = None,
    answer_segmenter: AnswerSegmenter | None = None,
    source_segmenter: Segmenter | None = None,
    embedder: Embedder | None = None,
    backend: Literal["auto", "python", "rust"] = "auto",
    format: Literal["markdown", "superscript", "footnote"] = "markdown",
    include_unsupported: bool = True,
) -> str:
    """Add inline citation markers to an answer.

    This function takes an answer and its sources, performs citation alignment,
    and returns the answer with inline citation markers inserted after each
    sentence/span.

    Args:
        answer: The answer text to annotate.
        sources: Source documents or text strings.
        config: Citation configuration options.
        tokenizer: Custom tokenizer (default: SimpleTokenizer).
        answer_segmenter: Custom answer segmenter.
        source_segmenter: Custom source segmenter.
        embedder: Optional embedder for semantic matching.
        backend: Alignment backend ("auto", "python", or "rust").
        format: Citation format:

            - ``"markdown"``: Uses [1], [2] style markers (default).
            - ``"superscript"``: Uses ^1, ^2 style markers.
            - ``"footnote"``: Uses [^1], [^2] style markers.

        include_unsupported: If True, marks unsupported spans with [?].

    Returns:
        The answer text with citation markers inserted.

    Example:
        >>> from cite_right import SourceDocument, annotate_answer
        >>> answer = "Revenue grew 15%. Profits doubled."
        >>> sources = [
        ...     SourceDocument(id="report", text="Revenue grew 15% in Q4."),
        ... ]
        >>> annotated = annotate_answer(answer, sources)
        >>> print(annotated)
        Revenue grew 15%.[1] Profits doubled.[?]
    """
    results = align_citations(
        answer,
        sources,
        config=config,
        tokenizer=tokenizer,
        answer_segmenter=answer_segmenter,
        source_segmenter=source_segmenter,
        embedder=embedder,
        backend=backend,
    )
    return format_with_citations(
        answer, results, format=format, include_unsupported=include_unsupported
    )


def format_with_citations(
    answer: str,
    span_citations: Sequence[SpanCitations],
    *,
    format: Literal["markdown", "superscript", "footnote"] = "markdown",
    include_unsupported: bool = True,
) -> str:
    """Format an answer with citation markers from pre-computed results.

    This function takes pre-computed citation alignment results and inserts
    citation markers into the answer text.

    Args:
        answer: The original answer text.
        span_citations: Citation results from align_citations().
        format: Citation format style.
        include_unsupported: If True, marks unsupported spans with [?].

    Returns:
        The answer text with citation markers inserted.

    Example:
        >>> from cite_right import align_citations, format_with_citations
        >>> results = align_citations(answer, sources)
        >>> annotated = format_with_citations(answer, results)
    """
    if not span_citations:
        return answer

    source_numbers: dict[str, int] = {}
    next_number = 1
    for sc in span_citations:
        for citation in sc.citations:
            if citation.source_id not in source_numbers:
                source_numbers[citation.source_id] = next_number
                next_number += 1

    def _format_marker(source_ids: list[str]) -> str:
        if not source_ids:
            return "[?]" if include_unsupported else ""
        numbers = sorted(
            source_numbers[sid] for sid in source_ids if sid in source_numbers
        )
        if format == "superscript":
            return "".join(f"^{n}" for n in numbers)
        elif format == "footnote":
            return "".join(f"[^{n}]" for n in numbers)
        else:  # markdown
            return "".join(f"[{n}]" for n in numbers)

    sorted_spans = sorted(
        span_citations, key=lambda sc: sc.answer_span.char_end, reverse=True
    )

    result = answer
    for sc in sorted_spans:
        source_ids = list({c.source_id for c in sc.citations})
        marker = _format_marker(source_ids)
        if marker:
            end = sc.answer_span.char_end
            insert_pos = end
            while (
                insert_pos > sc.answer_span.char_start
                and result[insert_pos - 1] in " \t\n\r"
            ):
                insert_pos -= 1
            result = result[:insert_pos] + marker + result[insert_pos:]

    return result


def get_citation_summary(
    span_citations: Sequence[SpanCitations],
) -> str:
    """Generate a human-readable summary of citation results."""
    if not span_citations:
        return "Citation Summary: No spans to analyze"

    counts = _count_statuses(span_citations)
    source_ids = _collect_source_ids(span_citations)
    return _format_summary(counts, len(span_citations), source_ids)


def _count_statuses(span_citations: Sequence[SpanCitations]) -> dict[str, int]:
    """Count spans by status."""
    counts = {"supported": 0, "partial": 0, "unsupported": 0}
    for sc in span_citations:
        counts[sc.status] = counts.get(sc.status, 0) + 1
    return counts


def _collect_source_ids(span_citations: Sequence[SpanCitations]) -> set[str]:
    """Collect unique source IDs from all citations."""
    source_ids: set[str] = set()
    for sc in span_citations:
        for c in sc.citations:
            source_ids.add(c.source_id)
    return source_ids


def _format_summary(counts: dict[str, int], total: int, source_ids: set[str]) -> str:
    """Format the summary string."""
    lines = ["Citation Summary:"]
    lines.append(f"- {counts['supported']} of {total} spans fully supported")
    if counts["partial"] > 0:
        lines.append(f"- {counts['partial']} spans partially supported")
    if counts["unsupported"] > 0:
        lines.append(f"- {counts['unsupported']} spans unsupported")
    if source_ids:
        lines.append(f"- Sources cited: {', '.join(sorted(source_ids))}")
    return "\n".join(lines)
