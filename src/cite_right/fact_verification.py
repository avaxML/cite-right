"""Fact-level verification for RAG responses.

This module provides fine-grained verification by decomposing answer
sentences into atomic claims and verifying each claim independently
against source documents.
"""

from __future__ import annotations

from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from cite_right.citations import align_citations
from cite_right.claims import (
    Claim,
    ClaimDecomposer,
    SimpleClaimDecomposer,
)
from cite_right.core.citation_config import CitationConfig
from cite_right.core.interfaces import AnswerSegmenter, Segmenter, Tokenizer
from cite_right.core.results import Citation, SourceChunk, SourceDocument
from cite_right.models.base import Embedder
from cite_right.text.answer_segmenter import SimpleAnswerSegmenter


class FactVerificationConfig(BaseModel):
    """Configuration for fact verification.

    Attributes:
        verified_coverage_threshold: Minimum answer_coverage for a claim
            to be considered "verified". Default 0.6.
        partial_coverage_threshold: Minimum answer_coverage for a claim
            to be considered "partial". Below this is "unverified". Default 0.3.
        citation_config: Configuration passed to align_citations.
            If None, uses default CitationConfig.
    """

    model_config = ConfigDict(frozen=True)

    verified_coverage_threshold: float = 0.6
    partial_coverage_threshold: float = 0.3
    citation_config: CitationConfig | None = None


class ClaimVerification(BaseModel):
    """Verification result for a single claim.

    Attributes:
        claim: The claim being verified.
        status: Verification status.
            - "verified": Claim is well-supported by sources.
            - "partial": Claim has some support but below threshold.
            - "unverified": Claim has no or very weak support.
        confidence: Confidence score (0-1) based on best citation coverage.
        best_citation: Best matching citation, if any.
        all_citations: All citations found for this claim.
        source_ids: List of source IDs that support this claim.
    """

    model_config = ConfigDict(frozen=True)

    claim: Claim
    status: Literal["verified", "partial", "unverified"]
    confidence: float
    best_citation: Citation | None = None
    all_citations: list[Citation] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class FactVerificationMetrics(BaseModel):
    """Aggregate metrics for fact-level verification.

    Attributes:
        num_claims: Total number of atomic claims analyzed.
        num_verified: Number of claims that are verified.
        num_partial: Number of claims with partial support.
        num_unverified: Number of claims with no support.
        verification_rate: Proportion of verified claims (0-1).
        avg_confidence: Average confidence across all claims.
        min_confidence: Minimum confidence across all claims.
        claim_verifications: Per-claim verification details.
        verified_claims: List of verified claims.
        unverified_claims: List of unverified claims.
        partial_claims: List of partially verified claims.
    """

    model_config = ConfigDict(frozen=True)

    num_claims: int
    num_verified: int
    num_partial: int
    num_unverified: int
    verification_rate: float
    avg_confidence: float
    min_confidence: float
    claim_verifications: list[ClaimVerification] = Field(default_factory=list)
    verified_claims: list[Claim] = Field(default_factory=list)
    unverified_claims: list[Claim] = Field(default_factory=list)
    partial_claims: list[Claim] = Field(default_factory=list)


def verify_facts(
    answer: str,
    sources: Sequence[str | SourceDocument | SourceChunk],
    *,
    config: FactVerificationConfig | None = None,
    claim_decomposer: ClaimDecomposer | None = None,
    answer_segmenter: AnswerSegmenter | None = None,
    source_segmenter: Segmenter | None = None,
    tokenizer: Tokenizer | None = None,
    embedder: Embedder | None = None,
    backend: Literal["auto", "python", "rust"] = "auto",
) -> FactVerificationMetrics:
    """Verify facts in an answer against source documents."""
    cfg = config or FactVerificationConfig()
    decomposer = claim_decomposer or SimpleClaimDecomposer()
    segmenter = answer_segmenter or SimpleAnswerSegmenter()

    citation_config = cfg.citation_config or CitationConfig(
        top_k=3,
        min_answer_coverage=0.2,
        supported_answer_coverage=cfg.verified_coverage_threshold,
    )

    all_claims = _decompose_answer(answer, segmenter, decomposer)

    if not all_claims:
        return _empty_verification_metrics()

    return _verify_all_claims(
        all_claims, sources, cfg, citation_config,
        source_segmenter, tokenizer, embedder, backend,
    )


def _decompose_answer(
    answer: str, segmenter: AnswerSegmenter, decomposer: ClaimDecomposer
) -> list[Claim]:
    """Segment answer and decompose into atomic claims."""
    answer_spans = segmenter.segment(answer)
    all_claims: list[Claim] = []
    for span in answer_spans:
        all_claims.extend(decomposer.decompose(span))
    return all_claims


def _empty_verification_metrics() -> FactVerificationMetrics:
    """Return metrics for empty input."""
    return FactVerificationMetrics(
        num_claims=0,
        num_verified=0,
        num_partial=0,
        num_unverified=0,
        verification_rate=1.0,
        avg_confidence=1.0,
        min_confidence=1.0,
        claim_verifications=[],
        verified_claims=[],
        unverified_claims=[],
        partial_claims=[],
    )


def _verify_all_claims(
    claims: list[Claim],
    sources: Sequence[str | SourceDocument | SourceChunk],
    cfg: FactVerificationConfig,
    citation_config: CitationConfig,
    source_segmenter: Segmenter | None,
    tokenizer: Tokenizer | None,
    embedder: Embedder | None,
    backend: Literal["auto", "python", "rust"],
) -> FactVerificationMetrics:
    """Verify all claims and aggregate results."""
    verifications: list[ClaimVerification] = []
    verified: list[Claim] = []
    unverified: list[Claim] = []
    partial: list[Claim] = []
    confidence_values: list[float] = []

    for claim in claims:
        v = _verify_claim(
            claim=claim, sources=sources, config=cfg,
            citation_config=citation_config, source_segmenter=source_segmenter,
            tokenizer=tokenizer, embedder=embedder, backend=backend,
        )
        verifications.append(v)
        confidence_values.append(v.confidence)
        _categorize_claim(claim, v.status, verified, partial, unverified)

    return FactVerificationMetrics(
        num_claims=len(claims),
        num_verified=len(verified),
        num_partial=len(partial),
        num_unverified=len(unverified),
        verification_rate=len(verified) / len(claims) if claims else 1.0,
        avg_confidence=sum(confidence_values) / len(confidence_values) if confidence_values else 1.0,
        min_confidence=min(confidence_values) if confidence_values else 1.0,
        claim_verifications=verifications,
        verified_claims=verified,
        unverified_claims=unverified,
        partial_claims=partial,
    )


def _categorize_claim(
    claim: Claim,
    status: str,
    verified: list[Claim],
    partial: list[Claim],
    unverified: list[Claim],
) -> None:
    """Add claim to the appropriate list based on status."""
    if status == "verified":
        verified.append(claim)
    elif status == "partial":
        partial.append(claim)
    else:
        unverified.append(claim)


def _verify_claim(
    claim: Claim,
    sources: Sequence[str | SourceDocument | SourceChunk],
    config: FactVerificationConfig,
    citation_config: CitationConfig,
    source_segmenter: Segmenter | None,
    tokenizer: Tokenizer | None,
    embedder: Embedder | None,
    backend: Literal["auto", "python", "rust"],
) -> ClaimVerification:
    """Verify a single claim against sources."""
    # Use align_citations to find evidence for this claim
    # We pass the claim text as a single-sentence "answer"
    results = align_citations(
        answer=claim.text,
        sources=sources,
        config=citation_config,
        backend=backend,
        source_segmenter=source_segmenter,
        tokenizer=tokenizer,
        embedder=embedder,
    )

    # Extract citations from results
    all_citations: list[Citation] = []
    for span_result in results:
        all_citations.extend(span_result.citations)

    if not all_citations:
        return ClaimVerification(
            claim=claim,
            status="unverified",
            confidence=0.0,
            best_citation=None,
            all_citations=[],
            source_ids=[],
        )

    # Get best citation by score
    best_citation = max(all_citations, key=lambda c: c.score)
    answer_coverage = float(best_citation.components.get("answer_coverage", 0.0))

    # Determine status based on coverage
    if answer_coverage >= config.verified_coverage_threshold:
        status: Literal["verified", "partial", "unverified"] = "verified"
    elif answer_coverage >= config.partial_coverage_threshold:
        status = "partial"
    else:
        status = "unverified"

    # Collect unique source IDs
    source_ids = list({c.source_id for c in all_citations})

    return ClaimVerification(
        claim=claim,
        status=status,
        confidence=answer_coverage,
        best_citation=best_citation,
        all_citations=all_citations,
        source_ids=source_ids,
    )
