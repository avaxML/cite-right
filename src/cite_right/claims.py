"""Claim decomposition for fact-level verification.

This module provides tools to decompose sentences into atomic claims
using spaCy dependency parsing, enabling fine-grained fact verification.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from cite_right.core.results import AnswerSpan


class Claim(BaseModel):
    """An atomic claim extracted from an answer span.

    Attributes:
        text: The claim text.
        char_start: Absolute 0-based start offset in the original answer.
        char_end: Absolute 0-based end offset (exclusive) in the original answer.
        source_span: The AnswerSpan this claim was extracted from.
        claim_index: Index of this claim within the source span.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    char_start: int
    char_end: int
    source_span: AnswerSpan
    claim_index: int = 0


@runtime_checkable
class ClaimDecomposer(Protocol):
    """Protocol for claim decomposition strategies."""

    def decompose(self, span: AnswerSpan) -> list[Claim]:
        """Decompose an answer span into atomic claims.

        Args:
            span: The answer span to decompose.

        Returns:
            List of claims. Returns a single claim wrapping the entire
            span if no decomposition is possible.
        """
        ...


class SimpleClaimDecomposer:
    """Simple claim decomposer that returns spans unchanged.

    This is a fallback when spaCy is not available. It treats each
    answer span as a single atomic claim.
    """

    def decompose(self, span: AnswerSpan) -> list[Claim]:
        """Return the span as a single claim."""
        return [
            Claim(
                text=span.text,
                char_start=span.char_start,
                char_end=span.char_end,
                source_span=span,
                claim_index=0,
            )
        ]


class SpacyClaimDecomposer:
    """Claim decomposer using spaCy dependency parsing.

    Uses the `conj` (conjunction) relation to identify coordinated
    clauses and split them into separate claims. This works across
    languages that spaCy supports.

    Example:
        >>> decomposer = SpacyClaimDecomposer()
        >>> span = AnswerSpan(
        ...     text="Revenue grew and profits increased",
        ...     char_start=0, char_end=34
        ... )
        >>> claims = decomposer.decompose(span)
        >>> [c.text for c in claims]
        ['Revenue grew', 'profits increased']
    """

    def __init__(
        self,
        model: str = "en_core_web_sm",
        *,
        min_claim_tokens: int = 2,
    ) -> None:
        """Initialize the claim decomposer.

        Args:
            model: spaCy model name to load.
            min_claim_tokens: Minimum tokens for a valid claim.
                Claims with fewer tokens are merged back.
        """
        try:
            import spacy  # pyright: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "spaCy is not installed. Install with 'cite-right[spacy]'."
            ) from exc

        try:
            self._nlp = spacy.load(model)
        except OSError as exc:  # pragma: no cover - model guard
            raise RuntimeError(
                f"spaCy model '{model}' is not installed. "
                "Run: python -m spacy download en_core_web_sm"
            ) from exc

        self._min_claim_tokens = min_claim_tokens

    def decompose(self, span: AnswerSpan) -> list[Claim]:
        """Decompose an answer span into atomic claims using dependency parsing.

        Args:
            span: The answer span to decompose.

        Returns:
            List of claims extracted from the span. Returns a single claim
            wrapping the entire span if no conjunctions are found.
        """
        doc = self._nlp(span.text)
        claim_boundaries = self._find_claim_boundaries(doc)

        if not claim_boundaries:
            # No conjunctions found, return span as single claim
            return [
                Claim(
                    text=span.text,
                    char_start=span.char_start,
                    char_end=span.char_end,
                    source_span=span,
                    claim_index=0,
                )
            ]

        claims = self._extract_claims(span, doc, claim_boundaries)
        return (
            claims
            if claims
            else [
                Claim(
                    text=span.text,
                    char_start=span.char_start,
                    char_end=span.char_end,
                    source_span=span,
                    claim_index=0,
                )
            ]
        )

    def _find_claim_boundaries(self, doc: Any) -> list[tuple[int, int]]:  # noqa: ANN401
        """Find character boundaries for claim splits based on conjunctions."""
        boundaries: list[tuple[int, int]] = []

        for token in doc:
            if token.dep_ != "conj":
                continue

            boundary = self._get_boundary_for_conj(token, doc, boundaries)
            if boundary is not None:
                boundaries.append(boundary)

        return sorted(set(boundaries))

    def _get_boundary_for_conj(
        self, token: Any, doc: Any, existing: list[tuple[int, int]]  # noqa: ANN401
    ) -> tuple[int, int] | None:
        """Get split boundary for a conjoined token."""
        cc_token = self._find_cc_token(token)

        if cc_token is not None:
            return self._boundary_from_cc(cc_token, doc)
        return self._boundary_from_separator(token, doc, existing)

    def _find_cc_token(self, token: Any) -> Any | None:  # noqa: ANN401
        """Find coordinating conjunction (cc) before this token."""
        for child in token.head.children:
            if child.dep_ == "cc" and child.i < token.i:
                return child
        return None

    def _boundary_from_cc(
        self, cc_token: Any, doc: Any  # noqa: ANN401
    ) -> tuple[int, int]:
        """Create boundary from coordinating conjunction."""
        split_start = cc_token.idx
        split_end = cc_token.idx + len(cc_token.text)
        while split_end < len(doc.text) and doc.text[split_end].isspace():
            split_end += 1
        return (split_start, split_end)

    def _boundary_from_separator(
        self, token: Any, doc: Any, existing: list[tuple[int, int]]  # noqa: ANN401
    ) -> tuple[int, int] | None:
        """Create boundary from comma/semicolon separator."""
        split_start = token.idx
        for prev_token in doc:
            if prev_token.i < token.i and prev_token.text in {",", ";"}:
                if not existing or prev_token.idx > existing[-1][1]:
                    split_start = prev_token.idx
                    break

        split_end = token.idx
        while split_end < len(doc.text) and doc.text[split_end].isspace():
            split_end += 1

        return (split_start, split_end) if split_start < split_end else None

    def _extract_claims(
        self,
        span: AnswerSpan,
        doc: Any,  # noqa: ANN401
        boundaries: list[tuple[int, int]],
    ) -> list[Claim]:
        """Extract claims from span using the identified boundaries."""
        claims: list[Claim] = []
        text = span.text
        cursor = 0
        claim_index = 0

        for split_start, split_end in boundaries:
            if cursor < split_start:
                claim_text = text[cursor:split_start].strip()
                if claim_text and self._is_valid_claim(claim_text):
                    # Calculate absolute offsets
                    left_offset = len(text[cursor:split_start]) - len(
                        text[cursor:split_start].lstrip()
                    )
                    right_offset = len(text[cursor:split_start]) - len(
                        text[cursor:split_start].rstrip()
                    )

                    claims.append(
                        Claim(
                            text=claim_text,
                            char_start=span.char_start + cursor + left_offset,
                            char_end=span.char_start + split_start - right_offset,
                            source_span=span,
                            claim_index=claim_index,
                        )
                    )
                    claim_index += 1
            cursor = split_end

        # Handle remaining text after last boundary
        if cursor < len(text):
            claim_text = text[cursor:].strip()
            if claim_text and self._is_valid_claim(claim_text):
                left_strip = len(text[cursor:]) - len(text[cursor:].lstrip())
                right_strip = len(text[cursor:]) - len(text[cursor:].rstrip())

                claims.append(
                    Claim(
                        text=claim_text,
                        char_start=span.char_start + cursor + left_strip,
                        char_end=span.char_start + len(text) - right_strip,
                        source_span=span,
                        claim_index=claim_index,
                    )
                )

        return claims

    def _is_valid_claim(self, text: str) -> bool:
        """Check if extracted text forms a valid claim."""
        # Quick token count check using whitespace
        tokens = text.split()
        return len(tokens) >= self._min_claim_tokens
