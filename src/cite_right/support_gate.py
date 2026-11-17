"""Decide support from the best localized citation."""

from collections.abc import Sequence
from typing import Literal

from cite_right.contradiction import check_contradiction
from cite_right.core.citation_config import CitationConfig
from cite_right.core.prepared_corpus import Candidate
from cite_right.core.results import Citation


def _contradiction_context(
    citation: Citation,
    candidates: Sequence[Candidate] | None,
) -> str:
    """Prefer the candidate passage over truncated Smith-Waterman evidence.

    Leftover n-grams (issue #48) attach to the wrong slot when alignment
    truncates evidence and hides the contradicting remainder of the passage.
    """
    if candidates:
        for candidate in candidates:
            if candidate.global_index == citation.candidate_index:
                passage = candidate.passage.text
                if passage:
                    return passage
    return citation.evidence


def span_status(
    citations: Sequence[Citation],
    cfg: CitationConfig,
    answer_text: str | None = None,
    candidates: Sequence[Candidate] | None = None,
) -> Literal["supported", "partial", "unsupported"]:
    if not citations:
        return "unsupported"
    best = citations[0]
    coverage = float(best.components.get("answer_coverage", 0.0))

    # Check for contradictions if answer text is provided.
    # Use the candidate passage so leftover tokens beyond truncated evidence
    # (e.g. "BC", "of which came in the first half") are visible.
    if answer_text is not None and check_contradiction(
        answer_text, _contradiction_context(best, candidates)
    ):
        # Downgrade to partial (not unsupported) if contradiction detected
        # because we have evidence, it just contradicts the claim
        return "partial"

    if coverage >= cfg.supported_answer_coverage:
        return "supported"
    return "partial"
