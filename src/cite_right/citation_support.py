"""Decide whether a citation is strong enough to count as supported evidence."""

from __future__ import annotations

from typing import Sequence

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


def citation_is_supported(
    citation: Citation,
    cfg: CitationConfig,
    answer_text: str | None = None,
    candidates: Sequence[Candidate] | None = None,
) -> bool:
    """Return True when ``citation`` would make a span ``supported`` on its own.

    A contradiction between the answer and the candidate passage disqualifies
    the citation (the span is downgraded to ``partial``). Otherwise the citation
    must cover at least ``cfg.supported_answer_coverage`` of the answer tokens.
    """
    if answer_text is not None and check_contradiction(
        answer_text, _contradiction_context(citation, candidates)
    ):
        return False
    coverage = float(citation.components.get("answer_coverage", 0.0))
    return coverage >= cfg.supported_answer_coverage
