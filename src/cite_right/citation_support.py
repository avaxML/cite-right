"""Decide whether a citation is strong enough to count as supported evidence."""

from __future__ import annotations

from typing import Callable, Literal, Sequence

from cite_right.contradiction import check_contradiction
from cite_right.core.citation_config import CitationConfig
from cite_right.core.prepared_corpus import Candidate
from cite_right.core.results import Citation, RetrievalSupport


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


SupportBuilder = Callable[[Candidate, float, float, float], RetrievalSupport]


def demote_unsupported_secondaries(
    citations: list[Citation],
    retrieval_support: list[RetrievalSupport],
    *,
    status: Literal["supported", "partial", "unsupported"],
    cfg: CitationConfig,
    answer_text: str | None,
    candidates: Sequence[Candidate] | None,
    build_support: SupportBuilder,
) -> tuple[list[Citation], list[RetrievalSupport]]:
    """Move secondary citations that are not supported on their own out of a span.

    Only ``supported`` spans are filtered, and the best citation always stays.
    Each demoted citation becomes a ``RetrievalSupport`` for its passage unless
    that candidate is already present. Citation order is preserved.
    """
    if status != "supported":
        return citations, retrieval_support

    kept = citations[:1]
    demoted: list[Citation] = []
    for citation in citations[1:]:
        if citation_is_supported(citation, cfg, answer_text, candidates):
            kept.append(citation)
        else:
            demoted.append(citation)
    unavailable = {entry.candidate_index for entry in retrieval_support}
    unavailable.update(citation.candidate_index for citation in kept)
    by_index = {candidate.global_index: candidate for candidate in candidates or ()}
    support = list(retrieval_support)
    for citation in demoted:
        candidate = by_index.get(citation.candidate_index)
        if candidate is None or citation.candidate_index in unavailable:
            continue
        unavailable.add(citation.candidate_index)
        support.append(
            build_support(
                candidate,
                citation.score,
                float(citation.components.get("lexical_score", 0.0)),
                float(citation.components.get("embedding_score", 0.0)),
            )
        )
    return kept, support
