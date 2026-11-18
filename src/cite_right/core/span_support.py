"""Classify ranked citations and attach optional retrieval diagnostics."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from math import fsum
from typing import TYPE_CHECKING

from cite_right.contradiction import iter_contradiction_checks
from cite_right.core.citation_config import CitationConfig
from cite_right.core.prepared_corpus import Candidate
from cite_right.core.results import (
    AnswerSpan,
    Citation,
    PossibleSourceMatch,
    RetrievalSupport,
    SpanCitations,
    SpanStatus,
    StatusReason,
    TokenizedText,
)
from cite_right.text.content_coverage import STOPWORDS
from cite_right.text.tokenizer import normalize_apostrophes

if TYPE_CHECKING:
    from cite_right._core import PreparedCorpus as RustPreparedCorpus
    from cite_right.core.prepared_corpus import IdfWeights
    from cite_right.text.term_aliases import TermAliases

_STATUS_BY_REASON: dict[StatusReason, SpanStatus] = {
    "supported": "supported",
    "below_coverage": "partial",
    "contradiction": "partial",
    "no_candidate": "unsupported",
}
_ANCHOR_STOPWORDS = STOPWORDS | {"whether"}
_MARKDOWN_ITEM_PREFIX = re.compile(
    r"^[ \t]*(?:>[ \t]*|(?:[-+*]|\d+[.)]|\[[xX ]\])[ \t]+)+",
    re.MULTILINE,
)
_NON_CLAIM_SPAN = re.compile(
    r"""
    \#{1,6}[ \t]+
    | [^\n]+\n[ \t]*(?:=+|-{2,})[ \t]*$
    | (?:document\s+)?(?:revision|version|(?:last\s+)?updated|last\s+revised|author|date|metadata)\s*:
    | last\s+(?:revised|updated)\s+(?:on\s+)?(?:[a-z]+\s+)?\d[\d\s,./-]*$
    | (?:revision|document)\s+metadata\s+follows[.!?]*$
    | (?:here['’]s|(?:here|below|the\s+following)\s+(?:is|are))\s+
      (?:(?:a|an|the|brief|quick|short|general)\s+)*
      (?:overview|summary|introduction|outline|recap)(?:\s+of\b[^:;\n]*)?[.!?:;]*$
    | (?:please\s+)?(?:always\s+)?(?:carefully\s+)?(?:follow|read|consult|check)\s+
      (?:(?:all|the|general|safety)\s+)*(?:instructions|directions|guidelines|manual)
      (?:\s+(?:carefully|for\s+details|before\s+proceeding))?[.!?:;]*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def build_span_citations(
    *,
    answer_span: AnswerSpan,
    answer_tokens: TokenizedText,
    reason: StatusReason,
    checks: list[str] | None,
    citations: list[Citation],
    retrieval_support: list[RetrievalSupport],
    candidates: Sequence[Candidate],
    candidate_ranking: Sequence[tuple[int, float, float]],
    idf: IdfWeights,
    rust_corpus: RustPreparedCorpus | None,
    config: CitationConfig,
    uses_alias_retrieval: bool,
) -> SpanCitations:
    possible_source = None
    if config.report_possible_source and reason != "supported" and candidate_ranking:
        index, embedding_score, lexical_score = candidate_ranking[0]
        possible_source = _possible_source_match(
            answer_tokens,
            candidates[index],
            rust_corpus,
            embedding_score,
            lexical_score,
            idf,
            config,
            uses_alias_retrieval=uses_alias_retrieval,
        )
    return SpanCitations(
        answer_span=answer_span,
        citations=citations,
        retrieval_support=retrieval_support,
        status=_STATUS_BY_REASON[reason],
        status_reason=reason if config.report_possible_source else None,
        contradiction_checks=checks,
        possible_source=possible_source,
    )


def citation_status_reason(
    answer_text: str,
    citation: Citation | None,
    candidates: Sequence[Candidate],
    config: CitationConfig,
    term_aliases: TermAliases | None,
    *,
    report_checks: bool = False,
) -> tuple[StatusReason, list[str] | None]:
    """Classify a citation using its full passage and the configured support rules."""
    checks: list[str] | None = [] if report_checks else None
    if citation is None:
        return "no_candidate", checks

    triggered = iter_contradiction_checks(
        answer_text,
        candidates[citation.candidate_index].passage.text or citation.evidence,
        entity_aliases=term_aliases.canonical_word if term_aliases else None,
        normalize_negation=normalize_apostrophes if term_aliases else None,
        normalize_temporal_clause=term_aliases.canonicalize_clause
        if term_aliases
        else None,
    )
    if checks is None:
        contradiction = next(triggered, None) is not None
    else:
        checks.extend(triggered)
        contradiction = bool(checks)
    if contradiction:
        return "contradiction", checks
    if float(
        citation.components.get("answer_coverage", 0.0)
    ) >= config.supported_answer_coverage and (
        not config.term_aliases or citation.components.get("content_coverage") == 1.0
    ):
        return "supported", checks
    return "below_coverage", checks


def _possible_source_match(
    answer_tokens: TokenizedText,
    candidate: Candidate,
    rust_corpus: RustPreparedCorpus | None,
    embedding_score: float,
    lexical_score: float,
    idf: IdfWeights,
    config: CitationConfig,
    *,
    uses_alias_retrieval: bool,
) -> PossibleSourceMatch | None:
    if _is_short_or_boilerplate(answer_tokens):
        return None
    passage_tokens = (
        candidate.token_ids
        if rust_corpus is None
        else rust_corpus.get_candidate_tokens([candidate.global_index])[0]
    )
    if not _has_content_anchors(answer_tokens, passage_tokens):
        return None

    if lexical_score > 0.0 and not uses_alias_retrieval:
        lexical_score = _hint_lexical_score(
            answer_tokens.token_ids, passage_tokens, idf
        )
    lexical_match = lexical_score > 0.0 and lexical_score >= config.min_answer_coverage
    embedding_match = (
        embedding_score > 0.0 and embedding_score >= config.min_embedding_similarity
    )
    if not (lexical_match or embedding_match):
        return None
    return _source_match(candidate, max(embedding_score, lexical_score))


def _hint_lexical_score(
    answer_tokens: Sequence[int], passage_tokens: Sequence[int], idf: IdfWeights
) -> float:
    # A reused tokenizer can assign different IDs to unseen words, changing set order.
    answer_set = set(answer_tokens)
    overlap = answer_set.intersection(passage_tokens)
    return fsum(idf.get(token, 1.0) for token in overlap) / fsum(
        idf.get(token, 1.0) for token in answer_set
    )


def _source_match(candidate: Candidate, score: float) -> PossibleSourceMatch | None:
    source = candidate.source
    char_start = source.base_doc_offset + candidate.passage.doc_char_start
    char_end = source.base_doc_offset + candidate.passage.doc_char_end
    if not 0 <= source.base_doc_offset <= char_start < char_end:
        return None
    if (
        source.declared_doc_char_end is not None
        and char_end > source.declared_doc_char_end
    ):
        return None
    return PossibleSourceMatch(
        source_id=source.source_id,
        source_index=source.source_index,
        char_start=char_start,
        char_end=char_end,
        score=score,
    )


def _is_short_or_boilerplate(answer_tokens: TokenizedText) -> bool:
    text = unicodedata.normalize("NFKC", answer_tokens.text).strip()
    text = _MARKDOWN_ITEM_PREFIX.sub("", text).strip("*_~").strip()
    return (
        len(answer_tokens.token_ids) <= 3
        or ((text.istitle() or text.isupper()) and not text.endswith((".", "!", "?")))
        or _NON_CLAIM_SPAN.match(text) is not None
    )


def _has_content_anchors(
    answer_tokens: TokenizedText, passage_tokens: Sequence[int]
) -> bool:
    """Require two distinct shared word tokens beyond function words."""
    shared = set(passage_tokens).intersection(answer_tokens.token_ids)
    anchors: set[int] = set()
    for token, (start, end) in zip(
        answer_tokens.token_ids, answer_tokens.token_spans, strict=True
    ):
        if token not in shared:
            continue
        word = (
            unicodedata.normalize("NFKC", answer_tokens.text[start:end])
            .casefold()
            .strip()
        )
        if word not in _ANCHOR_STOPWORDS and any(char.isalpha() for char in word):
            anchors.add(token)
            if len(anchors) >= 2:
                return True
    return False
