"""Decide support from the best localized citation."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from itertools import pairwise
from typing import Literal

from cite_right.contradiction import check_contradiction
from cite_right.core.citation_config import CitationConfig
from cite_right.core.prepared_corpus import Candidate
from cite_right.core.results import Citation, EvidenceSpan
from cite_right.models.embedding_index import EmbeddingIndex

_Status = Literal["supported", "partial", "unsupported"]


def paraphrase_admission_floor(
    config: CitationConfig, *, has_embedder: bool
) -> float | None:
    if not config.paraphrase_support:
        return None
    if has_embedder:
        return config.paraphrase_content_coverage
    return config.paraphrase_lexical_content_coverage


def span_status(
    citations: list[Citation],
    cfg: CitationConfig,
    answer_text: str | None = None,
    candidates: Sequence[Candidate] | None = None,
    *,
    answer_tokens: Sequence[int],
    stopword_ids: frozenset[int],
    embedding_index: EmbeddingIndex | None = None,
    query_vector: list[float] | None = None,
) -> _Status:
    if not citations:
        return "unsupported"

    best = citations[0]
    candidate = _candidate_for(best, candidates)
    passage = candidate.passage.text if candidate is not None else best.evidence
    contradiction = answer_text is not None and check_contradiction(
        answer_text, passage or best.evidence
    )
    extractive_support = not contradiction and _has_extractive_support(best, cfg)
    if not cfg.paraphrase_support:
        return "supported" if extractive_support else "partial"

    similarity = _best_similarity(best, embedding_index, query_vector)
    promoted = _paraphrase_is_supported(
        best,
        cfg,
        answer_text=answer_text,
        candidate=candidate,
        content_count=len(set(answer_tokens) - stopword_ids),
        contradiction=contradiction,
        extractive_support=extractive_support,
        has_embedder=embedding_index is not None,
        similarity=similarity,
    )
    citations[0] = _add_gate_components(
        best, candidate, promoted, embedding_index is not None, similarity
    )
    return "supported" if extractive_support or promoted else "partial"


def _has_extractive_support(citation: Citation, cfg: CitationConfig) -> bool:
    answer = citation.components.get("answer_coverage", 0.0)
    if not cfg.paraphrase_support:
        return answer >= cfg.supported_answer_coverage
    content = citation.components.get("content_coverage", 0.0)
    return (
        answer >= cfg.supported_answer_coverage
        and max(answer, content) >= cfg.min_answer_coverage
    )


def _candidate_for(
    citation: Citation, candidates: Sequence[Candidate] | None
) -> Candidate | None:
    if not candidates:
        return None
    index = citation.candidate_index
    if 0 <= index < len(candidates) and candidates[index].global_index == index:
        return candidates[index]
    return next((item for item in candidates if item.global_index == index), None)


def _best_similarity(
    citation: Citation,
    index: EmbeddingIndex | None,
    query_vector: list[float] | None,
) -> float | None:
    if index is None or query_vector is None:
        return None
    return index.similarity(citation.candidate_index, query_vector)


def _paraphrase_is_supported(
    citation: Citation,
    cfg: CitationConfig,
    *,
    answer_text: str | None,
    candidate: Candidate | None,
    content_count: int,
    contradiction: bool,
    extractive_support: bool,
    has_embedder: bool,
    similarity: float | None,
) -> bool:
    if contradiction or extractive_support or candidate is None or not answer_text:
        return False
    if not _passes_paraphrase_floors(
        cfg,
        content_count,
        citation.components.get("content_coverage", 0.0),
        has_embedder=has_embedder,
        similarity=similarity,
    ):
        return False

    passage = candidate.passage.text
    return _same_reordered_claim(answer_text, passage) and _has_exact_passage(
        candidate, passage
    )


def _passes_paraphrase_floors(
    cfg: CitationConfig,
    content_count: int,
    content_coverage: float,
    *,
    has_embedder: bool,
    similarity: float | None,
) -> bool:
    floor = paraphrase_admission_floor(cfg, has_embedder=has_embedder)
    if floor is None or content_count < 3 or content_coverage < floor:
        return False
    return not has_embedder or (
        similarity is not None and similarity >= cfg.paraphrase_embedding_similarity
    )


def _has_exact_passage(candidate: Candidate, passage: str) -> bool:
    full_text = candidate.source.full_text
    if full_text is None:
        return True
    start = candidate.source.base_doc_offset + candidate.passage.doc_char_start
    end = candidate.source.base_doc_offset + candidate.passage.doc_char_end
    return full_text[start:end] == passage


def _add_gate_components(
    citation: Citation,
    candidate: Candidate | None,
    promoted: bool,
    has_embedder: bool,
    similarity: float | None,
) -> Citation:
    components = dict(citation.components)
    components["content_gate"] = float(promoted)
    if has_embedder:
        components["embedding_similarity"] = 0.0 if similarity is None else similarity

    if not promoted or candidate is None:
        return citation.model_copy(update={"components": components})

    start = candidate.source.base_doc_offset + candidate.passage.doc_char_start
    end = candidate.source.base_doc_offset + candidate.passage.doc_char_end
    text = candidate.passage.text
    components["num_evidence_spans"] = 1.0
    components["evidence_chars_total"] = float(end - start)
    return citation.model_copy(
        update={
            "components": components,
            "source_id": candidate.source.source_id,
            "source_index": candidate.source.source_index,
            "char_start": start,
            "char_end": end,
            "evidence": text,
            "evidence_spans": [
                EvidenceSpan(char_start=start, char_end=end, evidence=text)
            ],
        }
    )


_RELATION_MARKERS = frozenset({"after", "before", "once", "when", "while"})
_CLAUSE_JOINERS = _RELATION_MARKERS | {
    "and",
    "because",
    "but",
    "if",
    "or",
    "otherwise",
    "plus",
    "so",
    "that",
    "then",
    "unless",
    "until",
    "where",
    "whether",
    "which",
    "who",
    "whose",
}
_CLAUSE_WORD = re.compile(
    r"[^\W_]+(?:['’][^\W_]+)*"
    r"|[A-Z0-9]+(?:[-‐‑][A-Z0-9]+)+"
    r"|[+-]?\d+[%°]?"
)
_LABEL_HYPHEN = re.compile("[-‐‑]")
_ARTICLES = frozenset({"a", "an", "the"})
_PREPOSITIONS = frozenset(
    {
        "above",
        "around",
        "as",
        "at",
        "below",
        "beside",
        "by",
        "for",
        "from",
        "in",
        "inside",
        "into",
        "near",
        "of",
        "on",
        "over",
        "through",
        "to",
        "under",
        "with",
    }
)
_DIRECT_ACTIONS = frozenset(
    {
        "attach",
        "close",
        "disconnect",
        "dry",
        "fasten",
        "fold",
        "insert",
        "install",
        "latch",
        "lift",
        "lock",
        "loosen",
        "lower",
        "open",
        "plug",
        "press",
        "raise",
        "release",
        "remove",
        "replace",
        "rinse",
        "rotate",
        "secure",
        "set",
        "slide",
        "switch",
        "tighten",
        "turn",
        "unplug",
        "wash",
        "wipe",
    }
)


def _paraphrase_clause_key(text: str) -> tuple[str, str, str] | None:
    """Canonicalize a single claim and its movable time clause."""
    body = _normalize_clause_text(text)
    if body.endswith("?"):
        return None
    body = body.rstrip(".!")
    parts = _split_time_clause(body.split())
    if parts is None:
        return None

    marker, condition, claim = parts
    if not (_is_simple_clause(condition) and _is_simple_clause(claim)):
        return None

    if marker == "once" and condition.endswith(" is complete"):
        marker = "after"
        condition = condition.removesuffix(" is complete")
    action, _, target = claim.partition(" ")
    action = action.casefold()
    if action not in _DIRECT_ACTIONS:
        return None
    return marker, condition, f"{action} {target}"


def _normalize_clause_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    decimal_text = "".join(str(unicodedata.decimal(char, char)) for char in normalized)
    return " ".join(decimal_text.split())


def _split_time_clause(words: list[str]) -> tuple[str, str, str] | None:
    markers = [
        index
        for index, word in enumerate(words)
        if word.casefold() in _RELATION_MARKERS
    ]
    if len(markers) != 1:
        return None

    index = markers[0]
    marker = words[index].casefold()
    if index:
        return marker, " ".join(words[index + 1 :]), " ".join(words[:index])

    condition, separator, claim = " ".join(words[1:]).partition(",")
    if not separator:
        return None
    return marker, condition.strip(), claim.strip()


def _is_simple_clause(text: str) -> bool:
    words = text.split()
    return (
        len(words) >= 2
        and all(
            word.casefold() not in _CLAUSE_JOINERS and _CLAUSE_WORD.fullmatch(word)
            for word in words
        )
        and not _has_second_action(words)
    )


def _has_second_action(words: list[str]) -> bool:
    folded = [word.casefold() for word in words]
    if _has_action_complement(folded):
        return True
    if any(
        _starts_separate_phrase(previous, word)
        for previous, word in pairwise(folded[1:])
    ):
        return True
    return any(
        part in _DIRECT_ACTIONS
        for word in folded
        for part in _LABEL_HYPHEN.split(word)[1:]
    )


def _starts_separate_phrase(previous: str, word: str) -> bool:
    if word in _ARTICLES:
        return previous not in _PREPOSITIONS
    return word in _DIRECT_ACTIONS and previous not in _ARTICLES


def _has_action_complement(words: list[str]) -> bool:
    return any(
        current == "to"
        and following not in _ARTICLES
        and not following.lstrip("+-")[:1].isdecimal()
        for current, following in pairwise(words)
    )


def _same_reordered_claim(answer: str, passage: str) -> bool:
    answer_key = _paraphrase_clause_key(answer)
    return answer_key is not None and answer_key == _paraphrase_clause_key(passage)
