from __future__ import annotations

import math
import time
from typing import Callable, Literal, Sequence, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from cite_right.core.aligner_py import SmithWatermanAligner
from cite_right.core.aligner_rust import RustSmithWatermanAligner
from cite_right.core.citation_config import CitationConfig
from cite_right.core.interfaces import Aligner, AnswerSegmenter, Segmenter, Tokenizer
from cite_right.core.results import (
    Alignment,
    AnswerSpan,
    Citation,
    EvidenceSpan,
    SourceChunk,
    SourceDocument,
    SpanCitations,
)
from cite_right.models.base import Embedder
from cite_right.models.embedding_index import EmbeddingIndex
from cite_right.text.answer_segmenter import SimpleAnswerSegmenter
from cite_right.text.passage import Passage, generate_passages
from cite_right.text.segmenter_simple import SimpleSegmenter
from cite_right.text.tokenizer import SimpleTokenizer

# Type aliases for complex types
CandidateSelection: TypeAlias = list[tuple[int, float, float]]
"""List of (candidate_index, embedding_score, lexical_score) tuples."""

LexicalScores: TypeAlias = dict[int, float]
"""Mapping from candidate index to lexical similarity score."""

IdfWeights: TypeAlias = dict[int, float]
"""Mapping from token ID to IDF weight."""


class AlignmentMetrics(BaseModel):
    """Observability metrics for the alignment pipeline.

    Attributes:
        total_time_ms: Total time spent in align_citations in milliseconds.
        num_answer_spans: Number of answer spans processed.
        num_candidates: Total number of candidate passages.
        num_alignments: Total number of alignments performed.
        embedding_time_ms: Time spent computing embeddings in milliseconds.
        alignment_time_ms: Time spent in alignment operations in milliseconds.
    """

    model_config = ConfigDict(frozen=True)

    total_time_ms: float
    num_answer_spans: int
    num_candidates: int
    num_alignments: int
    embedding_time_ms: float = 0.0
    alignment_time_ms: float = 0.0


MetricsCallback: TypeAlias = Callable[[AlignmentMetrics], None]
"""Callback function for receiving alignment metrics."""


class _NormalizedSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    source_index: int
    text: str
    base_doc_offset: int
    full_text: str | None


class _Candidate(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    global_index: int
    source: _NormalizedSource
    passage: Passage
    token_ids: list[int]
    token_spans: list[tuple[int, int]]
    token_set: frozenset[int]


class _EmbeddingCache(BaseModel):
    """Cache for batched answer span embeddings."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedder: Embedder
    answer_spans: list[AnswerSpan]
    vectors: list[list[float]] = Field(default_factory=list)
    _computed: bool = False

    def get_vector(self, span_index: int) -> list[float]:
        """Get the embedding vector for an answer span, computing batch if needed."""
        if not self._computed:
            texts = [span.text for span in self.answer_spans]
            self.vectors = self.embedder.encode(texts)
            self._computed = True
        return self.vectors[span_index]


def align_citations(
    answer: str,
    sources: Sequence[str | SourceDocument | SourceChunk],
    *,
    config: CitationConfig | None = None,
    backend: Literal["auto", "python", "rust"] = "auto",
    answer_segmenter: AnswerSegmenter | None = None,
    source_segmenter: Segmenter | None = None,
    tokenizer: Tokenizer | None = None,
    aligner: Aligner | None = None,
    embedder: Embedder | None = None,
    on_metrics: MetricsCallback | None = None,
) -> list[SpanCitations]:
    """Align answer spans to source citations.

    This is the main entry point for citation extraction. It segments the answer
    into spans (sentences by default), finds matching evidence in source documents
    using Smith-Waterman alignment, and returns character-accurate citations.

    Args:
        answer: The answer text to find citations for.
        sources: Source documents or text strings to search for evidence.
            Accepts plain strings, SourceDocument, or SourceChunk objects.
        config: Citation configuration options. See CitationConfig for details.
        backend: Alignment backend to use:

            - ``"auto"``: Use Rust if available, else Python (default).
            - ``"python"``: Force pure-Python implementation.
            - ``"rust"``: Force Rust implementation (raises if unavailable).

        answer_segmenter: Custom answer segmenter (default: SimpleAnswerSegmenter).
        source_segmenter: Custom source segmenter (default: SimpleSegmenter).
        tokenizer: Custom tokenizer (default: SimpleTokenizer).
        aligner: Custom aligner (default: SmithWatermanAligner).
        embedder: Optional embedder for semantic similarity retrieval.
            When provided, uses embedding similarity to find candidates
            in addition to lexical overlap.
        on_metrics: Optional callback to receive alignment metrics for
            observability and performance monitoring.

    Returns:
        List of SpanCitations, one per answer span. Each SpanCitations contains
        the answer segment, its citations (ranked by score), and a status
        indicating citation quality ("supported", "partial", or "unsupported").

    Examples:
        Basic usage with string sources:

        >>> from cite_right import align_citations
        >>> answer = "Revenue grew 15% in Q4."
        >>> sources = ["Annual report: Revenue grew 15% in Q4 2024."]
        >>> results = align_citations(answer, sources)
        >>> print(results[0].status)
        'supported'
        >>> print(results[0].citations[0].evidence)
        'Revenue grew 15% in Q4'

        Using SourceDocument for named sources:

        >>> from cite_right import SourceDocument, align_citations, CitationConfig
        >>> answer = "Heat pumps reduce emissions."
        >>> sources = [
        ...     SourceDocument(id="energy", text="Heat pumps reduce emissions by 50%."),
        ... ]
        >>> results = align_citations(answer, sources, config=CitationConfig(top_k=1))
        >>> citation = results[0].citations[0]
        >>> print(f"Found in {citation.source_id}: {citation.evidence!r}")
        Found in energy: 'Heat pumps reduce emissions'

        Verifying character offsets:

        >>> source_text = sources[0].text
        >>> assert source_text[citation.char_start:citation.char_end] == citation.evidence
    """
    start_time = time.perf_counter()
    embedding_time = 0.0
    alignment_time = 0.0
    num_alignments = 0

    cfg = config or CitationConfig()
    if cfg.top_k <= 0:
        if on_metrics is not None:
            on_metrics(
                AlignmentMetrics(
                    total_time_ms=0.0,
                    num_answer_spans=0,
                    num_candidates=0,
                    num_alignments=0,
                )
            )
        return []

    answer_segmenter = answer_segmenter or SimpleAnswerSegmenter()
    source_segmenter = source_segmenter or SimpleSegmenter()
    tokenizer = tokenizer or SimpleTokenizer()
    aligner = aligner or _default_aligner(cfg, backend=backend)

    normalized_sources = _normalize_sources(sources)
    answer_spans = answer_segmenter.segment(answer)
    source_passages = _build_source_passages(normalized_sources, source_segmenter, cfg)
    candidates = _build_candidates(source_passages, tokenizer)
    idf = _compute_idf(candidates)

    # Build embedding index and cache for batched answer span embeddings
    embedding_index: EmbeddingIndex | None = None
    embedding_cache: _EmbeddingCache | None = None
    if embedder is not None:
        embed_start = time.perf_counter()
        embedding_index = EmbeddingIndex.build(
            embedder, [candidate.passage.text for candidate in candidates]
        )
        # Pre-batch answer span embeddings
        embedding_cache = _EmbeddingCache(embedder=embedder, answer_spans=answer_spans)
        embedding_time += (time.perf_counter() - embed_start) * 1000

    output: list[SpanCitations] = []

    for span_index, answer_span in enumerate(answer_spans):
        answer_tokenized = tokenizer.tokenize(answer_span.text)
        answer_tokens = answer_tokenized.token_ids
        citations: list[Citation] = []

        if answer_tokens and candidates:
            answer_set = frozenset(answer_tokens)
            lexical_scores = _lexical_prefilter(answer_set, candidates, idf)

            # Get embedding vector from cache if available
            embed_start = time.perf_counter()
            query_vector: list[float] | None = None
            if embedding_cache is not None:
                query_vector = embedding_cache.get_vector(span_index)
            embedding_time += (time.perf_counter() - embed_start) * 1000

            selected = _select_candidates(
                candidates,
                lexical_scores=lexical_scores,
                embedding_index=embedding_index,
                query_vector=query_vector,
                cfg=cfg,
            )

            align_start = time.perf_counter()
            for candidate_index, embed_score, lexical_score in selected:
                candidate = candidates[candidate_index]
                alignment = aligner.align(answer_tokens, candidate.token_ids)
                num_alignments += 1
                matches = (
                    alignment.matches
                    if alignment.matches > 0
                    else max(0, alignment.score // max(1, cfg.match_score))
                )
                answer_len = len(answer_tokens)
                evidence_len = max(1, alignment.token_end - alignment.token_start)
                answer_coverage = matches / max(1, answer_len)
                evidence_coverage = matches / evidence_len
                normalized_alignment = alignment.score / max(
                    1, cfg.match_score * answer_len
                )

                use_alignment_evidence = (
                    alignment.score >= cfg.min_alignment_score
                    and alignment.token_start < alignment.token_end
                    and answer_coverage >= cfg.min_answer_coverage
                )
                use_embedding_only = (
                    cfg.allow_embedding_only
                    and embed_score >= cfg.min_embedding_similarity
                )
                if not use_alignment_evidence and not use_embedding_only:
                    continue

                if use_alignment_evidence:
                    evidence_spans = _alignment_to_evidence_spans(
                        candidate, alignment, cfg
                    )
                    if evidence_spans is None:
                        continue
                    abs_start = min(span.char_start for span in evidence_spans)
                    abs_end = max(span.char_end for span in evidence_spans)
                    evidence = _slice_source_text(candidate.source, abs_start, abs_end)
                else:
                    abs_start = (
                        candidate.source.base_doc_offset
                        + candidate.passage.doc_char_start
                    )
                    abs_end = (
                        candidate.source.base_doc_offset
                        + candidate.passage.doc_char_end
                    )
                    evidence = _slice_source_text(candidate.source, abs_start, abs_end)
                    evidence_spans = [
                        EvidenceSpan(
                            char_start=abs_start,
                            char_end=abs_end,
                            evidence=evidence,
                        )
                    ]

                final_score = (
                    cfg.weights.alignment * normalized_alignment
                    + cfg.weights.answer_coverage * answer_coverage
                    + cfg.weights.evidence_coverage * evidence_coverage
                    + cfg.weights.lexical * lexical_score
                    + cfg.weights.embedding * max(0.0, embed_score)
                )
                if final_score < cfg.min_final_score:
                    continue

                citations.append(
                    Citation(
                        score=final_score,
                        source_id=candidate.source.source_id,
                        source_index=candidate.source.source_index,
                        candidate_index=candidate.global_index,
                        char_start=abs_start,
                        char_end=abs_end,
                        evidence=evidence,
                        evidence_spans=evidence_spans,
                        components={
                            "embedding_only": 0.0 if use_alignment_evidence else 1.0,
                            "alignment_score": float(alignment.score),
                            "normalized_alignment": float(normalized_alignment),
                            "matches": float(matches),
                            "answer_coverage": float(answer_coverage),
                            "evidence_coverage": float(evidence_coverage),
                            "lexical_score": float(lexical_score),
                            "embedding_score": float(embed_score),
                            "num_evidence_spans": float(len(evidence_spans)),
                            "evidence_chars_total": float(
                                sum(
                                    span.char_end - span.char_start
                                    for span in evidence_spans
                                )
                            ),
                            "passage_char_start": float(
                                candidate.passage.doc_char_start
                            ),
                            "passage_char_end": float(candidate.passage.doc_char_end),
                        },
                    )
                )
            alignment_time += (time.perf_counter() - align_start) * 1000

        citations = _rank_and_limit_citations(citations, cfg)
        status = _span_status(citations, cfg)
        output.append(
            SpanCitations(answer_span=answer_span, citations=citations, status=status)
        )

    if on_metrics is not None:
        total_time = (time.perf_counter() - start_time) * 1000
        on_metrics(
            AlignmentMetrics(
                total_time_ms=total_time,
                num_answer_spans=len(answer_spans),
                num_candidates=len(candidates),
                num_alignments=num_alignments,
                embedding_time_ms=embedding_time,
                alignment_time_ms=alignment_time,
            )
        )

    return output


def _default_aligner(cfg: CitationConfig, *, backend: str) -> Aligner:
    if backend == "python":
        return SmithWatermanAligner(
            match_score=cfg.match_score,
            mismatch_score=cfg.mismatch_score,
            gap_score=cfg.gap_score,
            return_match_blocks=cfg.multi_span_evidence,
        )
    if backend == "rust":
        return RustSmithWatermanAligner(
            match_score=cfg.match_score,
            mismatch_score=cfg.mismatch_score,
            gap_score=cfg.gap_score,
            return_match_blocks=cfg.multi_span_evidence,
        )
    if backend != "auto":
        raise ValueError(f"Unknown backend: {backend}")
    try:
        return RustSmithWatermanAligner(
            match_score=cfg.match_score,
            mismatch_score=cfg.mismatch_score,
            gap_score=cfg.gap_score,
            return_match_blocks=cfg.multi_span_evidence,
        )
    except RuntimeError:
        return SmithWatermanAligner(
            match_score=cfg.match_score,
            mismatch_score=cfg.mismatch_score,
            gap_score=cfg.gap_score,
            return_match_blocks=cfg.multi_span_evidence,
        )


def _normalize_sources(
    sources: Sequence[str | SourceDocument | SourceChunk],
) -> list[_NormalizedSource]:
    normalized: list[_NormalizedSource] = []
    for index, item in enumerate(sources):
        if isinstance(item, str):
            normalized.append(
                _NormalizedSource(
                    source_id=str(index),
                    source_index=index,
                    text=item,
                    base_doc_offset=0,
                    full_text=item,
                )
            )
        elif isinstance(item, SourceDocument):
            normalized.append(
                _NormalizedSource(
                    source_id=item.id,
                    source_index=index,
                    text=item.text,
                    base_doc_offset=0,
                    full_text=item.text,
                )
            )
        else:
            source_index = item.source_index if item.source_index is not None else index
            full_text = item.document_text
            normalized.append(
                _NormalizedSource(
                    source_id=item.source_id,
                    source_index=source_index,
                    text=item.text,
                    base_doc_offset=item.doc_char_start,
                    full_text=full_text,
                )
            )
    return normalized


def _build_source_passages(
    sources: Sequence[_NormalizedSource],
    segmenter: Segmenter,
    cfg: CitationConfig,
) -> list[tuple[_NormalizedSource, list[Passage]]]:
    output: list[tuple[_NormalizedSource, list[Passage]]] = []
    for source in sources:
        passages = generate_passages(
            source.text,
            segmenter=segmenter,
            window_size_sentences=cfg.window_size_sentences,
            window_stride_sentences=cfg.window_stride_sentences,
        )
        output.append((source, passages))
    return output


def _build_candidates(
    source_passages: Sequence[tuple[_NormalizedSource, list[Passage]]],
    tokenizer: Tokenizer,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    global_index = 0
    for source, passages in source_passages:
        for passage in passages:
            tokenized = tokenizer.tokenize(passage.text)
            candidates.append(
                _Candidate(
                    global_index=global_index,
                    source=source,
                    passage=passage,
                    token_ids=tokenized.token_ids,
                    token_spans=tokenized.token_spans,
                    token_set=frozenset(tokenized.token_ids),
                )
            )
            global_index += 1
    return candidates


def _compute_idf(candidates: Sequence[_Candidate]) -> IdfWeights:
    df: dict[int, int] = {}
    for candidate in candidates:
        for token_id in candidate.token_set:
            df[token_id] = df.get(token_id, 0) + 1
    n = len(candidates)
    return {
        token_id: math.log((n + 1) / (count + 1)) + 1.0
        for token_id, count in df.items()
    }


def _lexical_prefilter(
    answer_set: frozenset[int],
    candidates: Sequence[_Candidate],
    idf: IdfWeights,
) -> LexicalScores:
    if not answer_set:
        return {}
    denom = sum(idf.get(token_id, 1.0) for token_id in answer_set)
    if denom <= 0.0:
        return {}

    scores: LexicalScores = {}
    for idx, candidate in enumerate(candidates):
        overlap = answer_set & candidate.token_set
        if not overlap:
            continue
        numer = sum(idf.get(token_id, 1.0) for token_id in overlap)
        scores[idx] = numer / denom
    return scores


def _select_candidates(
    candidates: Sequence[_Candidate],
    *,
    lexical_scores: LexicalScores,
    embedding_index: EmbeddingIndex | None,
    query_vector: list[float] | None,
    cfg: CitationConfig,
) -> CandidateSelection:
    selected: dict[int, tuple[float, float]] = {}

    if cfg.max_candidates_lexical > 0 and lexical_scores:
        ordered = sorted(
            lexical_scores.items(),
            key=lambda item: (
                -item[1],
                candidates[item[0]].source.source_index,
                item[0],
            ),
        )
        for idx, score in ordered[: cfg.max_candidates_lexical]:
            selected[idx] = (0.0, score)

    if (
        cfg.max_candidates_embedding > 0
        and query_vector is not None
        and embedding_index is not None
    ):
        for idx, score in embedding_index.top_k(
            query_vector, cfg.max_candidates_embedding
        ):
            prev = selected.get(idx)
            lexical_score = 0.0 if prev is None else prev[1]
            selected[idx] = (score, lexical_score)

    ordered = sorted(
        selected.items(),
        key=lambda item: (
            -max(item[1][0], item[1][1]),
            candidates[item[0]].source.source_index,
            item[0],
        ),
    )
    if cfg.max_candidates_total > 0:
        ordered = ordered[: cfg.max_candidates_total]

    return [(idx, values[0], values[1]) for idx, values in ordered]


def _slice_source_text(source: _NormalizedSource, abs_start: int, abs_end: int) -> str:
    if source.full_text is not None:
        return source.full_text[abs_start:abs_end]
    local_start = abs_start - source.base_doc_offset
    local_end = abs_end - source.base_doc_offset
    return source.text[local_start:local_end]


def _alignment_to_evidence_spans(
    candidate: _Candidate,
    alignment: Alignment,
    cfg: CitationConfig,
) -> list[EvidenceSpan] | None:
    """Convert an alignment into one or more evidence spans.

    The returned spans are absolute offsets into the source document (0-based,
    half-open) and each `EvidenceSpan.evidence` slices back exactly from the
    source text.

    Args:
        candidate: Candidate passage/tokenization context.
        alignment: Alignment output for the candidate.
        cfg: Citation configuration.

    Returns:
        A non-empty list of `EvidenceSpan` on success; otherwise None.
    """
    spans: list[EvidenceSpan] = []

    if cfg.multi_span_evidence and alignment.match_blocks:
        for token_start, token_end in alignment.match_blocks:
            char_span = _token_span_to_char_span(
                candidate.token_spans,
                token_start,
                token_end,
            )
            if char_span is None:
                continue
            seg_char_start, seg_char_end = char_span

            abs_start = (
                candidate.source.base_doc_offset
                + candidate.passage.doc_char_start
                + seg_char_start
            )
            abs_end = (
                candidate.source.base_doc_offset
                + candidate.passage.doc_char_start
                + seg_char_end
            )
            if abs_start >= abs_end:
                continue

            spans.append(
                EvidenceSpan(
                    char_start=abs_start,
                    char_end=abs_end,
                    evidence=_slice_source_text(candidate.source, abs_start, abs_end),
                )
            )

        spans = _merge_evidence_spans(
            candidate.source, spans, merge_gap_chars=cfg.multi_span_merge_gap_chars
        )

        if cfg.multi_span_max_spans > 0 and len(spans) > cfg.multi_span_max_spans:
            spans = []

    if not spans:
        char_span = _token_span_to_char_span(
            candidate.token_spans,
            alignment.token_start,
            alignment.token_end,
        )
        if char_span is None:
            return None
        seg_char_start, seg_char_end = char_span

        abs_start = (
            candidate.source.base_doc_offset
            + candidate.passage.doc_char_start
            + seg_char_start
        )
        abs_end = (
            candidate.source.base_doc_offset
            + candidate.passage.doc_char_start
            + seg_char_end
        )
        if abs_start >= abs_end:
            return None

        spans = [
            EvidenceSpan(
                char_start=abs_start,
                char_end=abs_end,
                evidence=_slice_source_text(candidate.source, abs_start, abs_end),
            )
        ]

    return spans


def _merge_evidence_spans(
    source: _NormalizedSource,
    spans: list[EvidenceSpan],
    *,
    merge_gap_chars: int,
) -> list[EvidenceSpan]:
    """Merge evidence spans that are close together in the source text.

    Args:
        source: Source context used to re-slice evidence after merging.
        spans: Evidence spans (absolute offsets).
        merge_gap_chars: Merge spans when the character gap between them is
            <= this value. Values <= 0 disable merging.

    Returns:
        Merged spans sorted by `(char_start, char_end)`.
    """
    if not spans:
        return []

    ordered = sorted(spans, key=lambda span: (span.char_start, span.char_end))
    if merge_gap_chars <= 0:
        return ordered

    merged: list[EvidenceSpan] = [ordered[0]]
    for span in ordered[1:]:
        prev = merged[-1]
        gap = span.char_start - prev.char_end
        if gap <= merge_gap_chars:
            abs_start = prev.char_start
            abs_end = max(prev.char_end, span.char_end)
            merged[-1] = EvidenceSpan(
                char_start=abs_start,
                char_end=abs_end,
                evidence=_slice_source_text(source, abs_start, abs_end),
            )
            continue
        merged.append(span)

    return merged


def _token_span_to_char_span(
    token_spans: list[tuple[int, int]], token_start: int, token_end: int
) -> tuple[int, int] | None:
    if token_start < 0 or token_end > len(token_spans) or token_start >= token_end:
        return None
    span_start = token_spans[token_start][0]
    span_end = token_spans[token_end - 1][1]
    return span_start, span_end


def _rank_and_limit_citations(
    citations: list[Citation], cfg: CitationConfig
) -> list[Citation]:
    citations.sort(key=lambda c: _citation_sort_key(c, cfg))

    seen: set[tuple[str, tuple[tuple[int, int], ...]]] = set()
    per_source: dict[str, int] = {}
    output: list[Citation] = []

    for citation in citations:
        spans = (
            tuple((span.char_start, span.char_end) for span in citation.evidence_spans)
            if citation.evidence_spans
            else ((citation.char_start, citation.char_end),)
        )
        key = (citation.source_id, spans)
        if key in seen:
            continue
        seen.add(key)
        per_source.setdefault(citation.source_id, 0)
        if per_source[citation.source_id] >= cfg.max_citations_per_source:
            continue
        per_source[citation.source_id] += 1
        output.append(citation)
        if len(output) >= cfg.top_k:
            break

    return output


def _citation_sort_key(
    citation: Citation, cfg: CitationConfig
) -> tuple[float, int, int, int, int]:
    length = citation.char_end - citation.char_start
    if cfg.prefer_source_order:
        return (
            -citation.score,
            citation.source_index,
            citation.char_start,
            -length,
            citation.candidate_index,
        )
    return (
        -citation.score,
        citation.char_start,
        -length,
        citation.source_index,
        citation.candidate_index,
    )


def _span_status(
    citations: Sequence[Citation],
    cfg: CitationConfig,
) -> Literal["supported", "partial", "unsupported"]:
    if not citations:
        return "unsupported"
    best = citations[0]
    coverage = float(best.components.get("answer_coverage", 0.0))
    if coverage >= cfg.supported_answer_coverage:
        return "supported"
    if cfg.allow_embedding_only:
        embed_score = float(best.components.get("embedding_score", 0.0))
        if embed_score >= cfg.supported_embedding_similarity:
            return "supported"
    return "partial"
