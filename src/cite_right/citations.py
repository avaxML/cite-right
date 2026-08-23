from __future__ import annotations

import time
from collections import Counter
from typing import Iterable, Literal, Sequence, TypeAlias

from pydantic import BaseModel, ConfigDict

try:
    from cite_right import _core

    HAS_RUST_CORE = True
except ImportError:
    HAS_RUST_CORE = False
    _core = None  # type: ignore[assignment]

from cite_right.contradiction import check_contradiction
from cite_right.core.aligner_py import SmithWatermanAligner
from cite_right.core.aligner_rust import RustSmithWatermanAligner
from cite_right.core.citation_config import CitationConfig
from cite_right.core.interfaces import Aligner, AnswerSegmenter, Segmenter, Tokenizer
from cite_right.core.prepared_corpus import (
    AlignmentMetrics,
    Candidate,
    EmbeddingCache,
    IdfWeights,
    LexicalScores,
    MetricsCallback,
    NormalizedSource,
    PreparedCitationCorpus,
    report_empty_metrics,
)
from cite_right.core.results import (
    Alignment,
    AnswerSpan,
    Citation,
    EvidenceSpan,
    RetrievalSupport,
    SourceChunk,
    SourceDocument,
    SpanCitations,
)
from cite_right.models.base import Embedder
from cite_right.models.embedding_index import EmbeddingIndex

CandidateSelection: TypeAlias = list[tuple[int, float, float]]
"""List of (candidate_index, embedding_score, lexical_score) tuples."""


class _SpanProcessingResult(BaseModel):
    """Result of processing a single answer span."""

    model_config = ConfigDict(frozen=True)

    span_citations: SpanCitations
    num_alignments: int
    embedding_time_ms: float
    alignment_time_ms: float


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
        the answer segment, any exact citations, any retrieval-only support
        passages, and a status indicating citation quality ("supported",
        "partial", or "unsupported").

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
    cfg = config or CitationConfig()
    if cfg.top_k <= 0:
        report_empty_metrics(on_metrics)
        return []

    start_time = time.perf_counter()
    corpus = PreparedCitationCorpus.from_sources(
        sources,
        config=cfg,
        source_segmenter=source_segmenter,
        tokenizer=tokenizer,
        embedder=embedder,
    )

    captured_metrics: list[AlignmentMetrics] = []
    results = corpus.align(
        answer,
        backend=backend,
        answer_segmenter=answer_segmenter,
        aligner=aligner,
        on_metrics=captured_metrics.append if on_metrics is not None else None,
        process_answer_span=_process_answer_span_with_backend,
    )

    if on_metrics is not None:
        align_metrics = captured_metrics[0]
        total_time = (time.perf_counter() - start_time) * 1000
        on_metrics(
            AlignmentMetrics(
                total_time_ms=total_time,
                num_answer_spans=align_metrics.num_answer_spans,
                num_candidates=align_metrics.num_candidates,
                num_alignments=align_metrics.num_alignments,
                embedding_time_ms=(
                    corpus.embedding_build_time_ms + align_metrics.embedding_time_ms
                ),
                alignment_time_ms=align_metrics.alignment_time_ms,
            )
        )

    return results


def _process_answer_span_with_backend(
    *,
    span_index: int,
    answer_span: AnswerSpan,
    tokenizer: Tokenizer,
    candidates: list[Candidate],
    idf: IdfWeights,
    embedding_cache: EmbeddingCache | None,
    embedding_index: EmbeddingIndex | None,
    inverted_index: dict[int, list[tuple[int, int, int, int]]] | None,
    aligner: Aligner | None,
    cfg: CitationConfig,
    backend: str,
) -> tuple[SpanCitations, int, float, float]:
    active_aligner = aligner or _default_aligner(cfg, backend=backend)
    span_result = _process_answer_span(
        span_index=span_index,
        answer_span=answer_span,
        tokenizer=tokenizer,
        candidates=candidates,
        idf=idf,
        embedding_cache=embedding_cache,
        embedding_index=embedding_index,
        inverted_index=inverted_index,
        aligner=active_aligner,
        cfg=cfg,
    )
    return (
        span_result.span_citations,
        span_result.num_alignments,
        span_result.embedding_time_ms,
        span_result.alignment_time_ms,
    )


def _process_answer_span(
    *,
    span_index: int,
    answer_span: AnswerSpan,
    tokenizer: Tokenizer,
    candidates: list[Candidate],
    idf: IdfWeights,
    embedding_cache: EmbeddingCache | None,
    embedding_index: EmbeddingIndex | None,
    inverted_index: dict[int, list[tuple[int, int, int, int]]] | None,
    aligner: Aligner,
    cfg: CitationConfig,
) -> _SpanProcessingResult:
    """Process a single answer span and return citations with timing info."""
    embedding_time = 0.0
    alignment_time = 0.0
    num_alignments = 0

    answer_tokenized = tokenizer.tokenize(answer_span.text)
    answer_tokens = answer_tokenized.token_ids
    citations: list[Citation] = []
    retrieval_support: list[RetrievalSupport] = []

    if answer_tokens and candidates:
        answer_set = frozenset(answer_tokens)
        lexical_scores = _lexical_prefilter(answer_set, candidates, idf)

        embed_start = time.perf_counter()
        query_vector: list[float] | None = None
        if embedding_cache is not None:
            query_vector = embedding_cache.get_vector(span_index)
        embedding_time = (time.perf_counter() - embed_start) * 1000

        selected = _select_candidates(
            candidates,
            answer_tokens=answer_tokens,
            lexical_scores=lexical_scores,
            embedding_index=embedding_index,
            inverted_index=inverted_index,
            query_vector=query_vector,
            cfg=cfg,
        )

        align_start = time.perf_counter()
        selected_candidates = [
            candidates[candidate_index] for candidate_index, _, _ in selected
        ]
        candidate_token_ids = [candidate.token_ids for candidate in selected_candidates]

        # Try Rust fast path for full citation building
        use_rust_fast_path = (
            HAS_RUST_CORE
            and isinstance(aligner, RustSmithWatermanAligner)
            and hasattr(_core, "rust_build_citations_fast")
        )

        if use_rust_fast_path:
            try:
                import json

                candidate_indices_orig = [
                    candidate_index for candidate_index, _, _ in selected
                ]
                embed_scores = [embed_score for _, embed_score, _ in selected]
                lexical_scores_list = [
                    lexical_score for _, _, lexical_score in selected
                ]

                # Build candidate data with NEW indices (0, 1, 2, ...)
                # But store the original global_index in the first field
                candidates_data = [
                    (
                        candidates[idx].global_index,  # Use original global index
                        candidates[idx].source.source_id,
                        candidates[idx].source.source_index,
                        candidates[idx].source.text,
                        candidates[idx].source.full_text,
                        candidates[idx].source.base_doc_offset,
                        candidates[idx].passage.doc_char_start,
                        candidates[idx].passage.doc_char_end,
                        candidates[idx].token_ids,
                        candidates[idx].token_spans,
                    )
                    for idx in candidate_indices_orig
                ]

                # Use sequential indices for alignment (matching candidates_data)
                candidate_indices = list(range(len(candidates_data)))

                # Store mapping back to original for later
                # (Not actually needed since we're using global_index directly)

                # Build config tuple
                config_tuple = (
                    cfg.min_alignment_score,
                    cfg.min_answer_coverage,
                    cfg.min_final_score,
                    cfg.require_all_answer_tokens_in_evidence,
                    cfg.match_score,
                    cfg.weights.alignment,
                    cfg.weights.answer_coverage,
                    cfg.weights.evidence_coverage,
                    cfg.weights.lexical,
                    cfg.weights.embedding,
                )

                multi_span_config = (
                    cfg.multi_span_evidence,
                    cfg.multi_span_merge_gap_chars,
                    cfg.multi_span_max_spans,
                )

                # Call Rust
                result_json = _core.rust_build_citations_fast(  # type: ignore[attr-defined]
                    answer_tokens,
                    candidates_data,
                    candidate_indices,
                    lexical_scores_list,
                    embed_scores,
                    config_tuple,
                    multi_span_config,
                    aligner.match_score,  # type: ignore[attr-defined]
                    aligner.mismatch_score,  # type: ignore[attr-defined]
                    aligner.gap_score,  # type: ignore[attr-defined]
                )

                result = json.loads(result_json)
                num_alignments = len(candidate_indices)

                # Convert to Pydantic models
                for cit in result["citations"]:
                    citations.append(
                        Citation(
                            score=cit["score"],
                            source_id=cit["source_id"],
                            source_index=cit["source_index"],
                            candidate_index=cit[
                                "candidate_index"
                            ],  # Already the global index from Rust
                            char_start=cit["char_start"],
                            char_end=cit["char_end"],
                            evidence=cit["evidence"],
                            evidence_spans=[
                                EvidenceSpan(
                                    char_start=es["char_start"],
                                    char_end=es["char_end"],
                                    evidence=es["evidence"],
                                )
                                for es in cit["evidence_spans"]
                            ],
                            components=cit["components"],
                        )
                    )

                for sup in result["supports"]:
                    retrieval_support.append(
                        RetrievalSupport(
                            retrieval_score=sup["retrieval_score"],
                            source_id=sup["source_id"],
                            source_index=sup["source_index"],
                            candidate_index=sup["candidate_index"],
                            passage_char_start=sup["passage_char_start"],
                            passage_char_end=sup["passage_char_end"],
                            passage_text=sup["passage_text"],
                            embedding_score=sup["embedding_score"],
                            lexical_score=sup["lexical_score"],
                        )
                    )

                alignment_time = (time.perf_counter() - align_start) * 1000
                use_rust_fast_path = True
            except Exception:
                # Fall back to standard path
                use_rust_fast_path = False

        if not use_rust_fast_path:
            align_batch = getattr(aligner, "align_batch", None)
            if align_batch is None:
                alignments = [
                    aligner.align(answer_tokens, token_ids)
                    for token_ids in candidate_token_ids
                ]
            else:
                alignments = align_batch(answer_tokens, candidate_token_ids)

            trusted_alignment_match_counts = type(aligner) in {
                SmithWatermanAligner,
                RustSmithWatermanAligner,
            }
            for (
                candidate_index,
                embed_score,
                lexical_score,
            ), candidate, alignment in zip(
                selected,
                selected_candidates,
                alignments,
                strict=True,
            ):
                candidate = candidates[candidate_index]
                num_alignments += 1

                citation = _build_exact_citation(
                    candidate=candidate,
                    alignment=alignment,
                    answer_tokens=answer_tokens,
                    trusted_alignment_match_counts=trusted_alignment_match_counts,
                    embed_score=embed_score,
                    lexical_score=lexical_score,
                    cfg=cfg,
                )
                if citation is not None:
                    citations.append(citation)
                    continue

                support = _build_retrieval_support_for_candidate(
                    candidate=candidate,
                    alignment=alignment,
                    answer_tokens=answer_tokens,
                    embed_score=embed_score,
                    lexical_score=lexical_score,
                    cfg=cfg,
                )
                if support is not None:
                    retrieval_support.append(support)
            alignment_time = (time.perf_counter() - align_start) * 1000

    citations = _rank_and_limit_citations(citations, cfg)
    status = _span_status(citations, cfg, answer_span.text)
    retrieval_support = _rank_retrieval_support(retrieval_support, cfg)

    return _SpanProcessingResult(
        span_citations=SpanCitations(
            answer_span=answer_span,
            citations=citations,
            retrieval_support=retrieval_support,
            status=status,
        ),
        num_alignments=num_alignments,
        embedding_time_ms=embedding_time,
        alignment_time_ms=alignment_time,
    )


def _build_exact_citation(
    *,
    candidate: Candidate,
    alignment: Alignment,
    answer_tokens: list[int],
    trusted_alignment_match_counts: bool,
    embed_score: float,
    lexical_score: float,
    cfg: CitationConfig,
) -> Citation | None:
    """Build an exact citation when alignment localizes evidence precisely."""
    metrics = _compute_alignment_metrics(alignment, answer_tokens, cfg)
    final_score = _compute_final_score(metrics, lexical_score, embed_score, cfg)
    if not _should_use_alignment(alignment, metrics, cfg):
        return None
    if cfg.require_all_answer_tokens_in_evidence and not (
        trusted_alignment_match_counts and alignment.matches == len(answer_tokens)
    ):
        if not _answer_tokens_match_evidence(
            answer_tokens=answer_tokens,
            evidence_tokens=candidate.token_ids[
                alignment.token_start : alignment.token_end
            ],
        ):
            return None
    evidence_result = _extract_evidence(candidate, alignment, cfg)
    if evidence_result is None or final_score < cfg.min_final_score:
        return None

    abs_start, abs_end, evidence, evidence_spans = evidence_result
    return _build_citation(
        candidate,
        abs_start,
        abs_end,
        evidence,
        evidence_spans,
        final_score,
        metrics,
        lexical_score,
        embed_score,
        alignment.score,
    )


def _build_retrieval_support_for_candidate(
    *,
    candidate: Candidate,
    alignment: Alignment,
    answer_tokens: list[int],
    embed_score: float,
    lexical_score: float,
    cfg: CitationConfig,
) -> RetrievalSupport | None:
    """Build retrieval-only support for candidates that were selected but not localized."""
    if lexical_score <= 0.0 and embed_score < cfg.min_embedding_similarity:
        return None
    metrics = _compute_alignment_metrics(alignment, answer_tokens, cfg)
    return _build_retrieval_support(
        candidate,
        _compute_final_score(metrics, lexical_score, embed_score, cfg),
        lexical_score,
        embed_score,
    )


def _compute_alignment_metrics(
    alignment: Alignment, answer_tokens: list[int], cfg: CitationConfig
) -> dict[str, float]:
    """Compute coverage and alignment metrics for a candidate."""
    matches = alignment.matches
    if alignment.score > 0 and matches <= 0:
        raise RuntimeError(
            "Alignment metrics require detailed match counts; use a traceback-capable "
            "aligner backend for citation scoring"
        )
    answer_len = len(answer_tokens)
    evidence_len = max(1, alignment.token_end - alignment.token_start)
    return {
        "matches": matches,
        "answer_coverage": matches / max(1, answer_len),
        "evidence_coverage": matches / evidence_len,
        "normalized_alignment": alignment.score / max(1, cfg.match_score * answer_len),
    }


def _should_use_alignment(
    alignment: Alignment, metrics: dict[str, float], cfg: CitationConfig
) -> bool:
    """Check if alignment evidence meets quality thresholds."""
    return (
        alignment.score >= cfg.min_alignment_score
        and alignment.token_start < alignment.token_end
        and metrics["answer_coverage"] >= cfg.min_answer_coverage
    )


def _compute_final_score(
    metrics: dict[str, float],
    lexical_score: float,
    embed_score: float,
    cfg: CitationConfig,
) -> float:
    """Compute weighted final citation score."""
    return (
        cfg.weights.alignment * metrics["normalized_alignment"]
        + cfg.weights.answer_coverage * metrics["answer_coverage"]
        + cfg.weights.evidence_coverage * metrics["evidence_coverage"]
        + cfg.weights.lexical * lexical_score
        + cfg.weights.embedding * max(0.0, embed_score)
    )


def _answer_tokens_match_evidence(
    *,
    answer_tokens: Sequence[int],
    evidence_tokens: Iterable[int],
) -> bool:
    if isinstance(evidence_tokens, list) and answer_tokens == evidence_tokens:
        return True
    if isinstance(evidence_tokens, Sequence) and len(answer_tokens) == len(
        evidence_tokens
    ):
        if all(
            answer_token == evidence_token
            for answer_token, evidence_token in zip(
                answer_tokens, evidence_tokens, strict=True
            )
        ):
            return True
    return _answer_token_counts_match_evidence(
        answer_token_counts=Counter(answer_tokens),
        evidence_tokens=evidence_tokens,
    )


def _answer_token_counts_match_evidence(
    *,
    answer_token_counts: Counter[int],
    evidence_tokens: Iterable[int],
) -> bool:
    remaining = answer_token_counts.copy()
    if not remaining:
        return True
    for token_id in evidence_tokens:
        count = remaining.get(token_id)
        if count is None:
            continue
        if count == 1:
            del remaining[token_id]
            if not remaining:
                return True
        else:
            remaining[token_id] = count - 1
    return False


def _build_citation(
    candidate: Candidate,
    abs_start: int,
    abs_end: int,
    evidence: str,
    evidence_spans: list[EvidenceSpan],
    final_score: float,
    metrics: dict[str, float],
    lexical_score: float,
    embed_score: float,
    alignment_score: int,
) -> Citation:
    """Build a Citation object with all components."""
    return Citation(
        score=final_score,
        source_id=candidate.source.source_id,
        source_index=candidate.source.source_index,
        candidate_index=candidate.global_index,
        char_start=abs_start,
        char_end=abs_end,
        evidence=evidence,
        evidence_spans=evidence_spans,
        components={
            "alignment_score": float(alignment_score),
            "normalized_alignment": metrics["normalized_alignment"],
            "matches": metrics["matches"],
            "answer_coverage": metrics["answer_coverage"],
            "evidence_coverage": metrics["evidence_coverage"],
            "lexical_score": float(lexical_score),
            "embedding_score": float(embed_score),
            "num_evidence_spans": float(len(evidence_spans)),
            "evidence_chars_total": float(
                sum(span.char_end - span.char_start for span in evidence_spans)
            ),
            "passage_char_start": float(candidate.passage.doc_char_start),
            "passage_char_end": float(candidate.passage.doc_char_end),
        },
    )


def _build_retrieval_support(
    candidate: Candidate,
    retrieval_score: float,
    lexical_score: float,
    embed_score: float,
) -> RetrievalSupport:
    """Build retrieval-only support metadata for a selected candidate."""
    abs_start = candidate.source.base_doc_offset + candidate.passage.doc_char_start
    abs_end = candidate.source.base_doc_offset + candidate.passage.doc_char_end
    return RetrievalSupport(
        retrieval_score=retrieval_score,
        source_id=candidate.source.source_id,
        source_index=candidate.source.source_index,
        candidate_index=candidate.global_index,
        passage_char_start=abs_start,
        passage_char_end=abs_end,
        passage_text=_slice_source_text(candidate.source, abs_start, abs_end),
        embedding_score=float(embed_score),
        lexical_score=float(lexical_score),
    )


def _extract_evidence(
    candidate: Candidate,
    alignment: Alignment,
    cfg: CitationConfig,
) -> tuple[int, int, str, list[EvidenceSpan]] | None:
    """Extract exact evidence spans from an alignment-backed candidate."""
    evidence_spans = _alignment_to_evidence_spans(candidate, alignment, cfg)
    if evidence_spans is None:
        return None
    abs_start = min(span.char_start for span in evidence_spans)
    abs_end = max(span.char_end for span in evidence_spans)
    evidence = _slice_source_text(candidate.source, abs_start, abs_end)

    return abs_start, abs_end, evidence, evidence_spans


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


def _lexical_prefilter(
    answer_set: frozenset[int],
    candidates: Sequence[Candidate],
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
    candidates: Sequence[Candidate],
    *,
    answer_tokens: list[int],
    lexical_scores: LexicalScores,
    embedding_index: EmbeddingIndex | None,
    inverted_index: dict[int, list[tuple[int, int, int, int]]] | None,
    query_vector: list[float] | None,
    cfg: CitationConfig,
) -> CandidateSelection:
    selected: dict[int, tuple[float, float]] = {}

    # Use inverted index for seeding if available
    if inverted_index is not None and HAS_RUST_CORE:
        _add_index_candidates(
            selected, answer_tokens, inverted_index, lexical_scores, cfg
        )
    else:
        _add_lexical_candidates(selected, candidates, lexical_scores, cfg)
    
    _add_embedding_candidates(selected, embedding_index, query_vector, cfg)

    return _rank_selected_candidates(selected, candidates, cfg)


def _add_index_candidates(
    selected: dict[int, tuple[float, float]],
    answer_tokens: list[int],
    inverted_index: dict[int, list[tuple[int, int, int, int]]],
    lexical_scores: LexicalScores,
    cfg: CitationConfig,
) -> None:
    """Add candidates from inverted index lookup."""
    if cfg.max_candidates_lexical <= 0:
        return
    
    # Convert index to format expected by Rust
    index_data = [
        (token_id, postings) for token_id, postings in inverted_index.items()
    ]
    
    try:
        # Query index to get seed candidates
        seed_candidates = _core.rust_query_inverted_index(  # type: ignore[attr-defined]
            answer_tokens, index_data, cfg.max_candidates_lexical * 3
        )
        
        # Add seed candidates with their lexical scores
        for idx in seed_candidates:
            lexical_score = lexical_scores.get(idx, 0.0)
            if lexical_score > 0.0 or len(selected) < cfg.max_candidates_lexical:
                selected[idx] = (0.0, lexical_score)
                if len(selected) >= cfg.max_candidates_lexical:
                    break
    except Exception:
        # Fall back to lexical prefilter if index query fails
        pass


def _add_lexical_candidates(
    selected: dict[int, tuple[float, float]],
    candidates: Sequence[Candidate],
    lexical_scores: LexicalScores,
    cfg: CitationConfig,
) -> None:
    """Add top lexical candidates to the selected set."""
    if cfg.max_candidates_lexical <= 0 or not lexical_scores:
        return
    ordered = sorted(
        lexical_scores.items(),
        key=lambda item: (-item[1], candidates[item[0]].source.source_index, item[0]),
    )
    for idx, score in ordered[: cfg.max_candidates_lexical]:
        selected[idx] = (0.0, score)


def _add_embedding_candidates(
    selected: dict[int, tuple[float, float]],
    embedding_index: EmbeddingIndex | None,
    query_vector: list[float] | None,
    cfg: CitationConfig,
) -> None:
    """Add top embedding candidates to the selected set."""
    if (
        cfg.max_candidates_embedding <= 0
        or query_vector is None
        or embedding_index is None
    ):
        return
    for idx, score in embedding_index.top_k(query_vector, cfg.max_candidates_embedding):
        prev = selected.get(idx)
        lexical_score = 0.0 if prev is None else prev[1]
        selected[idx] = (score, lexical_score)


def _rank_selected_candidates(
    selected: dict[int, tuple[float, float]],
    candidates: Sequence[Candidate],
    cfg: CitationConfig,
) -> CandidateSelection:
    """Rank and limit selected candidates."""
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


def _slice_source_text(source: NormalizedSource, abs_start: int, abs_end: int) -> str:
    if source.full_text is not None:
        return source.full_text[abs_start:abs_end]
    local_start = abs_start - source.base_doc_offset
    local_end = abs_end - source.base_doc_offset
    return source.text[local_start:local_end]


def _alignment_to_evidence_spans(
    candidate: Candidate,
    alignment: Alignment,
    cfg: CitationConfig,
) -> list[EvidenceSpan] | None:
    """Convert an alignment into one or more evidence spans."""
    spans = _extract_multi_span_evidence(candidate, alignment, cfg)

    if (
        cfg.multi_span_evidence
        and _alignment_has_disjoint_match_blocks(alignment)
        and spans
    ):
        return spans

    if not spans:
        spans = _extract_single_span_evidence(candidate, alignment)

    return spans if spans else None


def _extract_multi_span_evidence(
    candidate: Candidate, alignment: Alignment, cfg: CitationConfig
) -> list[EvidenceSpan]:
    """Extract evidence spans from match blocks if multi-span is enabled."""
    if not cfg.multi_span_evidence or not alignment.match_blocks:
        return []

    spans: list[EvidenceSpan] = []
    for token_start, token_end in alignment.match_blocks:
        span = _create_evidence_span(candidate, token_start, token_end)
        if span is not None:
            spans.append(span)

    spans = _merge_evidence_spans(
        candidate.source, spans, merge_gap_chars=cfg.multi_span_merge_gap_chars
    )

    if cfg.multi_span_max_spans > 0 and len(spans) > cfg.multi_span_max_spans:
        return []

    return spans


def _alignment_has_disjoint_match_blocks(alignment: Alignment) -> bool:
    """Return True when alignment evidence is explicitly split across gaps."""
    return len(alignment.match_blocks) > 1


def _extract_single_span_evidence(
    candidate: Candidate, alignment: Alignment
) -> list[EvidenceSpan]:
    """Extract a single evidence span from the alignment."""
    span = _create_evidence_span(candidate, alignment.token_start, alignment.token_end)
    return [span] if span is not None else []


def _create_evidence_span(
    candidate: Candidate, token_start: int, token_end: int
) -> EvidenceSpan | None:
    """Create an evidence span from token indices."""
    char_span = _token_span_to_char_span(candidate.token_spans, token_start, token_end)
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

    return EvidenceSpan(
        char_start=abs_start,
        char_end=abs_end,
        evidence=_slice_source_text(candidate.source, abs_start, abs_end),
    )


def _merge_evidence_spans(
    source: NormalizedSource,
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


def _rank_retrieval_support(
    retrieval_support: list[RetrievalSupport],
    cfg: CitationConfig,
) -> list[RetrievalSupport]:
    """Rank retrieval-only support entries by retrieval signal only."""
    retrieval_support.sort(
        key=lambda support: (
            -support.retrieval_score,
            support.source_index,
            support.passage_char_start,
            support.candidate_index,
        )
    )
    if cfg.max_retrieval_support <= 0:
        return []
    return retrieval_support[: cfg.max_retrieval_support]


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
    answer_text: str | None = None,
) -> Literal["supported", "partial", "unsupported"]:
    if not citations:
        return "unsupported"
    best = citations[0]
    coverage = float(best.components.get("answer_coverage", 0.0))
    
    # Check for contradictions if answer text is provided
    if answer_text is not None and check_contradiction(answer_text, best.evidence):
        # Downgrade to partial (not unsupported) if contradiction detected
        # because we have evidence, it just contradicts the claim
        return "partial"
    
    if coverage >= cfg.supported_answer_coverage:
        return "supported"
    return "partial"
