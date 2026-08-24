from __future__ import annotations

import math
import time
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from cite_right.core.citation_config import CitationConfig
from cite_right.core.interfaces import Aligner, AnswerSegmenter, Segmenter, Tokenizer
from cite_right.core.results import (
    AnswerSpan,
    SourceChunk,
    SourceDocument,
    SpanCitations,
    TokenizedText,
)
from cite_right.models.base import Embedder
from cite_right.models.embedding_index import EmbeddingIndex
from cite_right.text.answer_segmenter import SimpleAnswerSegmenter
from cite_right.text.passage import Passage, generate_passages
from cite_right.text.segmenter_simple import SimpleSegmenter
from cite_right.text.tokenizer import SimpleTokenizer

if TYPE_CHECKING:
    from cite_right._core import InvertedIndex, PreparedCorpus as RustPreparedCorpus

try:
    from cite_right._core import (  # type: ignore[attr-defined]
        InvertedIndex,
        rust_tokenize_and_prepare,
    )
    from cite_right._core import (
        PreparedCorpus as RustPreparedCorpus,
    )

    RUST_PREPARE_AVAILABLE = True
except ImportError:
    RUST_PREPARE_AVAILABLE = False
    if not TYPE_CHECKING:
        InvertedIndex = object  # type: ignore[misc,assignment]
        RustPreparedCorpus = object  # type: ignore[misc,assignment]

MetricsCallback = Callable[["AlignmentMetrics"], None]
LexicalScores = dict[int, float]
IdfWeights = dict[int, float]


class AlignmentMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_time_ms: float
    num_answer_spans: int
    num_candidates: int
    num_alignments: int
    embedding_time_ms: float = 0.0
    alignment_time_ms: float = 0.0


class NormalizedSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    source_index: int
    text: str
    base_doc_offset: int
    full_text: str | None


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    global_index: int
    source: NormalizedSource
    passage: Passage
    token_ids: list[int]
    token_spans: list[tuple[int, int]]
    token_set: frozenset[int]


class EmbeddingCache(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedder: Embedder
    answer_spans: list[AnswerSpan]
    vectors: list[list[float]] = Field(default_factory=list)
    _computed: bool = False

    def get_vector(self, span_index: int) -> list[float]:
        if not self._computed:
            self.vectors = self.embedder.encode(
                [span.text for span in self.answer_spans]
            )
            self._computed = True
        return self.vectors[span_index]


class PreparedCitationCorpus(BaseModel):
    """Prepared source-side corpus state for repeated citation alignment."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: CitationConfig
    tokenizer: Tokenizer
    source_segmenter: Segmenter
    embedder: Embedder | None = None
    normalized_sources: list[NormalizedSource]
    source_passages: list[tuple[NormalizedSource, list[Passage]]]
    candidates: list[Candidate]
    idf: IdfWeights
    embedding_index: EmbeddingIndex | None = None
    inverted_index: InvertedIndex | None = None  # Rust InvertedIndex object
    rust_corpus: RustPreparedCorpus | None = (
        None  # Rust PreparedCorpus object (when available)
    )
    _embedding_build_time_ms: float = PrivateAttr(default=0.0)

    @property
    def embedding_build_time_ms(self) -> float:
        """Time spent building the reusable source embedding index."""
        return self._embedding_build_time_ms

    @classmethod
    def from_sources(
        cls,
        sources: Sequence[str | SourceDocument | SourceChunk],
        *,
        config: CitationConfig | None = None,
        source_segmenter: Segmenter | None = None,
        tokenizer: Tokenizer | None = None,
        embedder: Embedder | None = None,
        use_rust: bool = True,
    ) -> "PreparedCitationCorpus":
        cfg = config or CitationConfig()
        source_segmenter = source_segmenter or SimpleSegmenter()
        tokenizer = tokenizer or SimpleTokenizer()

        normalized_sources = normalize_sources(sources)

        # Try Rust fast path for prepare phase if available
        if (
            use_rust
            and RUST_PREPARE_AVAILABLE
            and embedder is None
            and isinstance(tokenizer, SimpleTokenizer)
            and isinstance(source_segmenter, SimpleSegmenter)
        ):
            try:
                return cls._from_sources_rust(
                    normalized_sources, cfg, source_segmenter, tokenizer
                )
            except Exception as e:
                # Fall back to Python if Rust path fails
                import sys

                print(
                    f"Rust prepare path failed: {e}, falling back to Python",
                    file=sys.stderr,
                )
                pass

        # Python fallback path
        source_passages = build_source_passages(
            normalized_sources, source_segmenter, cfg
        )
        candidates = build_candidates(source_passages, tokenizer)
        idf = compute_idf(candidates)
        embedding_build_time_ms = 0.0
        embedding_index = None
        if embedder is not None:
            embedding_start = time.perf_counter()
            embedding_index = build_embedding_index(embedder, candidates)
            embedding_build_time_ms = (time.perf_counter() - embedding_start) * 1000

        corpus = cls(
            config=cfg,
            tokenizer=tokenizer,
            source_segmenter=source_segmenter,
            embedder=embedder,
            normalized_sources=normalized_sources,
            source_passages=source_passages,
            candidates=candidates,
            idf=idf,
            embedding_index=embedding_index,
            inverted_index=None,
        )
        corpus._embedding_build_time_ms = embedding_build_time_ms
        return corpus

    @classmethod
    def _from_sources_rust(
        cls,
        normalized_sources: list[NormalizedSource],
        cfg: CitationConfig,
        source_segmenter: SimpleSegmenter,
        tokenizer: SimpleTokenizer,
    ) -> "PreparedCitationCorpus":
        """Fast path using Rust for tokenization, passages, candidates, and IDF."""
        # Call Rust to get PreparedCorpus object (keeps data in Rust)
        source_texts = [src.text for src in normalized_sources]
        rust_corpus = rust_tokenize_and_prepare(
            source_texts,
            cfg.window_size_sentences,
            cfg.window_stride_sentences,
        )

        # Populate the Python tokenizer's vocab from Rust
        rust_vocab = rust_corpus.get_vocab()
        tokenizer._vocab = {
            normalized: int(token_id) for normalized, token_id in rust_vocab
        }
        tokenizer._next_id = (
            max(tokenizer._vocab.values()) + 1 if tokenizer._vocab else 1
        )

        # Build lightweight Python candidates WITHOUT fetching token data
        # Token IDs will be fetched on-demand at alignment time
        candidates: list[Candidate] = []
        source_passages: list[tuple[NormalizedSource, list[Passage]]] = []

        # Fetch minimal candidate info (no tokens)
        all_candidate_info = rust_corpus.get_all_candidate_info()

        current_source_idx = 0
        passages: list[Passage] = []

        for global_idx, source_idx, passage_start, passage_end in all_candidate_info:
            # Check if we've moved to a new source
            if source_idx != current_source_idx:
                source_passages.append(
                    (normalized_sources[current_source_idx], passages)
                )
                passages = []
                current_source_idx = source_idx

            source = normalized_sources[source_idx]
            passage = Passage(
                doc_char_start=passage_start,
                doc_char_end=passage_end,
                segment_start=len(passages),
                segment_end=len(passages) + 1,
                source_text=source.text,
            )
            passages.append(passage)

            # Build candidate WITHOUT token_ids (empty lists)
            # Will fetch from rust_corpus when needed for alignment
            candidates.append(
                Candidate(
                    global_index=global_idx,
                    source=source,
                    passage=passage,
                    token_ids=[],  # Empty - will fetch from rust_corpus on demand
                    token_spans=[],
                    token_set=frozenset(),
                )
            )

        # Append last source
        if passages:
            source_passages.append((normalized_sources[current_source_idx], passages))

        # Convert IDF from Rust
        rust_idf = rust_corpus.get_idf()
        idf: IdfWeights = {int(token_id): weight for token_id, weight in rust_idf}

        return cls(
            config=cfg,
            tokenizer=tokenizer,
            source_segmenter=source_segmenter,
            embedder=None,
            normalized_sources=normalized_sources,
            source_passages=source_passages,
            candidates=candidates,
            idf=idf,
            embedding_index=None,
            inverted_index=None,  # Index is in rust_corpus now
            rust_corpus=rust_corpus,
        )

    def align(
        self,
        answer: str,
        *,
        backend: str = "auto",
        answer_segmenter: AnswerSegmenter | None = None,
        aligner: Aligner | None = None,
        on_metrics: MetricsCallback | None = None,
        process_answer_span: Callable[..., tuple[SpanCitations, int, float, float]]
        | None = None,
    ) -> list[SpanCitations]:
        if self.config.top_k <= 0:
            report_empty_metrics(on_metrics)
            return []

        if process_answer_span is None:
            from cite_right.citations import _process_answer_span_with_backend

            process_answer_span = _process_answer_span_with_backend

        start_time = time.perf_counter()
        answer_segmenter = answer_segmenter or SimpleAnswerSegmenter()
        answer_spans = answer_segmenter.segment(answer)
        embedding_cache = build_answer_embedding_cache(self.embedder, answer_spans)
        resolved_aligner = aligner

        if resolved_aligner is None:
            from cite_right.citations import _default_aligner

            resolved_aligner = _default_aligner(self.config, backend=backend)

        output: list[SpanCitations] = []
        num_alignments = 0
        alignment_time = 0.0
        embedding_time = 0.0

        for span_index, answer_span in enumerate(answer_spans):
            span_citations, span_alignments, span_embedding_ms, span_alignment_ms = (
                process_answer_span(
                    span_index=span_index,
                    answer_span=answer_span,
                    tokenizer=self.tokenizer,
                    candidates=self.candidates,
                    idf=self.idf,
                    embedding_cache=embedding_cache,
                    embedding_index=self.embedding_index,
                    inverted_index=self.inverted_index,
                    rust_corpus=self.rust_corpus,
                    aligner=resolved_aligner,
                    cfg=self.config,
                    backend=backend,
                )
            )
            output.append(span_citations)
            num_alignments += span_alignments
            embedding_time += span_embedding_ms
            alignment_time += span_alignment_ms

        if on_metrics is not None:
            on_metrics(
                AlignmentMetrics(
                    total_time_ms=(time.perf_counter() - start_time) * 1000,
                    num_answer_spans=len(answer_spans),
                    num_candidates=len(self.candidates),
                    num_alignments=num_alignments,
                    embedding_time_ms=embedding_time,
                    alignment_time_ms=alignment_time,
                )
            )

        return output


def report_empty_metrics(on_metrics: MetricsCallback | None) -> None:
    if on_metrics is not None:
        on_metrics(
            AlignmentMetrics(
                total_time_ms=0.0,
                num_answer_spans=0,
                num_candidates=0,
                num_alignments=0,
            )
        )


def build_embedding_index(
    embedder: Embedder | None,
    candidates: list[Candidate],
) -> EmbeddingIndex | None:
    if embedder is None:
        return None
    return EmbeddingIndex.build(
        embedder, [candidate.passage.text for candidate in candidates]
    )


def build_answer_embedding_cache(
    embedder: Embedder | None,
    answer_spans: list[AnswerSpan],
) -> EmbeddingCache | None:
    if embedder is None:
        return None
    return EmbeddingCache(embedder=embedder, answer_spans=answer_spans)


def normalize_sources(
    sources: Sequence[str | SourceDocument | SourceChunk],
) -> list[NormalizedSource]:
    normalized: list[NormalizedSource] = []
    for index, item in enumerate(sources):
        if isinstance(item, str):
            normalized.append(
                NormalizedSource(
                    source_id=str(index),
                    source_index=index,
                    text=item,
                    base_doc_offset=0,
                    full_text=item,
                )
            )
        elif isinstance(item, SourceDocument):
            normalized.append(
                NormalizedSource(
                    source_id=item.id,
                    source_index=index,
                    text=item.text,
                    base_doc_offset=0,
                    full_text=item.text,
                )
            )
        else:
            source_index = item.source_index if item.source_index is not None else index
            normalized.append(
                NormalizedSource(
                    source_id=item.source_id,
                    source_index=source_index,
                    text=item.text,
                    base_doc_offset=item.doc_char_start,
                    full_text=item.document_text,
                )
            )
    return normalized


def build_source_passages(
    sources: Sequence[NormalizedSource],
    segmenter: Segmenter,
    cfg: CitationConfig,
) -> list[tuple[NormalizedSource, list[Passage]]]:
    output: list[tuple[NormalizedSource, list[Passage]]] = []
    for source in sources:
        passages = generate_passages(
            source.text,
            segmenter=segmenter,
            window_size_sentences=cfg.window_size_sentences,
            window_stride_sentences=cfg.window_stride_sentences,
        )
        output.append((source, passages))
    return output


def build_candidates(
    source_passages: Sequence[tuple[NormalizedSource, list[Passage]]],
    tokenizer: Tokenizer,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    global_index = 0
    for source, passages in source_passages:
        tokenized_source = tokenizer.tokenize(source.text)
        token_starts = [start for start, _ in tokenized_source.token_spans]
        token_ends = [end for _, end in tokenized_source.token_spans]
        for passage in passages:
            tokenized = slice_tokenized_text(
                tokenized_source,
                passage,
                token_starts=token_starts,
                token_ends=token_ends,
            )
            candidates.append(
                Candidate(
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


def slice_tokenized_text(
    tokenized: TokenizedText,
    passage: Passage,
    *,
    token_starts: Sequence[int] | None = None,
    token_ends: Sequence[int] | None = None,
) -> TokenizedText:
    start = passage.doc_char_start
    end = passage.doc_char_end
    token_ids: list[int] = []
    token_spans: list[tuple[int, int]] = []

    token_starts = token_starts or [
        span_start for span_start, _ in tokenized.token_spans
    ]
    token_ends = token_ends or [span_end for _, span_end in tokenized.token_spans]
    start_index = bisect_right(token_ends, start)
    end_index = bisect_left(token_starts, end, lo=start_index)

    for token_id, (token_start, token_end) in zip(
        tokenized.token_ids[start_index:end_index],
        tokenized.token_spans[start_index:end_index],
        strict=False,
    ):
        local_start = max(token_start, start) - start
        local_end = min(token_end, end) - start
        if local_start >= local_end:
            continue
        token_ids.append(token_id)
        token_spans.append((local_start, local_end))

    return TokenizedText(
        text=passage.text, token_ids=token_ids, token_spans=token_spans
    )


def compute_idf(candidates: Sequence[Candidate]) -> IdfWeights:
    df: dict[int, int] = {}
    for candidate in candidates:
        for token_id in candidate.token_set:
            df[token_id] = df.get(token_id, 0) + 1
    n = len(candidates)
    return {
        token_id: math.log((n + 1) / (count + 1)) + 1.0
        for token_id, count in df.items()
    }
