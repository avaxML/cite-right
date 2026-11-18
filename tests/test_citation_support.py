from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal
from unittest.mock import Mock

import pytest

from cite_right import PreparedCitationCorpus, SourceDocument, align_citations
from cite_right import citations as citations_module
from cite_right.core.citation_config import CitationConfig, CitationWeights

from .conftest import requires_rust


class RankedPassageEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


def _track_native_builders(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from cite_right import _core

    completed: list[str] = []

    def track(name: str, build: Callable[..., Any]) -> Callable[..., Any]:
        def build_and_record(*args: Any, **kwargs: Any) -> Any:
            result = build(*args, **kwargs)
            completed.append(name)
            return result

        return build_and_record

    monkeypatch.setattr(
        _core.PreparedCorpus,
        "build_citations",
        track("prepared", _core.PreparedCorpus.build_citations),
    )
    monkeypatch.setattr(
        _core,
        "rust_build_citations_fast",
        track("legacy", _core.rust_build_citations_fast),
    )
    return completed


@pytest.mark.parametrize(
    ("backend", "use_rust"),
    [
        pytest.param("python", False, id="python"),
        pytest.param("rust", True, marks=requires_rust, id="prepared-rust"),
        pytest.param("rust", False, marks=requires_rust, id="legacy-rust"),
    ],
)
@pytest.mark.parametrize("per_source_limit", [False, True], ids=["top-k", "per-source"])
@pytest.mark.parametrize(
    "weak_passage",
    [
        pytest.param(
            "The brass rotor spins smoothly every 30 days.", id="contradiction"
        ),
        pytest.param(
            "The brass rotor spins every 90 days.", id="incomplete-alias-coverage"
        ),
    ],
)
def test_supporting_citation_fills_slot_released_by_demotion(
    backend: Literal["python", "rust"],
    use_rust: bool,
    per_source_limit: bool,
    weak_passage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer = "The brass rotor spins smoothly every 90 days."
    alternative = "The brass rotor turns smoothly every 90 days."
    retrieval_only = "Rotor maintenance is documented separately."
    sources = [SourceDocument(id="primary", text=answer)]
    if per_source_limit:
        sources.append(
            SourceDocument(id="fallback", text=f"{weak_passage} {alternative}")
        )
    else:
        sources.extend(
            [
                SourceDocument(id="weak", text=weak_passage),
                SourceDocument(id="fallback", text=alternative),
            ]
        )
    sources.append(SourceDocument(id="retrieval-only", text=retrieval_only))
    config = CitationConfig(
        top_k=3 if per_source_limit else 2,
        max_citations_per_source=1 if per_source_limit else 2,
        max_retrieval_support=1 if per_source_limit else 2,
        min_answer_coverage=0.6,
        term_aliases=(("spins", "turns"),),
        report_possible_source=True,
        weights=CitationWeights(
            alignment=0.0, answer_coverage=0.0, lexical=0.0, embedding=2.0
        ),
    )
    corpus = PreparedCitationCorpus.from_sources(
        sources,
        config=config,
        use_rust=use_rust,
        embedder=RankedPassageEmbedder(
            {
                answer: [1.0, 0.0],
                weak_passage: [0.8, 0.6],
                alternative: [0.6, 0.8],
                retrieval_only: [0.6, -0.8],
            }
        ),
    )
    assert (corpus.rust_corpus is not None) == use_rust
    python_builder = Mock(wraps=citations_module._build_python_citations)
    monkeypatch.setattr(citations_module, "_build_python_citations", python_builder)
    native_builders = _track_native_builders(monkeypatch) if backend == "rust" else []
    [span] = corpus.align(answer, backend=backend)

    if backend == "rust":
        assert native_builders == ["prepared" if use_rust else "legacy"]
        python_builder.assert_not_called()
    else:
        python_builder.assert_called_once()

    assert span.status == span.status_reason == "supported"
    assert span.contradiction_checks == []
    assert span.possible_source is None
    primary, secondary = span.citations
    assert primary.source_id == "primary"
    assert secondary.source_id == "fallback"
    assert secondary.evidence == alternative.rstrip(".")
    expected_support = (
        [weak_passage] if per_source_limit else [weak_passage, retrieval_only]
    )
    assert [
        support.passage_text for support in span.retrieval_support
    ] == expected_support
    assert span.retrieval_support[0].retrieval_score == pytest.approx(1.6)


def test_supporting_secondary_does_not_replace_contradicted_primary() -> None:
    answer = "The brass rotor spins smoothly every 90 days."
    changed = "The brass rotor spins smoothly every 30 days."
    alternative = "The brass rotor turns smoothly every 90 days."
    sources = [
        SourceDocument(id="changed", text=changed),
        SourceDocument(id="supporting", text=alternative),
    ]
    [span] = align_citations(
        answer,
        sources,
        config=CitationConfig(
            term_aliases=(("spins", "turns"),),
            report_possible_source=True,
            weights=CitationWeights(
                alignment=0.0, answer_coverage=0.0, lexical=0.0, embedding=1.0
            ),
        ),
        embedder=RankedPassageEmbedder(
            {answer: [1.0, 0.0], changed: [1.0, 0.0], alternative: [0.6, 0.8]}
        ),
    )
    assert [citation.source_id for citation in span.citations] == [
        "changed",
        "supporting",
    ]
    assert span.status == "partial"
    assert span.status_reason == "contradiction"
    assert span.contradiction_checks == ["has_number_mismatch"]
    assert span.possible_source is not None
    assert span.possible_source.source_id == "changed"
