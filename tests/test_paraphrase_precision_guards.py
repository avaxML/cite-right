"""An opt-in paraphrase must preserve the grounded claim and source."""

from typing import Literal

import pytest

from cite_right import CitationConfig, SourceDocument, align_citations

from .conftest import requires_rust
from .paraphrase_fixtures import GUARDS
from .test_citations_embedding_only_edge_cases import KeywordEmbedder


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("rust", marks=requires_rust)]
)
@pytest.mark.parametrize("embedded", [False, True], ids=["lexical", "embedded"])
@pytest.mark.parametrize(
    ("answer", "passages"),
    [pytest.param(answer, passages, id=name) for name, answer, passages in GUARDS],
)
def test_paraphrase_gate_keeps_precision(
    backend: Literal["python", "rust"],
    embedded: bool,
    answer: str,
    passages: tuple[str, ...],
) -> None:
    cfg = CitationConfig.model_validate(
        {"paraphrase_support": True, "supported_answer_coverage": 1.0, "top_k": 2}
    )
    sources = [
        SourceDocument(id=f"manual-{index}", text=text)
        for index, text in enumerate(passages)
    ]
    embedder = KeywordEmbedder("synthetic probe") if embedded else None
    [span] = align_citations(
        answer, sources, config=cfg, backend=backend, embedder=embedder
    )
    assert span.status == "partial"
    best = span.citations[0]
    assert best.source_id == "manual-0"
    assert best.components["content_coverage"] >= 0.8
    assert best.components["content_gate"] == 0.0
    if embedded:
        assert best.components["embedding_similarity"] == pytest.approx(1.0)
    else:
        assert "embedding_similarity" not in best.components
    assert all(
        "content_gate" not in citation.components for citation in span.citations[1:]
    )


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("rust", marks=requires_rust)]
)
def test_new_admission_requires_the_paraphrase_support_check(
    backend: Literal["python", "rust"],
) -> None:
    config = CitationConfig.model_validate(
        {
            "paraphrase_support": True,
            "min_answer_coverage": 0.95,
            "supported_answer_coverage": 0.35,
        }
    )
    [span] = align_citations(
        "Slowly loosen the clamp after drying the latch.",
        ["After drying the latch, slowly tighten the clamp."],
        config=config,
        backend=backend,
    )
    [citation] = span.citations
    assert citation.components["answer_coverage"] >= config.supported_answer_coverage
    assert citation.components["content_coverage"] < config.min_answer_coverage
    assert span.status == "partial"
