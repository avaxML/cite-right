"""Replay the synthetic manual with and without paraphrase support."""

from collections import Counter
from typing import Literal

import pytest

from cite_right import CitationConfig, SourceDocument, align_citations
from cite_right.contradiction import check_contradiction, has_entity_swap

from .conftest import KeywordEmbedder, requires_rust
from .paraphrase_fixtures import ANSWERS, PASSAGES

EXPECTED_PASSAGES = (0, 1, 2, (3, 4), 5, 6, 7)

CFG = CitationConfig.strict().model_copy(
    update={
        "top_k": 1,
        "max_citations_per_source": 1,
        "require_all_answer_tokens_in_evidence": False,
        "window_size_sentences": 2,
        "window_stride_sentences": 1,
        "multi_span_evidence": True,
        "prefer_source_order": True,
        "supported_answer_coverage": 0.75,
    }
)


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("rust", marks=requires_rust)]
)
def test_manual_replay_with_gate_disabled(backend: Literal["python", "rust"]) -> None:
    sources = [
        SourceDocument(id=f"manual-{i}", text=passage)
        for i, passage in enumerate(PASSAGES)
    ]
    spans = align_citations(" ".join(ANSWERS), sources, config=CFG, backend=backend)
    assert [span.answer_span.text for span in spans] == list(ANSWERS)
    assert Counter(span.status for span in spans) == {"partial": 3, "unsupported": 15}
    assert [span.status for span in spans] == (
        ["unsupported"] * 5 + ["partial"] * 3 + ["unsupported"] * 10
    )

    relaxed = CFG.model_copy(update={"min_answer_coverage": 0, "min_final_score": 0})
    for i, expected in enumerate((0.22, 0.33, 0.38, 0.19, 0.14)):
        passage_indices = EXPECTED_PASSAGES[i]
        indices = (
            passage_indices
            if isinstance(passage_indices, tuple)
            else (passage_indices,)
        )
        for index in indices:
            probe = align_citations(
                ANSWERS[i], [sources[index]], config=relaxed, backend=backend
            )
            coverage = probe[0].citations[0].components
            assert coverage["answer_coverage"] == pytest.approx(expected, abs=0.05)
            assert 0.5 <= coverage["evidence_coverage"] <= 1.0

    for i, expected_answer, expected_evidence in ((5, 0.55, 0.86), (6, 0.40, 1.0)):
        citation = spans[i].citations[0]
        assert not has_entity_swap(ANSWERS[i], PASSAGES[citation.source_index])
        assert citation.components["answer_coverage"] == pytest.approx(
            expected_answer, abs=0.05
        )
        assert citation.components["evidence_coverage"] == pytest.approx(
            expected_evidence, abs=0.05
        )
    assert check_contradiction(ANSWERS[7], PASSAGES[spans[7].citations[0].source_index])


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("rust", marks=requires_rust)]
)
@pytest.mark.parametrize("embedded", [False, True], ids=["lexical", "embedded"])
@pytest.mark.xfail(
    strict=True,
    reason="The replay's broad paraphrases have insufficient content evidence",
)
def test_manual_replay_target_recovery(
    backend: Literal["python", "rust"], embedded: bool
) -> None:
    config = CitationConfig.model_validate(
        CFG.model_dump() | {"paraphrase_support": True}
    )
    embedder = KeywordEmbedder("synthetic probe") if embedded else None
    spans = align_citations(
        " ".join(ANSWERS),
        list(PASSAGES),
        config=config,
        backend=backend,
        embedder=embedder,
    )
    assert [span.status for span in spans[:8]] == ["supported"] * 7 + ["partial"]


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("rust", marks=requires_rust)]
)
@pytest.mark.parametrize("embedded", [False, True], ids=["lexical", "embedded"])
def test_enabled_replay_preserves_abstentions(
    backend: Literal["python", "rust"], embedded: bool
) -> None:
    config = CitationConfig.model_validate(
        CFG.model_dump() | {"paraphrase_support": True}
    )
    embedder = KeywordEmbedder("synthetic probe") if embedded else None
    spans = align_citations(
        " ".join(ANSWERS),
        list(PASSAGES),
        config=config,
        backend=backend,
        embedder=embedder,
    )
    assert [span.answer_span.text for span in spans] == list(ANSWERS)
    assert spans[7].status == "partial"
    assert all(span.status != "supported" for span in spans[8:])
