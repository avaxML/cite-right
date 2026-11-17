"""An opt-in paraphrase must preserve the grounded claim and source."""

from typing import Literal

import pytest

from cite_right import CitationConfig, SourceDocument, align_citations

from .conftest import KeywordEmbedder, requires_rust
from .paraphrase_fixtures import GUARDS


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


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("rust", marks=requires_rust)]
)
@pytest.mark.parametrize(
    ("answer", "passage"),
    [
        (
            "Maria said John left after the inspection.",
            "After the inspection, Maria said John left.",
        ),
        (
            "Maria asked John to leave after the inspection.",
            "After the inspection, Maria asked John to leave.",
        ),
        (
            "Set the power to 1 MW on the external supply after reconnecting the cable.",
            "After reconnecting the cable, set the power to 1 mW on the external supply.",
        ),
        (
            "Set the controller to close the valve after the inspection.",
            "After the inspection, set the controller to close the valve.",
        ),
        (
            "Tighten the clamp—replace the housing after drying the latch.",
            "After drying the latch, tighten the clamp—replace the housing.",
        ),
        (
            "Set the timer to 102 seconds after closing the hatch.",
            "After closing the hatch, set the timer to 10² seconds.",
        ),
    ],
    ids=[
        "reported-action",
        "requested-action",
        "case-sensitive-unit",
        "embedded-command",
        "compound-command",
        "superscript",
    ],
)
def test_paraphrase_gate_preserves_time_attachment_and_unit_case(
    backend: Literal["python", "rust"], answer: str, passage: str
) -> None:
    [span] = align_citations(
        answer,
        [passage],
        config=CitationConfig(paraphrase_support=True, supported_answer_coverage=1.0),
        backend=backend,
    )
    assert span.citations[0].components["content_coverage"] == pytest.approx(1.0)
    assert span.status == "partial"
    assert span.citations[0].components["content_gate"] == 0.0


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("rust", marks=requires_rust)]
)
def test_lexical_gate_requires_three_distinct_content_tokens(
    backend: Literal["python", "rust"],
) -> None:
    [span] = align_citations(
        "Turn it after warming it.",
        ["After warming it, turn it."],
        config=CitationConfig(
            paraphrase_support=True,
            supported_answer_coverage=1.0,
            paraphrase_lexical_content_coverage=0.0,
        ),
        backend=backend,
    )
    assert span.status == "partial"
    assert span.citations[0].components["content_coverage"] == 1.0
    assert span.citations[0].components["content_gate"] == 0.0


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("rust", marks=requires_rust)]
)
@pytest.mark.parametrize("embedded", [False, True], ids=["lexical", "embedded"])
@pytest.mark.parametrize(
    "separator",
    [
        " - ",
        "--",
        "/",
        " /",
        "/ ",
        " / ",
        "|",
        " | ",
        "&",
        " & ",
        "-",
        " -",
        "- ",
        " then ",
        " plus ",
        " otherwise ",
        " also ",
        " ",
    ],
)
def test_order_of_compound_instructions_is_ambiguous(
    backend: Literal["python", "rust"], separator: str, embedded: bool
) -> None:
    [span] = align_citations(
        f"Tighten the clamp{separator}replace the housing after drying the latch.",
        [f"After drying the latch, tighten the clamp{separator}replace the housing."],
        config=CitationConfig(paraphrase_support=True, supported_answer_coverage=1.0),
        backend=backend,
        embedder=KeywordEmbedder("synthetic probe") if embedded else None,
    )
    assert span.status == "partial"
    assert span.citations[0].components["content_gate"] == 0.0


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("rust", marks=requires_rust)]
)
@pytest.mark.parametrize("embedded", [False, True], ids=["lexical", "embedded"])
def test_hyphenated_uppercase_command_cannot_attach_time_ambiguously(
    backend: Literal["python", "rust"], embedded: bool
) -> None:
    [span] = align_citations(
        "TIGHTEN THE CLAMP-REPLACE THE HOUSING AFTER DRYING THE LATCH.",
        ["AFTER DRYING THE LATCH, TIGHTEN THE CLAMP-REPLACE THE HOUSING."],
        config=CitationConfig(paraphrase_support=True, supported_answer_coverage=1.0),
        backend=backend,
        embedder=KeywordEmbedder("synthetic probe") if embedded else None,
    )
    assert span.status == "partial"
    assert span.citations[0].components["content_gate"] == 0.0
