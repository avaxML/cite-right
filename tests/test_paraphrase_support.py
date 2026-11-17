"""Tests for grounded how-to/news paraphrases (issue #49).

These fixtures follow the RAGTruth repros. Sequential Smith-Waterman match
count on a one-sentence window was marking them unsupported even though the
same content words are in the source. Status may be partial.
"""

import os
from typing import Literal, Sequence, cast

import pytest
from pydantic import ValidationError

from cite_right import SourceChunk, SourceDocument, align_citations
from cite_right.core.citation_config import CitationConfig
from cite_right.text.content_coverage import content_token_coverage, stopword_token_ids
from cite_right.text.tokenizer import SimpleTokenizer

from .conftest import KeywordEmbedder, requires_embeddings, requires_rust

_BNP_ANSWER = (
    "The normal B-type Natriuretic Peptide (BNP) refers to a low amount of the "
    "BNP hormone found in the blood, produced by the heart as a measure of how "
    "well it is working."
)

_BNP_SOURCES = [
    SourceDocument(
        id="bnp_p1",
        text=(
            "A brain natriuretic peptide (BNP) test measures the amount of the "
            "BNP hormone in the blood. BNP values tend to increase with age and "
            "are higher in women than men."
        ),
    ),
    SourceDocument(
        id="bnp_p2",
        text=(
            "The B-type Natriuretic Peptide (BNP) Test is a blood test for heart "
            "failure. If your B-type natriuretic peptide level is high, you "
            "probably have heart failure."
        ),
    ),
    SourceDocument(
        id="bnp_p3",
        text=(
            "BNP is made by your heart and shows how well your heart is working. "
            "Normally, only a low amount of BNP is found in your blood."
        ),
    ),
]

_FARAGE_ANSWER = (
    "Farage has declined the challenge, saying he does not want to engage in a "
    "physical altercation and prefers to focus on the upcoming elections."
)

_FARAGE_SOURCE = SourceDocument(
    id="farage_cnn",
    text=(
        "Farage, who is on the campaign trail ahead of Britain's general "
        "elections, said he did not intend to cross swords with the prince. "
        "I'm not intending to accept the offer, a spokesman quoted him as saying."
    ),
)

_ZIPPER_SOURCE = SourceDocument(
    id="zipper_p2",
    text=(
        "Make a zipper sandwich. Place 1 piece of lining fabric face up, then "
        "the zipper and then the outer fabric right side facing down. Line up "
        "the three edges. With a zipper foot sew across the top between the "
        "edge of your zipper sandwich and the zipper teeth."
    ),
)


def _lexical_config() -> CitationConfig:
    return CitationConfig(top_k=3)


def _align(answer: str, sources: list[SourceDocument], *, backend: str = "auto"):
    return align_citations(
        answer,
        sources,
        config=_lexical_config(),
        backend=backend,  # type: ignore[arg-type]
    )


def test_definition_paraphrase_bnp_not_unsupported() -> None:
    """14760-style definition restatement across three RAG passages."""
    results = _align(_BNP_ANSWER, _BNP_SOURCES)
    assert len(results) == 1
    assert results[0].status in {"supported", "partial"}
    assert results[0].citations, "Smith-Waterman must still localize evidence"


def test_negation_paraphrase_farage_not_unsupported() -> None:
    """466-style refusal paraphrase: declined vs did not intend / not intending."""
    results = _align(_FARAGE_ANSWER, [_FARAGE_SOURCE])
    assert len(results) == 1
    assert results[0].status in {"supported", "partial"}
    assert results[0].citations


def test_extractive_zipper_near_copy_stays_supported() -> None:
    """12840-style extractive how-to copy stays supported."""
    answer = (
        "Place 1 piece of lining fabric face up, then the zipper and then the "
        "outer fabric right side facing down. Line up the three edges."
    )
    results = _align(answer, [_ZIPPER_SOURCE])
    assert results
    assert all(result.status == "supported" for result in results)


def test_conversational_wrapper_stays_unsupported() -> None:
    """Wrappers with no source content stay unsupported."""
    results = _align("Sure!", _BNP_SOURCES)
    assert len(results) == 1
    assert results[0].status == "unsupported"
    assert results[0].citations == []


def test_heres_the_summary_wrapper_stays_unsupported() -> None:
    results = _align("Here's the summary.", _BNP_SOURCES)
    assert len(results) == 1
    assert results[0].status == "unsupported"


def test_contradiction_stays_partial_not_unsupported() -> None:
    """Issue #49: contradiction remains partial, not unsupported."""
    sources = [SourceDocument(id="vax", text="The vaccine is safe and effective.")]
    results = _align("The vaccine is not safe.", sources)
    assert len(results) == 1
    assert results[0].status == "partial"
    assert results[0].citations


@requires_rust
def test_paraphrase_python_and_rust_agree() -> None:
    python = _align(_BNP_ANSWER, _BNP_SOURCES, backend="python")
    rust = _align(_BNP_ANSWER, _BNP_SOURCES, backend="rust")
    assert python[0].status == rust[0].status
    assert python[0].status in {"supported", "partial"}


def test_content_token_coverage_ignores_stopwords() -> None:
    tokenizer = SimpleTokenizer()
    answer = tokenizer.tokenize("Farage declined the challenge")
    passage = tokenizer.tokenize("Farage said he did not intend to cross swords")
    vocab = tokenizer._vocab
    coverage = content_token_coverage(
        answer.token_ids,
        passage.token_ids,
        stopword_token_ids(vocab),
    )
    assert coverage > 0.0
    # "the" must not be enough to count as coverage by itself
    wrapper = tokenizer.tokenize("Here's the summary")
    empty = content_token_coverage(
        wrapper.token_ids,
        passage.token_ids,
        stopword_token_ids(vocab),
    )
    assert empty == 0.0


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("rust", marks=requires_rust)]
)
@pytest.mark.parametrize("with_embeddings", [False, True], ids=["lexical", "embedding"])
@pytest.mark.parametrize(
    ("answer", "coverage"),
    [
        ("Tighten the clamp after drying the latch.", 1.0),
        ("Tighten the clamp once drying the latch is complete.", 0.8),
    ],
)
def test_paraphrase_gate_supports_clause_reordering(
    backend: Literal["python", "rust"],
    with_embeddings: bool,
    answer: str,
    coverage: float,
) -> None:
    passage = "After drying the latch, tighten the clamp."
    embedder = KeywordEmbedder("clamp") if with_embeddings else None
    config = CitationConfig(supported_answer_coverage=0.9)
    [original] = align_citations(
        answer, [passage], config=config, backend=backend, embedder=embedder
    )
    [promoted] = align_citations(
        answer,
        [passage],
        config=CitationConfig.model_validate(
            config.model_dump() | {"paraphrase_support": True}
        ),
        backend=backend,
        embedder=embedder,
    )
    assert original.status == "partial"
    assert promoted.status == "supported"
    [citation] = promoted.citations
    assert citation.components["content_gate"] == 1.0
    assert citation.components["content_coverage"] == pytest.approx(coverage)
    assert ("embedding_similarity" in citation.components) is with_embeddings
    assert passage[citation.char_start : citation.char_end] == citation.evidence
    assert "drying the latch" in citation.evidence
    assert "tighten the clamp" in citation.evidence


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("rust", marks=requires_rust)]
)
def test_paraphrase_gate_can_admit_reworded_evidence(
    backend: Literal["python", "rust"],
) -> None:
    text = "After drying the latch, tighten the clamp."
    answer = "Tighten the clamp once drying the latch is complete."
    cfg = CitationConfig(min_answer_coverage=0.9, supported_answer_coverage=0.9)
    [without] = align_citations(answer, [text], config=cfg, backend=backend)
    assert without.status == "unsupported"
    active = CitationConfig.model_validate(
        cfg.model_dump() | {"paraphrase_support": True}
    )
    [with_gate] = align_citations(answer, [text], config=active, backend=backend)
    assert with_gate.status == "supported"


@pytest.mark.parametrize("bad_vector", [[0.0, 0.0], [float("nan"), 1.0], [1.0]])
@pytest.mark.parametrize(
    "backend", ["python", pytest.param("rust", marks=requires_rust)]
)
def test_paraphrase_gate_abstains_on_invalid_similarity(
    backend: Literal["python", "rust"], bad_vector: list[float]
) -> None:
    class InvalidQueryEmbedder:
        def encode(self, texts: Sequence[str]) -> list[list[float]]:
            return [
                bad_vector if text.startswith("Tighten") else [1.0, 0.0]
                for text in texts
            ]

    [span] = align_citations(
        "Tighten the clamp after drying the latch.",
        ["After drying the latch, tighten the clamp."],
        config=CitationConfig.model_validate(
            {"paraphrase_support": True, "supported_answer_coverage": 0.9}
        ),
        embedder=InvalidQueryEmbedder(),
        backend=backend,
    )
    assert span.status == "partial"
    assert span.citations[0].components["content_gate"] == 0.0


@pytest.mark.parametrize(
    "field",
    [
        "paraphrase_content_coverage",
        "paraphrase_embedding_similarity",
        "paraphrase_lexical_content_coverage",
    ],
)
@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan")])
def test_paraphrase_config_validates_thresholds(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        CitationConfig.model_validate({field: value})


@pytest.mark.embeddings
@requires_embeddings
@pytest.mark.skipif(
    os.environ.get("CITE_RIGHT_RUN_EMBEDDINGS_TESTS") != "1",
    reason="Set CITE_RIGHT_RUN_EMBEDDINGS_TESTS=1 to run embeddings tests",
)
def test_embeddings_paraphrase_supports_reordered_instruction() -> None:
    from cite_right.models.sbert_embedder import SentenceTransformerEmbedder

    cfg = CitationConfig.model_validate(
        {"paraphrase_support": True, "supported_answer_coverage": 0.9}
    )
    [span] = align_citations(
        "Tighten the clamp after drying the latch.",
        ["After drying the latch, tighten the clamp."],
        config=cfg,
        embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
    )
    assert span.status == "supported"
    assert span.citations[0].components["content_gate"] == 1.0


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("rust", marks=requires_rust)]
)
def test_paraphrase_gate_needs_embedding_similarity_and_batches_queries(
    backend: Literal["python", "rust"],
) -> None:
    class OrthogonalEmbedder:
        def __init__(self) -> None:
            self.batches: list[tuple[str, ...]] = []

        def encode(self, texts: Sequence[str]) -> list[list[float]]:
            self.batches.append(tuple(texts))
            return [
                [1.0, 0.0] if text.startswith("After") else [0.0, 1.0] for text in texts
            ]

    embedder = OrthogonalEmbedder()
    passages = [
        "After drying the latch, tighten the clamp.",
        "After heating the sleeve, fold the wire.",
    ]
    answer = "Tighten the clamp after drying the latch. Fold the wire after heating the sleeve."
    config = CitationConfig.model_validate(
        {"paraphrase_support": True, "supported_answer_coverage": 0.9}
    )
    spans = align_citations(
        answer, passages, config=config, backend=backend, embedder=embedder
    )
    assert [span.status for span in spans] == ["partial", "partial"]
    assert [span.citations[0].components["embedding_similarity"] for span in spans] == [
        0.0,
        0.0,
    ]
    assert len(embedder.batches) == 2
    assert set(embedder.batches[0]) == set(passages)
    assert embedder.batches[1] == tuple(span.answer_span.text for span in spans)


def test_paraphrase_gate_respects_stricter_content_floor() -> None:
    [span] = align_citations(
        "Tighten the clamp once drying the latch is complete.",
        ["After drying the latch, tighten the clamp."],
        config=CitationConfig.model_validate(
            {
                "paraphrase_support": True,
                "supported_answer_coverage": 0.9,
                "paraphrase_lexical_content_coverage": 0.85,
            }
        ),
    )
    assert span.status == "partial"
    assert span.citations[0].components["content_coverage"] == pytest.approx(0.8)
    assert span.citations[0].components["content_gate"] == 0.0


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("rust", marks=requires_rust)]
)
def test_promoted_citation_retains_parent_identity_and_offsets(
    backend: Literal["python", "rust"],
) -> None:
    preceding = "The tray has copper feet."
    evidence = "After drying the latch, tighten the clamp."
    full_text = preceding + " " + evidence
    chunks = [
        SourceChunk(
            source_id="manual",
            source_index=0,
            document_text=full_text,
            text=chunk,
            doc_char_start=start,
            doc_char_end=start + len(chunk),
        )
        for start, chunk in [(0, preceding), (len(preceding) + 1, evidence)]
    ]
    [span] = align_citations(
        "Tighten the clamp after drying the latch.",
        chunks,
        config=CitationConfig(paraphrase_support=True, supported_answer_coverage=0.9),
        backend=backend,
    )
    assert span.status == "supported"
    [citation] = span.citations
    assert citation.components["content_gate"] == 1.0
    assert citation.source_id == "manual"
    assert citation.source_index == 0
    assert citation.char_start == len(preceding) + 1
    assert citation.char_end == len(full_text)
    assert (
        full_text[citation.char_start : citation.char_end]
        == citation.evidence
        == evidence
    )


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("rust", marks=requires_rust)]
)
@pytest.mark.parametrize("malformed", ["missing", "extra", "ragged", "non-numeric"])
def test_paraphrase_gate_rejects_malformed_answer_batches(
    backend: Literal["python", "rust"], malformed: str
) -> None:
    class MalformedEmbedder:
        def encode(self, texts: Sequence[str]) -> list[list[float]]:
            if texts[0].startswith("After"):
                return [[1.0, 0.0] for _ in texts]
            if malformed == "missing":
                return [[1.0, 0.0]]
            if malformed == "extra":
                return [[1.0, 0.0] for _ in range(len(texts) + 1)]
            if malformed == "ragged":
                return [cast("list[float]", [[1.0], [0.0, 1.0]]) for _ in texts]
            return [cast("list[float]", ["bad", "values"]) for _ in texts]

    spans = align_citations(
        "Tighten the clamp after drying the latch. Fold the wire after heating the sleeve.",
        [
            "After drying the latch, tighten the clamp.",
            "After heating the sleeve, fold the wire.",
        ],
        config=CitationConfig(paraphrase_support=True, supported_answer_coverage=0.9),
        backend=backend,
        embedder=MalformedEmbedder(),
    )
    assert [span.status for span in spans] == ["partial", "partial"]
    for span in spans:
        assert span.citations[0].components["embedding_similarity"] == 0.0
        assert span.citations[0].components["content_gate"] == 0.0


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("rust", marks=requires_rust)]
)
def test_paraphrase_gate_normalizes_decimal_digits(
    backend: Literal["python", "rust"],
) -> None:
    [span] = align_citations(
        "Set the timer to ３０ seconds after closing the hatch.",
        ["After closing the hatch, set the timer to 30 seconds."],
        config=CitationConfig(paraphrase_support=True, supported_answer_coverage=1.0),
        backend=backend,
    )
    assert span.status == "supported"
    assert span.citations[0].components["content_gate"] == 1.0


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("rust", marks=requires_rust)]
)
@pytest.mark.parametrize("modifier", ["R-X", "Rx-Label", "spring-loaded"])
def test_paraphrase_gate_preserves_internal_hyphens(
    backend: Literal["python", "rust"],
    modifier: str,
) -> None:
    [span] = align_citations(
        f"Tighten the {modifier} clamp after drying the latch.",
        [f"After drying the latch, tighten the {modifier} clamp."],
        config=CitationConfig(paraphrase_support=True, supported_answer_coverage=1.0),
        backend=backend,
    )
    assert span.status == "supported"
    assert span.citations[0].components["content_gate"] == 1.0


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("rust", marks=requires_rust)]
)
@pytest.mark.parametrize("linked_phrase", ["against the stop", "using a wrench"])
def test_paraphrase_gate_keeps_a_linked_phrase_in_its_action(
    backend: Literal["python", "rust"], linked_phrase: str
) -> None:
    [span] = align_citations(
        f"Tighten the clamp {linked_phrase} after drying the latch.",
        [f"After drying the latch, tighten the clamp {linked_phrase}."],
        config=CitationConfig(paraphrase_support=True, supported_answer_coverage=1.0),
        backend=backend,
    )
    assert span.status == "supported"
    assert span.citations[0].components["content_gate"] == 1.0


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("rust", marks=requires_rust)]
)
def test_default_setting_as_phrase_belongs_to_the_same_action(
    backend: Literal["python", "rust"],
) -> None:
    [span] = align_citations(
        "Set Pro as the default workspace plan after auditing the settings.",
        ["After auditing the settings, set Pro as the default workspace plan."],
        config=CitationConfig(paraphrase_support=True, supported_answer_coverage=1.0),
        backend=backend,
    )
    assert span.status == "supported"
    assert span.citations[0].components["content_gate"] == 1.0
