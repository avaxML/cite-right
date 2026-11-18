from __future__ import annotations

from typing import Literal, Sequence

import pytest
from pydantic import ValidationError

from cite_right import (
    CitationConfig,
    PossibleSourceMatch,
    PreparedCitationCorpus,
    SimpleTokenizer,
    SourceChunk,
    SourceDocument,
    SpanCitations,
    TokenizedText,
    align_citations,
)
from tests.status_reason_fixtures import (
    CERAMIC_RESTATEMENT,
    GENERIC_SPANS,
    QUANTITY_PASSAGE,
    QUANTITY_TWINS,
    RESTATEMENTS,
    SOURCES,
)

_DIAGNOSTIC_FIELDS = {"status_reason", "contradiction_checks", "possible_source"}
Backend = Literal["python", "rust"]
PathName = Literal["one-shot", "prepared"]


@pytest.fixture(params=["python", "rust"])
def backend(request: pytest.FixtureRequest) -> Backend:
    if request.param == "rust":
        pytest.importorskip("cite_right._core")
        return "rust"
    return "python"


def _config() -> CitationConfig:
    return CitationConfig.strict().model_copy(
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


def _with_diagnostics(config: CitationConfig) -> CitationConfig:
    return CitationConfig.model_validate(
        {**config.model_dump(), "report_possible_source": True}
    )


def _align(
    answer: str,
    sources: Sequence[str | SourceDocument | SourceChunk],
    config: CitationConfig,
    backend: Backend,
    path: PathName,
) -> SpanCitations:
    if path == "prepared":
        [span] = PreparedCitationCorpus.from_sources(sources, config=config).align(
            answer, backend=backend
        )
    else:
        [span] = align_citations(answer, sources, config=config, backend=backend)
    return span


@pytest.mark.parametrize("path", ["one-shot", "prepared"])
@pytest.mark.parametrize(("answer", "source_id", "reason"), RESTATEMENTS)
def test_restatement_source(
    answer: str, source_id: str, reason: str, backend: Backend, path: PathName
) -> None:
    assert 8 <= len(answer.split()) <= 14
    config = _config()
    base = _align(answer, SOURCES, config, backend, path)
    span = _align(answer, SOURCES, _with_diagnostics(config), backend, path)

    assert (
        span.status
        == base.status
        == ("unsupported" if reason == "no_candidate" else "partial")
    )
    assert span.status_reason == reason
    assert span.contradiction_checks == (
        ["has_entity_swap"] if reason == "contradiction" else []
    )
    assert span.model_dump(exclude=_DIAGNOSTIC_FIELDS) == base.model_dump()
    assert span.possible_source is not None
    hint = span.possible_source
    assert hint.source_id == source_id
    assert SOURCES[hint.source_index].id == source_id
    assert (hint.char_start, hint.char_end) == (0, len(SOURCES[hint.source_index].text))
    assert 0.4 <= hint.score <= 1.0
    assert set(hint.model_dump()) == {
        "source_id",
        "source_index",
        "char_start",
        "char_end",
        "score",
    }


@pytest.mark.parametrize("answer", GENERIC_SPANS)
@pytest.mark.parametrize("config", [CitationConfig(), _config()])
def test_generic_spans_have_no_hint(
    answer: str, config: CitationConfig, backend: Backend
) -> None:
    base = _align(answer, SOURCES, config, backend, "one-shot")
    span = _align(answer, SOURCES, _with_diagnostics(config), backend, "one-shot")
    assert span.status_reason in {"below_coverage", "contradiction", "no_candidate"}
    assert span.possible_source is None
    assert span.model_dump(exclude=_DIAGNOSTIC_FIELDS) == base.model_dump()


def test_diagnostics_have_exact_backend_and_entry_point_parity() -> None:
    pytest.importorskip("cite_right._core")
    config = _with_diagnostics(_config())
    prepared = PreparedCitationCorpus.from_sources(SOURCES, config=config)
    assert prepared.rust_corpus is not None
    python_prepared = PreparedCitationCorpus.from_sources(
        SOURCES, config=config, use_rust=False
    )
    answers = [answer for answer, _, _ in RESTATEMENTS] + list(GENERIC_SPANS)
    for answer in answers:
        reference = prepared.align(answer, backend="python")[0]
        assert reference.status_reason is not None
        for result in (
            python_prepared.align(answer, backend="python")[0],
            python_prepared.align(answer, backend="rust")[0],
            prepared.align(answer, backend="rust")[0],
            _align(answer, SOURCES, config, "python", "one-shot"),
            _align(answer, SOURCES, config, "rust", "one-shot"),
        ):
            assert result.model_dump(
                include=_DIAGNOSTIC_FIELDS
            ) == reference.model_dump(include=_DIAGNOSTIC_FIELDS)


@pytest.mark.parametrize("path", ["one-shot", "prepared"])
def test_multiple_contradiction_checks_use_the_full_passage(
    backend: Backend, path: PathName
) -> None:
    answer = "The ceramic filter needs 18 liters of water."
    sources = ["The ceramic filter needs 20 liters of water and does not need soap."]
    base = _align(answer, sources, CitationConfig(), backend, path)
    span = _align(answer, sources, _with_diagnostics(CitationConfig()), backend, path)
    assert span.status == "partial"
    assert span.status_reason == "contradiction"
    assert span.contradiction_checks == ["has_negation_mismatch", "has_number_mismatch"]
    assert span.citations[0].components["answer_coverage"] >= 0.6
    assert span.model_dump(exclude=_DIAGNOSTIC_FIELDS) == base.model_dump()


@pytest.mark.parametrize(("answer", "reason", "checks"), QUANTITY_TWINS)
def test_quantity_and_polarity_twins(
    answer: str, reason: str, checks: tuple[str, ...], backend: Backend
) -> None:
    config = CitationConfig()
    base = _align(answer, [QUANTITY_PASSAGE], config, backend, "one-shot")
    span = _align(
        answer, [QUANTITY_PASSAGE], _with_diagnostics(config), backend, "one-shot"
    )
    assert span.status == ("partial" if checks else "supported")
    assert span.status_reason == reason
    assert span.contradiction_checks == list(checks)
    assert (span.possible_source is not None) is bool(checks)
    assert span.model_dump(exclude=_DIAGNOSTIC_FIELDS) == base.model_dump()


def test_supported_and_no_sources_have_reasons_but_no_hint(backend: Backend) -> None:
    answer = "Disconnect the cable before servicing the chassis."
    config = _with_diagnostics(CitationConfig())
    supported = _align(answer, [answer], config, backend, "one-shot")
    unsupported = _align(answer, [], config, backend, "one-shot")
    for span, status, reason in (
        (supported, "supported", "supported"),
        (unsupported, "unsupported", "no_candidate"),
    ):
        assert span.status == status
        assert span.status_reason == reason
        assert span.contradiction_checks == []
        assert span.possible_source is None
        assert "possible_source" not in span.model_dump()
        assert SpanCitations.model_validate_json(span.model_dump_json()) == span


def test_off_preserves_result_and_config_serialization(backend: Backend) -> None:
    config = CitationConfig(report_possible_source=False)
    span = _align("An unrelated claim.", [], config, backend, "one-shot")
    assert span.status_reason is None
    assert span.contradiction_checks is None
    assert span.possible_source is None
    assert not _DIAGNOSTIC_FIELDS.intersection(span.model_dump())
    assert all(field not in span.model_dump_json() for field in _DIAGNOSTIC_FIELDS)
    assert "report_possible_source" not in config.model_dump()
    assert _with_diagnostics(config).model_dump()["report_possible_source"] is True
    with pytest.raises(ValidationError, match="report_possible_source"):
        CitationConfig(report_possible_source="invalid")  # type: ignore[arg-type]


class _TrackingEmbedder:
    def __init__(self, answer: str, passage: str) -> None:
        self.answer = answer
        self.passage = passage
        self.calls: list[list[str]] = []

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [
            [1.0, 0.0]
            if text == self.answer
            else [0.8, 0.6]
            if text == self.passage
            else [0.0, 1.0]
            for text in texts
        ]


@pytest.mark.parametrize("path", ["one-shot", "prepared"])
def test_hint_uses_existing_cosine_and_absolute_chunk_offsets(
    backend: Backend, path: PathName
) -> None:
    answer = CERAMIC_RESTATEMENT
    passage = SOURCES[-1].text
    chunk = SourceChunk(
        source_id="ceramic",
        text=passage,
        doc_char_start=40,
        doc_char_end=40 + len(passage),
    )
    sources = [SOURCES[0], chunk]
    config = _with_diagnostics(
        CitationConfig(min_answer_coverage=0.95, max_retrieval_support=0)
    )
    without_embedder = _align(answer, sources, config, backend, path)
    assert without_embedder.status == "unsupported"
    assert without_embedder.possible_source is None
    embedder = _TrackingEmbedder(answer, passage)
    if path == "prepared":
        [span] = PreparedCitationCorpus.from_sources(
            sources, config=config, embedder=embedder
        ).align(answer, backend=backend)
    else:
        [span] = align_citations(
            answer, sources, config=config, backend=backend, embedder=embedder
        )

    assert span.status == "unsupported"
    assert span.retrieval_support == []
    assert span.possible_source is not None
    assert span.possible_source.model_dump() == {
        "source_id": "ceramic",
        "source_index": 1,
        "char_start": 40,
        "char_end": 40 + len(passage),
        "score": pytest.approx(0.8),
    }
    assert embedder.calls == [[SOURCES[0].text, passage], [answer]]


@pytest.mark.parametrize(
    ("answer", "passage"),
    [
        (
            "Here is a quick overview.",
            "An overview includes a quick summary of the latest changes.",
        ),
        ("Filter maintenance", "Maintenance on this filter requires goggles."),
        (
            "### Routine equipment maintenance",
            "Maintenance of routine equipment needs two people.",
        ),
        ("Revision: updated recently.", "The revision was updated last month."),
        (
            "Always follow the safety instructions.",
            "Safety instructions accompany the cargo.",
        ),
    ],
)
def test_generic_spans_abstain_even_with_multiple_shared_words(
    answer: str, passage: str, backend: Backend
) -> None:
    config = CitationConfig(supported_answer_coverage=0.9)
    base = _align(answer, [passage], config, backend, "one-shot")
    span = _align(answer, [passage], _with_diagnostics(config), backend, "one-shot")
    assert base.status != "supported"
    assert span.possible_source is None
    assert span.model_dump(exclude=_DIAGNOSTIC_FIELDS) == base.model_dump()


class _LeadingSpaceTokenizer:
    def __init__(self) -> None:
        self._tokenizer = SimpleTokenizer()

    def tokenize(self, text: str) -> TokenizedText:
        tokens = self._tokenizer.tokenize(text)
        spans = [
            (start - 1 if start > 0 and text[start - 1].isspace() else start, end)
            for start, end in tokens.token_spans
        ]
        return TokenizedText(text=text, token_ids=tokens.token_ids, token_spans=spans)


def test_function_word_anchors_skip_tokenizer_whitespace(backend: Backend) -> None:
    [span] = align_citations(
        "Check if the nozzle is open.",
        ["Decide if the valve is leaking."],
        backend=backend,
        config=_with_diagnostics(CitationConfig()),
        tokenizer=_LeadingSpaceTokenizer(),
    )
    assert span.possible_source is None


@pytest.mark.parametrize(
    ("answer", "passage", "expects_hint"),
    [
        pytest.param(
            "Seasonal Ceramic Filter Maintenance",
            "Seasonal maintenance for the ceramic filter requires a clean towel.",
            False,
            id="title-case-heading",
        ),
        pytest.param(
            "Below is a brief summary.",
            "The brief summary is printed below the wiring diagram.",
            False,
            id="summary-preamble",
        ),
        pytest.param(
            "Last revised November 12, 2026.",
            "The November revision was last revised by the assembly team.",
            False,
            id="revision-date",
        ),
        pytest.param(
            "Please read all safety instructions carefully.",
            "Carefully read the safety instructions before each inspection.",
            False,
            id="generic-advice",
        ),
        pytest.param(
            "Ceramic filter cleaning schedule\n--------------------------------",
            "A schedule for cleaning the ceramic filter is posted each month.",
            False,
            id="setext-heading",
        ),
        pytest.param(
            "Below is a brief summary:",
            "The brief summary is printed below the wiring diagram.",
            False,
            id="summary-colon",
        ),
        pytest.param(
            "Last updated November 12, 2026.",
            "The assembly team last updated the November schedule.",
            False,
            id="updated-date",
        ),
        pytest.param(
            "Please carefully read all safety instructions.",
            "Carefully read the safety instructions before each inspection.",
            False,
            id="generic-advice-adverb",
        ),
        pytest.param(
            "Here is the overview panel selected by default.",
            "By default, the overview panel is selected when the unit starts.",
            True,
            id="specific-overview-panel",
        ),
        pytest.param(
            "**Here is a quick overview.**",
            "An overview includes a quick summary of the latest changes.",
            False,
            id="markdown-emphasis",
        ),
        pytest.param(
            "- Below is a brief summary:",
            "The brief summary is printed below the wiring diagram.",
            False,
            id="markdown-list",
        ),
        pytest.param(
            "> Please carefully read all safety instructions.",
            "Carefully read the safety instructions before each inspection.",
            False,
            id="markdown-quote",
        ),
        pytest.param(
            "- [ ] Always follow the safety instructions.",
            "Safety instructions accompany the cargo.",
            False,
            id="markdown-task-list",
        ),
        pytest.param(
            "> Ceramic filter cleaning schedule\n> --------------------------------",
            "A schedule for cleaning the ceramic filter is posted each month.",
            False,
            id="markdown-quoted-setext",
        ),
        pytest.param(
            "Here is a summary of the overview: rotate the amber dial before lifting the lid.",
            "Before lifting the lid, turn the amber dial.",
            True,
            id="specific-claim-after-colon",
        ),
    ],
)
def test_hint_distinguishes_boilerplate_from_a_specific_claim(
    answer: str, passage: str, expects_hint: bool, backend: Backend
) -> None:
    config = _with_diagnostics(CitationConfig(min_answer_coverage=0.95))
    embedder = _TrackingEmbedder(answer, passage)
    [span] = align_citations(
        answer, [passage], backend=backend, config=config, embedder=embedder
    )
    assert span.status != "supported"
    assert (span.possible_source is not None) is expects_hint
    assert embedder.calls == [[passage], [answer]]


@pytest.mark.parametrize(
    ("start", "end", "prefix"),
    [(5, 10, ""), (-5, 70, ""), (-5, 140, "The gauge flashes twice. ")],
)
def test_hint_respects_declared_chunk_bounds(
    backend: Backend, start: int, end: int, prefix: str
) -> None:
    answer = CERAMIC_RESTATEMENT
    source = SourceChunk(
        source_id="ceramic",
        text=prefix + SOURCES[-1].text,
        doc_char_start=start,
        doc_char_end=end,
    )
    config = CitationConfig(min_alignment_score=100)
    # The native prepared builder expects a non-negative chunk start.
    base = PreparedCitationCorpus.from_sources(
        [source], config=config, use_rust=start >= 0
    ).align(answer, backend=backend)[0]
    span = PreparedCitationCorpus.from_sources(
        [source], config=_with_diagnostics(config), use_rust=start >= 0
    ).align(answer, backend=backend)[0]
    assert span.status_reason == "no_candidate"
    assert span.model_dump(exclude=_DIAGNOSTIC_FIELDS) == base.model_dump()
    assert span.possible_source is None


@pytest.mark.parametrize("prefix", ["2. ", "> 2. "])
def test_ordered_checkbox_abstains_after_answer_segmentation(
    backend: Backend, prefix: str
) -> None:
    advice = "[X] Always follow the safety instructions."
    answer = prefix + advice
    sources = ["Safety instructions accompany the cargo."]
    config = CitationConfig(supported_answer_coverage=0.95)
    base = align_citations(answer, sources, config=config, backend=backend)
    spans = align_citations(
        answer, sources, config=_with_diagnostics(config), backend=backend
    )
    assert [span.answer_span.text for span in spans] == [prefix.strip(), advice]
    assert [span.model_dump(exclude=_DIAGNOSTIC_FIELDS) for span in spans] == [
        span.model_dump() for span in base
    ]
    assert all(span.possible_source is None for span in spans)


@pytest.mark.parametrize(("start", "end"), [(-1, 5), (5, 2), (5, 5)])
def test_possible_source_requires_a_valid_range(start: int, end: int) -> None:
    with pytest.raises(ValidationError, match="non-negative, non-empty range"):
        PossibleSourceMatch(
            source_id="ceramic",
            source_index=0,
            char_start=start,
            char_end=end,
            score=0.8,
        )
