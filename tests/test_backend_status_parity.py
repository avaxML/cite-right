"""Content coverage and admission across the native and Python builders."""

from typing import Any

import pytest

from cite_right import CitationConfig, PreparedCitationCorpus, SourceDocument
from cite_right.core.results import SpanCitations, TokenizedText
from cite_right.text.tokenizer import SimpleTokenizer

from .conftest import requires_rust
from .paraphrase_fixtures import ANSWERS, PASSAGES
from .test_paraphrase_support import (
    _BNP_ANSWER,
    _BNP_SOURCES,
    _FARAGE_ANSWER,
    _FARAGE_SOURCE,
)


class ProtocolTokenizer:
    def __init__(self) -> None:
        self.tokenizer = SimpleTokenizer()

    def tokenize(self, text: str) -> TokenizedText:
        return self.tokenizer.tokenize(text)


STRICT = CitationConfig.strict().model_copy(
    update={
        "require_all_answer_tokens_in_evidence": False,
        "top_k": 1,
        "window_size_sentences": 2,
        "multi_span_evidence": True,
        "supported_answer_coverage": 0.75,
    }
)
REORDERED = "Dry the latch after rinsing the tray before packing the sleeve."
MANUAL = "Before packing the sleeve, rinse the tray and dry the latch."


@requires_rust
@pytest.mark.parametrize(
    ("answer", "sources", "config"),
    [
        pytest.param(REORDERED, [MANUAL], STRICT, id="below-sequential-minimum"),
        pytest.param(
            "And the hinge opens safely.",
            ["The hinge opens."],
            CitationConfig(),
            id="query-only-stopword",
        ),
        pytest.param(
            "Here is the overview.",
            ["The hinge opens."],
            CitationConfig(),
            id="stopword-only",
        ),
        pytest.param(" ".join(ANSWERS), list(PASSAGES), STRICT, id="manual-replay"),
        pytest.param(
            _BNP_ANSWER, _BNP_SOURCES, CitationConfig(top_k=1), id="issue49-bnp"
        ),
        pytest.param(
            _FARAGE_ANSWER, [_FARAGE_SOURCE], CitationConfig(), id="issue49-refusal"
        ),
    ],
)
def test_builders_agree(
    answer: str, sources: list[str | SourceDocument], config: CitationConfig
) -> None:
    python = PreparedCitationCorpus.from_sources(sources, config=config, use_rust=False)
    rust = PreparedCitationCorpus.from_sources(sources, config=config, use_rust=True)
    custom = PreparedCitationCorpus.from_sources(
        sources, config=config, tokenizer=ProtocolTokenizer()
    )
    assert python.rust_corpus is None
    assert rust.rust_corpus is not None
    assert custom.rust_corpus is None
    expected = python.align(answer, backend="python")
    for candidate in (
        python.align(answer, backend="rust"),
        rust.align(answer, backend="python"),
        rust.align(answer, backend="rust"),
        custom.align(answer, backend="python"),
        custom.align(answer, backend="rust"),
    ):
        assert [span.status for span in candidate] == [span.status for span in expected]
        for actual_span, expected_span in zip(candidate, expected, strict=True):
            assert_citation_coverage(actual_span, expected_span)


def assert_citation_coverage(actual: SpanCitations, expected: SpanCitations) -> None:
    assert [c.source_id for c in actual.citations] == [
        c.source_id for c in expected.citations
    ]
    for left, right in zip(actual.citations, expected.citations, strict=True):
        for key in ("answer_coverage", "content_coverage"):
            assert left.components[key] == pytest.approx(
                right.components[key], abs=1e-9
            )


def test_protocol_tokenizer_covers_content_words() -> None:
    corpus = PreparedCitationCorpus.from_sources(
        ["The hinge opens."], tokenizer=ProtocolTokenizer()
    )
    [span] = corpus.align("And the hinge opens safely.", backend="python")
    assert span.status == "supported"
    [citation] = span.citations
    assert citation.components["content_coverage"] == pytest.approx(2 / 3)


@requires_rust
def test_native_builders_admit_reordered_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    from cite_right import _core

    prepared = PreparedCitationCorpus.from_sources([MANUAL], config=STRICT)
    legacy = PreparedCitationCorpus.from_sources(
        [MANUAL], config=STRICT, tokenizer=ProtocolTokenizer()
    )
    prepared_build = _core.PreparedCorpus.build_citations
    legacy_build = _core.rust_build_citations_fast
    calls: list[str] = []

    def track_prepared(*args: Any, **kwargs: Any) -> _core.PyCitationResult:
        result = prepared_build(*args, **kwargs)
        calls.append("prepared")
        return result

    def track_legacy(*args: Any, **kwargs: Any) -> str:
        result = legacy_build(*args, **kwargs)
        calls.append("legacy")
        return result

    monkeypatch.setattr(_core.PreparedCorpus, "build_citations", track_prepared)
    monkeypatch.setattr(_core, "rust_build_citations_fast", track_legacy)

    for corpus in (prepared, legacy):
        [span] = corpus.align(REORDERED, backend="rust")
        assert span.status == "partial"
        [citation] = span.citations
        assert citation.components["answer_coverage"] == pytest.approx(4 / 11)
        assert citation.components["content_coverage"] == pytest.approx(5 / 6)

    assert calls == ["prepared", "legacy"]


class PunctuationTokenizer(ProtocolTokenizer):
    def tokenize(self, text: str) -> TokenizedText:
        tokens = super().tokenize(text)
        if not text.endswith("."):
            return tokens
        return TokenizedText(
            text=text,
            token_ids=[*tokens.token_ids, 0],
            token_spans=[*tokens.token_spans, (len(text) - 1, len(text))],
        )


@pytest.mark.parametrize(
    "backend", ["python", pytest.param("rust", marks=requires_rust)]
)
def test_punctuation_only_overlap_does_not_admit_citation(backend: str) -> None:
    corpus = PreparedCitationCorpus.from_sources(
        ["Copper glows."], tokenizer=PunctuationTokenizer()
    )
    [span] = corpus.align("Zinc bends.", backend=backend)
    assert span.status == "unsupported"
    assert not span.citations
