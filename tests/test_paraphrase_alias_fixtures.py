"""Caller-declared matches and precision controls at the deployed threshold."""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from cite_right import CitationConfig, PreparedCitationCorpus

from .conftest import requires_rust
from .paraphrase_fixtures import (
    ALIASES,
    BASE_POSITIVE_STATUSES,
    DEPLOYMENT_GUARDS,
    LONG_POLISH_TWIN,
    PASSAGES,
    REALISTIC_RESTATEMENTS,
    RESTATEMENTS,
    Restatement,
)

FIXTURES = RESTATEMENTS + REALISTIC_RESTATEMENTS


class FixtureEmbedder:
    def __init__(self, fixture: Restatement, index: int) -> None:
        self.fixture = fixture
        self.index = index

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * (2 * len(FIXTURES) + 1)
            if text == self.fixture.passage:
                vector[2 * self.index] = 1.0
            elif text in (self.fixture.positive, *self.fixture.twins):
                vector[2 * self.index] = 0.8
                vector[2 * self.index + 1] = 0.6
            else:
                vector[-1] = 1.0
            vectors.append(vector)
        return vectors


def deploy_config(aliases: tuple[tuple[str, ...], ...] = ()) -> CitationConfig:
    return CitationConfig.model_validate(
        {
            **CitationConfig.strict().model_dump(),
            "top_k": 1,
            "max_citations_per_source": 1,
            "require_all_answer_tokens_in_evidence": False,
            "window_size_sentences": 2,
            "window_stride_sentences": 1,
            "multi_span_evidence": True,
            "prefer_source_order": True,
            "supported_answer_coverage": 0.75,
            "term_aliases": aliases,
        }
    )


@pytest.mark.parametrize(
    "index,fixture",
    [pytest.param(i, case, id=case.name) for i, case in enumerate(FIXTURES)],
)
@pytest.mark.parametrize("embedded", [False, True], ids=["lexical", "embedded"])
@requires_rust
def test_alias_fixture_paths(
    index: int, fixture: Restatement, embedded: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cite_right import _core

    calls = []
    prepared_build = _core.PreparedCorpus.build_citations
    legacy_build = _core.rust_build_citations_fast

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

    distractor = "Store spare gaskets in labelled drawers."
    embedder = FixtureEmbedder(fixture, index) if embedded else None
    for aliases in ((), fixture.aliases or ALIASES):
        results = []
        calls.clear()
        for use_rust, backend in (
            (False, "python"),
            (False, "rust"),
            (True, "python"),
            (True, "rust"),
        ):
            corpus = PreparedCitationCorpus.from_sources(
                [distractor, fixture.passage],
                config=deploy_config(aliases),
                embedder=embedder,
                use_rust=use_rust,
            )
            assert (corpus.rust_corpus is not None) == use_rust
            observed = []
            for answer in (fixture.positive, *fixture.twins):
                [span] = corpus.align(answer, backend=backend)
                for citation in span.citations:
                    assert citation.source_index == 1
                    assert (
                        citation.evidence
                        == fixture.passage[citation.char_start : citation.char_end]
                    )
                    assert citation.components["embedding_score"] == pytest.approx(
                        0.8 if embedded else 0.0
                    )
                observed.append(
                    (
                        span.status,
                        [
                            (
                                c.source_index,
                                c.components["answer_coverage"],
                                c.components["content_coverage"],
                            )
                            for c in span.citations
                        ],
                    )
                )
            if aliases:
                assert observed[0][0] == "supported"
            else:
                assert observed[0][0] == BASE_POSITIVE_STATUSES[fixture.name]
            for twin, (status, _) in zip(fixture.twins, observed[1:], strict=True):
                assert status != "supported", twin
            results.append(observed)
        assert results[1:] == [results[0]] * 3
        assert calls == (
            ["legacy"] * (1 + len(fixture.twins))
            + ["prepared"] * (1 + len(fixture.twins))
        )


def test_long_polishing_claim_with_added_soft_modifier() -> None:
    corpus = PreparedCitationCorpus.from_sources(
        [PASSAGES[3]], use_rust=False, config=deploy_config(ALIASES)
    )
    [span] = corpus.align(LONG_POLISH_TWIN, backend="python")
    assert span.status != "supported"


@pytest.mark.parametrize(
    "answer,passages",
    [
        pytest.param(answer, passages, id=name)
        for name, answer, passages in DEPLOYMENT_GUARDS
    ],
)
@requires_rust
def test_original_guards_at_deployment_threshold(
    answer: str, passages: tuple[str, ...]
) -> None:
    statuses = []
    for use_rust, backend in (
        (False, "python"),
        (False, "rust"),
        (True, "python"),
        (True, "rust"),
    ):
        corpus = PreparedCitationCorpus.from_sources(
            passages, config=deploy_config(ALIASES), use_rust=use_rust
        )
        [span] = corpus.align(answer, backend=backend)
        statuses.append(span.status)
    assert statuses == ["partial"] * 4


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="The contradictory before/after passage still outranks the agreeing passage",
)
def test_wrong_source_ranks_agreeing_passage_first() -> None:
    _, answer, passages = next(
        case for case in DEPLOYMENT_GUARDS if case[0] == "wrong-source"
    )
    corpus = PreparedCitationCorpus.from_sources(
        passages, config=deploy_config(ALIASES), use_rust=False
    )
    [span] = corpus.align(answer, backend="python")
    assert span.citations[0].source_index == 1
