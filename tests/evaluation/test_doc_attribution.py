from __future__ import annotations

from collections import Counter

import pytest

from evaluation.doc_attribution import (
    CONFIGS,
    Case,
    cite_right_predictor,
    load_cases,
    run_cases,
    summarize,
)


def test_cases_are_internally_consistent() -> None:
    documents, cases = load_cases()

    assert len({case.id for case in cases}) == len(cases)
    for case in cases:
        assert case.answer.strip(), case.id
        assert case.split in {"train", "dev"}, case.id
        assert set(case.docs) <= set(documents), case.id
        if case.label == "must_not_cite":
            assert case.docs == (), case.id
        else:
            assert case.docs, case.id
    counts = Counter((case.split, case.label) for case in cases)
    for split in ("train", "dev"):
        assert counts[(split, "must_cite")] >= 45
        assert counts[(split, "must_not_cite")] >= 12


def test_grader_scores_oracle_null_and_wrong_document_predictors() -> None:
    documents, cases = load_cases()

    def oracle(case: Case) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return (case.docs if case.label == "must_cite" else ()), ()

    def null(case: Case) -> tuple[tuple[str, ...], tuple[str, ...]]:
        del case
        return (), ()

    def wrong_document(case: Case) -> tuple[tuple[str, ...], tuple[str, ...]]:
        other = sorted(set(documents) - set(case.docs))[:1]
        return tuple(other), ()

    oracle_summary = summarize(run_cases(cases, oracle))
    null_summary = summarize(run_cases(cases, null))
    wrong_summary = summarize(run_cases(cases, wrong_document))

    assert (oracle_summary.wrong, oracle_summary.recall) == (0, 1.0)
    assert (null_summary.wrong, null_summary.recall) == (0, 0.0)
    assert null_summary.must_not_cite_clean == null_summary.must_not_cite
    assert (wrong_summary.wrong, wrong_summary.recall) == (len(cases), 0.0)


@pytest.mark.parametrize("config_name", sorted(CONFIGS))
def test_no_config_cites_an_unsupported_claim(config_name: str) -> None:
    # Swapped numbers, swapped entities, and contradictions come back
    # ``partial``; only ``supported`` spans may cite.
    documents, cases = load_cases()
    negatives = tuple(case for case in cases if case.label == "must_not_cite")

    results = run_cases(
        negatives, cite_right_predictor(documents, config=CONFIGS[config_name]())
    )

    assert [r.id for r in results if r.cited] == []
