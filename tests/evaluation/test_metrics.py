from __future__ import annotations

from typing import Any, get_args

import pytest
from evaluation.metrics import (
    CaseMetricRecord,
    MetricReport,
    Rate,
    StatusConfusionMatrix,
    StatusCountRow,
    StatusLabel,
    aggregate_metrics,
)
from pydantic import ValidationError


def test_status_label_and_models_are_frozen_contracts() -> None:
    assert get_args(StatusLabel) == ("supported", "partial", "unsupported")

    rate = Rate(
        numerator=5,
        denominator=8,
        estimate=0.625,
        lower=0.3057423946026273,
        upper=0.8631557141764027,
    )
    record = _record()

    with pytest.raises((ValidationError, TypeError, AttributeError)):
        rate.numerator = 4  # type: ignore[misc]

    with pytest.raises((ValidationError, TypeError, AttributeError)):
        record.exact_true_positives = 1  # type: ignore[misc]


def test_case_metric_record_requires_one_rank_per_eligible_claim_and_positive_ranks() -> None:
    with pytest.raises(
        ValidationError,
        match="retrieval_ranks must contain one entry per retrieval-eligible claim",
    ):
        _record(
            retrieval_eligible_claim_count=2,
            retrieval_ranks=(1,),
        )

    with pytest.raises(
        ValidationError,
        match="retrieval ranks must be positive integers or None",
    ):
        _record(
            retrieval_eligible_claim_count=1,
            retrieval_ranks=(0,),
        )


def test_aggregate_metrics_returns_hand_calculated_raw_count_metrics() -> None:
    report = aggregate_metrics(_hand_calculated_records())

    assert isinstance(report, MetricReport)
    assert report.record_count == 4
    assert report.evaluator_error_count == 1

    # Exact precision must use pooled raw counts TP / (TP + FP), never the mean
    # of per-case precision values.
    _assert_rate(
        report.exact_precision,
        numerator=5,
        denominator=8,
        estimate=0.625,
        lower=0.3057423946026273,
        upper=0.8631557141764027,
    )
    _assert_rate(
        report.exact_recall,
        numerator=5,
        denominator=8,
        estimate=0.625,
        lower=0.3057423946026273,
        upper=0.8631557141764027,
    )
    _assert_rate(
        report.requirement_recall,
        numerator=5,
        denominator=8,
        estimate=0.625,
        lower=0.3057423946026273,
        upper=0.8631557141764027,
    )
    _assert_rate(
        report.fully_attributed_claim_recall,
        numerator=2,
        denominator=4,
        estimate=0.5,
        lower=0.15003898915214953,
        upper=0.8499610108478505,
    )
    _assert_rate(
        report.source_accuracy,
        numerator=5,
        denominator=8,
        estimate=0.625,
        lower=0.3057423946026273,
        upper=0.8631557141764027,
    )
    _assert_rate(
        report.offset_validity,
        numerator=6,
        denominator=8,
        estimate=0.75,
        lower=0.40927543031016883,
        upper=0.9285207872478909,
    )
    _assert_rate(
        report.recall_at_0_9,
        numerator=7,
        denominator=8,
        estimate=0.875,
        lower=0.5291118177871464,
        upper=0.9775825085499433,
    )
    _assert_rate(
        report.recall_at_0_5,
        numerator=8,
        denominator=8,
        estimate=1.0,
        lower=0.6755924351161198,
        upper=1.0,
    )
    _assert_rate(
        report.multi_span_precision,
        numerator=2,
        denominator=4,
        estimate=0.5,
        lower=0.15003898915214953,
        upper=0.8499610108478505,
    )
    _assert_rate(
        report.multi_span_recall,
        numerator=2,
        denominator=4,
        estimate=0.5,
        lower=0.15003898915214953,
        upper=0.8499610108478505,
    )
    _assert_rate(
        report.contradiction_false_citation_rate,
        numerator=2,
        denominator=4,
        estimate=0.5,
        lower=0.15003898915214953,
        upper=0.8499610108478505,
    )

    assert report.status_confusion_matrix == StatusConfusionMatrix(
        supported=StatusCountRow(supported=1, partial=0, unsupported=0),
        partial=StatusCountRow(supported=0, partial=1, unsupported=1),
        unsupported=StatusCountRow(supported=0, partial=1, unsupported=0),
    )
    assert report.status_macro_f1 == pytest.approx(0.5)

    # None ranks are eligible misses: they count in the denominator, contribute
    # zero to recall@k, and zero reciprocal rank to MRR.
    _assert_rate(
        report.retrieval_recall_at_1,
        numerator=1,
        denominator=4,
        estimate=0.25,
        lower=0.04558726080970055,
        upper=0.6993581574175981,
    )
    _assert_rate(
        report.retrieval_recall_at_3,
        numerator=2,
        denominator=4,
        estimate=0.5,
        lower=0.15003898915214953,
        upper=0.8499610108478505,
    )
    _assert_rate(
        report.retrieval_recall_at_5,
        numerator=2,
        denominator=4,
        estimate=0.5,
        lower=0.15003898915214953,
        upper=0.8499610108478505,
    )
    assert report.retrieval_mrr == pytest.approx(1 / 3)


def test_aggregate_metrics_returns_none_for_zero_denominator_rates() -> None:
    report = aggregate_metrics((_record(),))

    assert report.record_count == 1
    assert report.evaluator_error_count == 0

    assert report.exact_precision == _zero_rate()
    assert report.exact_recall == _zero_rate()
    assert report.requirement_recall == _zero_rate()
    assert report.fully_attributed_claim_recall == _zero_rate()
    assert report.source_accuracy == _zero_rate()
    assert report.offset_validity == _zero_rate()
    assert report.recall_at_0_9 == _zero_rate()
    assert report.recall_at_0_5 == _zero_rate()
    assert report.multi_span_precision == _zero_rate()
    assert report.multi_span_recall == _zero_rate()
    assert report.contradiction_false_citation_rate == _zero_rate()
    assert report.retrieval_recall_at_1 == _zero_rate()
    assert report.retrieval_recall_at_3 == _zero_rate()
    assert report.retrieval_recall_at_5 == _zero_rate()


def test_aggregate_metrics_is_permutation_invariant() -> None:
    records = _hand_calculated_records()

    assert aggregate_metrics(records) == aggregate_metrics(tuple(reversed(records)))


def _hand_calculated_records() -> tuple[CaseMetricRecord, ...]:
    return (
        _record(
            expected_status="supported",
            observed_status="supported",
            exact_true_positives=2,
            exact_false_positives=1,
            exact_false_negatives=1,
            recall_at_0_9_true_positives=3,
            recall_at_0_9_false_negatives=0,
            recall_at_0_5_true_positives=3,
            recall_at_0_5_false_negatives=0,
            requirement_count=3,
            matched_requirement_count=2,
            entailed_claim_count=2,
            fully_attributed_claim_count=1,
            source_selection_attempt_count=3,
            source_selection_correct_count=2,
            emitted_citation_count=3,
            valid_offset_count=2,
            multi_span_true_positives=1,
            multi_span_false_positives=0,
            multi_span_false_negatives=1,
            contradicted_claim_count=0,
            contradicted_claim_citation_count=0,
            retrieval_eligible_claim_count=2,
            retrieval_ranks=(1, None),
        ),
        _record(
            expected_status="partial",
            observed_status="unsupported",
            exact_true_positives=1,
            exact_false_positives=0,
            exact_false_negatives=2,
            recall_at_0_9_true_positives=2,
            recall_at_0_9_false_negatives=1,
            recall_at_0_5_true_positives=3,
            recall_at_0_5_false_negatives=0,
            requirement_count=3,
            matched_requirement_count=1,
            entailed_claim_count=1,
            fully_attributed_claim_count=0,
            source_selection_attempt_count=1,
            source_selection_correct_count=1,
            emitted_citation_count=1,
            valid_offset_count=1,
            multi_span_true_positives=0,
            multi_span_false_positives=1,
            multi_span_false_negatives=1,
            contradicted_claim_count=2,
            contradicted_claim_citation_count=1,
            retrieval_eligible_claim_count=1,
            retrieval_ranks=(3,),
            evaluator_error=True,
        ),
        _record(
            expected_status="unsupported",
            observed_status="partial",
            exact_true_positives=0,
            exact_false_positives=2,
            exact_false_negatives=0,
            recall_at_0_9_true_positives=0,
            recall_at_0_9_false_negatives=0,
            recall_at_0_5_true_positives=0,
            recall_at_0_5_false_negatives=0,
            requirement_count=0,
            matched_requirement_count=0,
            entailed_claim_count=0,
            fully_attributed_claim_count=0,
            source_selection_attempt_count=2,
            source_selection_correct_count=0,
            emitted_citation_count=2,
            valid_offset_count=1,
            multi_span_true_positives=0,
            multi_span_false_positives=1,
            multi_span_false_negatives=0,
            contradicted_claim_count=1,
            contradicted_claim_citation_count=1,
            retrieval_eligible_claim_count=1,
            retrieval_ranks=(None,),
        ),
        _record(
            expected_status="partial",
            observed_status="partial",
            exact_true_positives=2,
            exact_false_positives=0,
            exact_false_negatives=0,
            recall_at_0_9_true_positives=2,
            recall_at_0_9_false_negatives=0,
            recall_at_0_5_true_positives=2,
            recall_at_0_5_false_negatives=0,
            requirement_count=2,
            matched_requirement_count=2,
            entailed_claim_count=1,
            fully_attributed_claim_count=1,
            source_selection_attempt_count=2,
            source_selection_correct_count=2,
            emitted_citation_count=2,
            valid_offset_count=2,
            multi_span_true_positives=1,
            multi_span_false_positives=0,
            multi_span_false_negatives=0,
            contradicted_claim_count=1,
            contradicted_claim_citation_count=0,
            retrieval_eligible_claim_count=0,
            retrieval_ranks=(),
        ),
    )


def _record(**overrides: Any) -> CaseMetricRecord:
    payload = {
        "expected_status": "unsupported",
        "observed_status": "unsupported",
        "exact_true_positives": 0,
        "exact_false_positives": 0,
        "exact_false_negatives": 0,
        "recall_at_0_9_true_positives": 0,
        "recall_at_0_9_false_negatives": 0,
        "recall_at_0_5_true_positives": 0,
        "recall_at_0_5_false_negatives": 0,
        "requirement_count": 0,
        "matched_requirement_count": 0,
        "entailed_claim_count": 0,
        "fully_attributed_claim_count": 0,
        "source_selection_attempt_count": 0,
        "source_selection_correct_count": 0,
        "emitted_citation_count": 0,
        "valid_offset_count": 0,
        "multi_span_true_positives": 0,
        "multi_span_false_positives": 0,
        "multi_span_false_negatives": 0,
        "contradicted_claim_count": 0,
        "contradicted_claim_citation_count": 0,
        "retrieval_eligible_claim_count": 0,
        "retrieval_ranks": (),
        "evaluator_error": False,
    }
    payload.update(overrides)
    return CaseMetricRecord.model_validate(payload)


def _assert_rate(
    rate: Rate,
    *,
    numerator: int,
    denominator: int,
    estimate: float | None,
    lower: float | None,
    upper: float | None,
) -> None:
    assert rate.numerator == numerator
    assert rate.denominator == denominator
    if estimate is None:
        assert rate.estimate is None
        assert rate.lower is None
        assert rate.upper is None
        return
    assert rate.estimate == pytest.approx(estimate)
    assert rate.lower == pytest.approx(lower)
    assert rate.upper == pytest.approx(upper)


def _zero_rate() -> Rate:
    return Rate(
        numerator=0,
        denominator=0,
        estimate=None,
        lower=None,
        upper=None,
    )
