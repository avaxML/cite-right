"""Deterministic aggregate metrics for strict attribution evaluation."""

from __future__ import annotations

from math import sqrt
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

STRICT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
StatusLabel = Literal["supported", "partial", "unsupported"]
WILSON_95_Z = 1.959963984540054
_STATUS_LABELS: tuple[StatusLabel, ...] = ("supported", "partial", "unsupported")


class Rate(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    numerator: int
    denominator: int
    estimate: float | None
    lower: float | None
    upper: float | None

    @model_validator(mode="after")
    def _validate_rate(self) -> Rate:
        _validate_nonnegative(self.numerator, "rate numerator must be non-negative")
        _validate_nonnegative(
            self.denominator, "rate denominator must be non-negative"
        )
        if self.numerator > self.denominator:
            raise ValueError("rate numerator must not exceed denominator")

        if self.denominator == 0:
            if any(value is not None for value in (self.estimate, self.lower, self.upper)):
                raise ValueError(
                    "zero-denominator rates must set estimate and interval bounds to None"
                )
            return self

        if any(value is None for value in (self.estimate, self.lower, self.upper)):
            raise ValueError(
                "non-zero-denominator rates must define estimate and interval bounds"
            )

        assert self.estimate is not None
        assert self.lower is not None
        assert self.upper is not None
        for value, name in (
            (self.estimate, "rate estimate"),
            (self.lower, "rate lower bound"),
            (self.upper, "rate upper bound"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must stay within [0, 1]")
        if self.lower > self.estimate or self.estimate > self.upper:
            raise ValueError("rate bounds must satisfy lower <= estimate <= upper")
        return self


class StatusCountRow(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    supported: int
    partial: int
    unsupported: int

    @model_validator(mode="after")
    def _validate_counts(self) -> StatusCountRow:
        _validate_nonnegative(self.supported, "status counts must be non-negative")
        _validate_nonnegative(self.partial, "status counts must be non-negative")
        _validate_nonnegative(self.unsupported, "status counts must be non-negative")
        return self


class StatusConfusionMatrix(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    supported: StatusCountRow
    partial: StatusCountRow
    unsupported: StatusCountRow


class CaseMetricRecord(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    expected_status: StatusLabel
    observed_status: StatusLabel
    exact_true_positives: int
    exact_false_positives: int
    exact_false_negatives: int
    recall_at_0_9_true_positives: int
    recall_at_0_9_false_negatives: int
    recall_at_0_5_true_positives: int
    recall_at_0_5_false_negatives: int
    requirement_count: int
    matched_requirement_count: int
    entailed_claim_count: int
    fully_attributed_claim_count: int
    source_selection_attempt_count: int
    source_selection_correct_count: int
    emitted_citation_count: int
    valid_offset_count: int
    multi_span_true_positives: int
    multi_span_false_positives: int
    multi_span_false_negatives: int
    contradicted_claim_count: int
    contradicted_claim_citation_count: int
    retrieval_eligible_claim_count: int
    retrieval_ranks: tuple[int | None, ...]
    evaluator_error: bool = False

    @model_validator(mode="after")
    def _validate_counts(self) -> CaseMetricRecord:
        for field_name in (
            "exact_true_positives",
            "exact_false_positives",
            "exact_false_negatives",
            "recall_at_0_9_true_positives",
            "recall_at_0_9_false_negatives",
            "recall_at_0_5_true_positives",
            "recall_at_0_5_false_negatives",
            "requirement_count",
            "matched_requirement_count",
            "entailed_claim_count",
            "fully_attributed_claim_count",
            "source_selection_attempt_count",
            "source_selection_correct_count",
            "emitted_citation_count",
            "valid_offset_count",
            "multi_span_true_positives",
            "multi_span_false_positives",
            "multi_span_false_negatives",
            "contradicted_claim_count",
            "contradicted_claim_citation_count",
            "retrieval_eligible_claim_count",
        ):
            _validate_nonnegative(
                getattr(self, field_name),
                f"{field_name} must be non-negative",
            )

        _validate_not_greater_than(
            self.matched_requirement_count,
            self.requirement_count,
            "matched requirement count must not exceed requirement count",
        )
        _validate_not_greater_than(
            self.fully_attributed_claim_count,
            self.entailed_claim_count,
            "fully attributed claim count must not exceed entailed claim count",
        )
        _validate_not_greater_than(
            self.source_selection_correct_count,
            self.source_selection_attempt_count,
            "source selection correct count must not exceed attempt count",
        )
        _validate_not_greater_than(
            self.valid_offset_count,
            self.emitted_citation_count,
            "valid offset count must not exceed emitted citation count",
        )
        _validate_not_greater_than(
            self.contradicted_claim_citation_count,
            self.contradicted_claim_count,
            "contradicted claim citation count must not exceed contradicted claim count",
        )

        if len(self.retrieval_ranks) != self.retrieval_eligible_claim_count:
            raise ValueError(
                "retrieval_ranks must contain one entry per retrieval-eligible claim"
            )
        for rank in self.retrieval_ranks:
            if rank is not None and rank <= 0:
                raise ValueError(
                    "retrieval ranks must be positive integers or None"
                )
        return self


class MetricReport(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    record_count: int
    evaluator_error_count: int
    exact_precision: Rate
    exact_recall: Rate
    requirement_recall: Rate
    fully_attributed_claim_recall: Rate
    source_accuracy: Rate
    offset_validity: Rate
    recall_at_0_9: Rate
    recall_at_0_5: Rate
    multi_span_precision: Rate
    multi_span_recall: Rate
    contradiction_false_citation_rate: Rate
    status_confusion_matrix: StatusConfusionMatrix
    status_macro_f1: float | None
    retrieval_recall_at_1: Rate
    retrieval_recall_at_3: Rate
    retrieval_recall_at_5: Rate
    retrieval_mrr: float | None

    @model_validator(mode="after")
    def _validate_report(self) -> MetricReport:
        _validate_nonnegative(self.record_count, "record count must be non-negative")
        _validate_nonnegative(
            self.evaluator_error_count, "evaluator error count must be non-negative"
        )
        _validate_not_greater_than(
            self.evaluator_error_count,
            self.record_count,
            "evaluator error count must not exceed record count",
        )
        if self.status_macro_f1 is not None and not 0.0 <= self.status_macro_f1 <= 1.0:
            raise ValueError("status macro F1 must stay within [0, 1]")
        if self.retrieval_mrr is not None and not 0.0 <= self.retrieval_mrr <= 1.0:
            raise ValueError("retrieval MRR must stay within [0, 1]")
        return self


def aggregate_metrics(records: tuple[CaseMetricRecord, ...]) -> MetricReport:
    confusion_counts = {
        expected: {observed: 0 for observed in _STATUS_LABELS}
        for expected in _STATUS_LABELS
    }

    exact_tp = 0
    exact_fp = 0
    exact_fn = 0
    recall_09_tp = 0
    recall_09_fn = 0
    recall_05_tp = 0
    recall_05_fn = 0
    requirement_count = 0
    matched_requirement_count = 0
    entailed_claim_count = 0
    fully_attributed_claim_count = 0
    source_selection_attempt_count = 0
    source_selection_correct_count = 0
    emitted_citation_count = 0
    valid_offset_count = 0
    multi_span_tp = 0
    multi_span_fp = 0
    multi_span_fn = 0
    contradicted_claim_count = 0
    contradicted_claim_citation_count = 0
    retrieval_eligible_claim_count = 0
    retrieval_hits_at_1 = 0
    retrieval_hits_at_3 = 0
    retrieval_hits_at_5 = 0
    retrieval_reciprocal_rank_sum = 0.0
    evaluator_error_count = 0

    for record in records:
        confusion_counts[record.expected_status][record.observed_status] += 1

        exact_tp += record.exact_true_positives
        exact_fp += record.exact_false_positives
        exact_fn += record.exact_false_negatives
        recall_09_tp += record.recall_at_0_9_true_positives
        recall_09_fn += record.recall_at_0_9_false_negatives
        recall_05_tp += record.recall_at_0_5_true_positives
        recall_05_fn += record.recall_at_0_5_false_negatives
        requirement_count += record.requirement_count
        matched_requirement_count += record.matched_requirement_count
        entailed_claim_count += record.entailed_claim_count
        fully_attributed_claim_count += record.fully_attributed_claim_count
        source_selection_attempt_count += record.source_selection_attempt_count
        source_selection_correct_count += record.source_selection_correct_count
        emitted_citation_count += record.emitted_citation_count
        valid_offset_count += record.valid_offset_count
        multi_span_tp += record.multi_span_true_positives
        multi_span_fp += record.multi_span_false_positives
        multi_span_fn += record.multi_span_false_negatives
        contradicted_claim_count += record.contradicted_claim_count
        contradicted_claim_citation_count += record.contradicted_claim_citation_count
        retrieval_eligible_claim_count += record.retrieval_eligible_claim_count
        evaluator_error_count += int(record.evaluator_error)

        for rank in record.retrieval_ranks:
            if rank is None:
                continue
            if rank <= 1:
                retrieval_hits_at_1 += 1
            if rank <= 3:
                retrieval_hits_at_3 += 1
            if rank <= 5:
                retrieval_hits_at_5 += 1
            retrieval_reciprocal_rank_sum += 1.0 / rank

    confusion_matrix = StatusConfusionMatrix(
        supported=StatusCountRow(**confusion_counts["supported"]),
        partial=StatusCountRow(**confusion_counts["partial"]),
        unsupported=StatusCountRow(**confusion_counts["unsupported"]),
    )

    return MetricReport(
        record_count=len(records),
        evaluator_error_count=evaluator_error_count,
        exact_precision=_rate(exact_tp, exact_tp + exact_fp),
        exact_recall=_rate(exact_tp, exact_tp + exact_fn),
        requirement_recall=_rate(matched_requirement_count, requirement_count),
        fully_attributed_claim_recall=_rate(
            fully_attributed_claim_count, entailed_claim_count
        ),
        source_accuracy=_rate(
            source_selection_correct_count, source_selection_attempt_count
        ),
        offset_validity=_rate(valid_offset_count, emitted_citation_count),
        recall_at_0_9=_rate(recall_09_tp, recall_09_tp + recall_09_fn),
        recall_at_0_5=_rate(recall_05_tp, recall_05_tp + recall_05_fn),
        multi_span_precision=_rate(multi_span_tp, multi_span_tp + multi_span_fp),
        multi_span_recall=_rate(multi_span_tp, multi_span_tp + multi_span_fn),
        contradiction_false_citation_rate=_rate(
            contradicted_claim_citation_count, contradicted_claim_count
        ),
        status_confusion_matrix=confusion_matrix,
        status_macro_f1=_status_macro_f1(confusion_matrix),
        retrieval_recall_at_1=_rate(retrieval_hits_at_1, retrieval_eligible_claim_count),
        retrieval_recall_at_3=_rate(retrieval_hits_at_3, retrieval_eligible_claim_count),
        retrieval_recall_at_5=_rate(retrieval_hits_at_5, retrieval_eligible_claim_count),
        retrieval_mrr=(
            None
            if retrieval_eligible_claim_count == 0
            else retrieval_reciprocal_rank_sum / retrieval_eligible_claim_count
        ),
    )


def _rate(numerator: int, denominator: int) -> Rate:
    if denominator == 0:
        return Rate(
            numerator=numerator,
            denominator=denominator,
            estimate=None,
            lower=None,
            upper=None,
        )

    estimate = numerator / denominator
    z2 = WILSON_95_Z * WILSON_95_Z
    adjusted_denominator = 1.0 + z2 / denominator
    center = (estimate + z2 / (2.0 * denominator)) / adjusted_denominator
    margin = (
        WILSON_95_Z
        * sqrt((estimate * (1.0 - estimate) + z2 / (4.0 * denominator)) / denominator)
        / adjusted_denominator
    )
    return Rate(
        numerator=numerator,
        denominator=denominator,
        estimate=estimate,
        lower=max(0.0, center - margin),
        upper=min(1.0, center + margin),
    )


def _status_macro_f1(confusion_matrix: StatusConfusionMatrix) -> float | None:
    rows = {
        "supported": confusion_matrix.supported,
        "partial": confusion_matrix.partial,
        "unsupported": confusion_matrix.unsupported,
    }
    total = sum(
        getattr(row, observed)
        for row in rows.values()
        for observed in _STATUS_LABELS
    )
    if total == 0:
        return None

    per_label_f1: list[float] = []
    for label in _STATUS_LABELS:
        tp = getattr(rows[label], label)
        fp = sum(
            getattr(rows[expected], label)
            for expected in _STATUS_LABELS
            if expected != label
        )
        fn = sum(
            getattr(rows[label], observed)
            for observed in _STATUS_LABELS
            if observed != label
        )
        denominator = (2 * tp) + fp + fn
        per_label_f1.append(0.0 if denominator == 0 else (2.0 * tp) / denominator)
    return sum(per_label_f1) / len(per_label_f1)


def _validate_nonnegative(value: int, message: str) -> None:
    if value < 0:
        raise ValueError(message)


def _validate_not_greater_than(numerator: int, denominator: int, message: str) -> None:
    if numerator > denominator:
        raise ValueError(message)


__all__ = [
    "CaseMetricRecord",
    "MetricReport",
    "Rate",
    "StatusConfusionMatrix",
    "StatusCountRow",
    "StatusLabel",
    "aggregate_metrics",
]
