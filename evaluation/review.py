"""Manual review queue and completion ledger for evaluation claims."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections.abc import Iterable, Sequence
from datetime import date
from html import escape
from pathlib import Path
from re import Pattern
from typing import Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from evaluation.canonical import canonical_json_bytes, sha256_hex
from evaluation.schema import (
    CharSpan,
    ClaimAnnotation,
    EvaluationCase,
    EvaluationUnit,
    GenerationRecipe,
    Provenance,
    Source,
    Split,
)

STRICT_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
LEDGER_SCHEMA_VERSION = "1.0.0"
REVIEW_DECISIONS: tuple[Literal["approve", "correct", "reject"], ...] = (
    "approve",
    "correct",
    "reject",
)
REVIEW_STATES: tuple[Literal["missing", "current", "stale"], ...] = (
    "missing",
    "current",
    "stale",
)
_HEX_64_RE: Pattern[str] = re.compile(r"^[0-9a-f]{64}$")
_ISO_DATE_RE: Pattern[str] = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_DATETIME_RE: Pattern[str] = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


class ClaimReviewRecord(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    dataset_version: str
    case_id: str
    claim_id: str
    reviewer: str
    reviewed_at: str
    decision: Literal["approve", "correct", "reject"]
    binding_sha256: str
    notes: str | None = None
    correction_summary: str | None = None

    @model_validator(mode="after")
    def _validate_record(self) -> ClaimReviewRecord:
        if not self.reviewer.strip():
            raise ValueError("reviewer must be non-empty")
        if not (
            _ISO_DATE_RE.fullmatch(self.reviewed_at)
            or _ISO_DATETIME_RE.fullmatch(self.reviewed_at)
        ):
            raise ValueError(
                "reviewed_at must be an ISO date or timezone-qualified ISO datetime"
            )
        if not _HEX_64_RE.fullmatch(self.binding_sha256):
            raise ValueError("binding_sha256 must be a 64-character lowercase hex SHA-256")
        return self


class ReviewLedger(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    dataset_version: str
    schema_version: str
    entries: tuple[ClaimReviewRecord, ...] = Field(default_factory=tuple)

    @field_validator("entries", mode="before")
    @classmethod
    def _coerce_entries(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_entries(self) -> ReviewLedger:
        seen: set[tuple[str, str]] = set()
        ordered = tuple(
            sorted(
                self.entries,
                key=lambda entry: (entry.case_id, entry.claim_id, entry.reviewed_at),
            )
        )
        if ordered != self.entries:
            raise ValueError(
                "review ledger entries must be in canonical (case_id, claim_id) order"
            )
        for entry in self.entries:
            key = (entry.case_id, entry.claim_id)
            if key in seen:
                raise ValueError(
                    "review ledger entries must be unique by (case_id, claim_id)"
                )
            if entry.dataset_version != self.dataset_version:
                raise ValueError(
                    "review ledger entry dataset versions must match the ledger dataset_version"
                )
            seen.add(key)
        return self


class ReviewQueueItem(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    case: EvaluationCase
    unit: EvaluationUnit
    claim: ClaimAnnotation
    binding_sha256: str
    review: ClaimReviewRecord | None = None
    review_state: Literal["missing", "current", "stale"]


class ReviewQueue(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    dataset_version: str
    schema_version: str
    shard_index: int
    shard_count: int
    total_claims: int
    reviewed_claims: int
    current_claims: int
    stale_claims: int
    approved_claims: int
    corrected_claims: int
    rejected_claims: int
    missing_claims: int
    items: tuple[ReviewQueueItem, ...]


class ReviewCompletionFinding(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    code: Literal[
        "missing_review",
        "stale_review",
        "correct_review",
        "rejected_review",
    ]
    split: Split
    case_id: str
    claim_id: str
    message: str


class ReviewCompletionReport(BaseModel):
    model_config = STRICT_MODEL_CONFIG

    dataset_version: str
    schema_version: str
    splits: tuple[Split, ...]
    total_claims: int
    reviewed_claims: int
    current_claims: int
    stale_claims: int
    approved_claims: int
    corrected_claims: int
    rejected_claims: int
    missing_claims: int
    findings: tuple[ReviewCompletionFinding, ...]
    complete: bool


def claim_review_binding(case: EvaluationCase, claim: ClaimAnnotation) -> str:
    unit = _unit_for_claim(case, claim.claim_id)
    payload = {
        "dataset_version": case.dataset_version,
        "case_id": case.case_id,
        "answer": case.answer,
        "provenance": case.provenance.model_dump(mode="json"),
        "sources": tuple(source.model_dump(mode="json") for source in case.sources),
        "unit": {
            "unit_id": unit.unit_id,
            "answer_span": unit.answer_span.model_dump(mode="json"),
            "text": unit.text,
        },
        "claim": {
            "claim_id": claim.claim_id,
            "answer_span": claim.answer_span.model_dump(mode="json"),
            "text": claim.text,
            "label": claim.label,
            "acceptable_retrieval_source_ids": tuple(claim.acceptable_retrieval_source_ids),
            "citation_requirements": tuple(
                requirement.model_dump(mode="json")
                for requirement in claim.citation_requirements
            ),
        },
    }
    return sha256_hex(canonical_json_bytes(payload))


def make_review_record(
    case: EvaluationCase,
    claim: ClaimAnnotation,
    *,
    reviewer: str,
    reviewed_at: str,
    decision: Literal["approve", "correct", "reject"],
    notes: str | None = None,
    correction_summary: str | None = None,
) -> ClaimReviewRecord:
    return ClaimReviewRecord(
        dataset_version=case.dataset_version,
        case_id=case.case_id,
        claim_id=claim.claim_id,
        reviewer=reviewer,
        reviewed_at=reviewed_at,
        decision=decision,
        binding_sha256=claim_review_binding(case, claim),
        notes=notes,
        correction_summary=correction_summary,
    )


def build_review_queue(
    cases: Iterable[EvaluationCase],
    ledger: ReviewLedger | None = None,
    *,
    splits: Sequence[Split] | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> ReviewQueue:
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be between 0 and shard_count - 1")

    ordered_cases = _normalize_cases(cases)
    dataset_version = _single_dataset_version(ordered_cases)
    review_ledger = ledger or ReviewLedger(
        dataset_version=dataset_version,
        schema_version=LEDGER_SCHEMA_VERSION,
        entries=(),
    )
    if review_ledger.dataset_version != dataset_version:
        raise ValueError("ledger dataset_version does not match the provided cases")
    selected_splits = _normalize_splits(splits)
    split_filtered = tuple(
        case for case in ordered_cases if selected_splits is None or case.split in selected_splits
    )
    sharded_cases = _shard_cases(split_filtered, shard_index=shard_index, shard_count=shard_count)
    items = _queue_items(sharded_cases, review_ledger)
    counts = _queue_counts(items)
    return ReviewQueue(
        dataset_version=dataset_version,
        schema_version=review_ledger.schema_version,
        shard_index=shard_index,
        shard_count=shard_count,
        total_claims=len(items),
        reviewed_claims=counts["reviewed"],
        current_claims=counts["current"],
        stale_claims=counts["stale"],
        approved_claims=counts["approve"],
        corrected_claims=counts["correct"],
        rejected_claims=counts["reject"],
        missing_claims=counts["missing"],
        items=items,
    )


def render_review_queue_html(queue: ReviewQueue) -> str:
    lines = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<title>Cite Right Manual Review Queue</title>",
        "<style>",
        "body{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;background:#f5f1e8;color:#1c1917;}",
        "main{max-width:1200px;margin:0 auto;padding:32px 24px 64px;}",
        ".summary{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:24px 0;}",
        ".card,.item,.source-block,.claim,.unit{background:#fff;border:1px solid #d6d3d1;border-radius:12px;}",
        ".card{padding:12px 14px;}",
        ".item{padding:20px;margin-top:24px;}",
        ".meta{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 20px;}",
        ".label{font-size:12px;text-transform:uppercase;letter-spacing:0.06em;color:#57534e;}",
        "pre{white-space:pre-wrap;word-break:break-word;background:#fafaf9;padding:12px;border-radius:10px;}",
        ".unit,.claim,.source-block{padding:14px;margin-top:16px;}",
        ".pill{display:inline-block;padding:4px 10px;border-radius:999px;background:#e7e5e4;margin-right:8px;font-size:12px;}",
        ".target-span{background:#fde68a;border-radius:3px;}",
        ".source-title{display:flex;gap:10px;flex-wrap:wrap;align-items:center;}",
        "</style>",
        "</head>",
        "<body>",
        "<main>",
        "<h1>Manual Review Queue</h1>",
        f"<p>Dataset version {escape(queue.dataset_version)} | shard {queue.shard_index + 1} of {queue.shard_count}</p>",
        "<section class=\"summary\">",
    ]
    for label, value in (
        ("Total claims", queue.total_claims),
        ("Reviewed", queue.reviewed_claims),
        ("Current", queue.current_claims),
        ("Stale", queue.stale_claims),
        ("Approved", queue.approved_claims),
        ("Corrected", queue.corrected_claims),
        ("Rejected", queue.rejected_claims),
        ("Missing", queue.missing_claims),
    ):
        lines.extend(
            (
                "<div class=\"card\">",
                f"<div class=\"label\">{escape(label)}</div>",
                f"<div>{value}</div>",
                "</div>",
            )
        )
    lines.append("</section>")
    for item in queue.items:
        lines.extend(_render_item(item))
    lines.extend(("</main>", "</body>", "</html>"))
    return "\n".join(lines)


def review_completion(
    cases: Iterable[EvaluationCase],
    ledger: ReviewLedger,
    *,
    splits: Sequence[Split] | None = None,
) -> ReviewCompletionReport:
    queue = build_review_queue(cases, ledger, splits=splits)
    selected_splits = _normalize_splits(splits)
    if selected_splits is None:
        selected_splits = cast(
            tuple[Split, ...],
            tuple(sorted({item.case.split for item in queue.items})),
        )
    findings: list[ReviewCompletionFinding] = []
    complete = queue.total_claims > 0
    for item in queue.items:
        if item.review_state == "missing":
            complete = False
            findings.append(
                ReviewCompletionFinding(
                    code="missing_review",
                    split=item.case.split,
                    case_id=item.case.case_id,
                    claim_id=item.claim.claim_id,
                    message="claim is missing a review record",
                )
            )
            continue
        if item.review_state == "stale":
            complete = False
            findings.append(
                ReviewCompletionFinding(
                    code="stale_review",
                    split=item.case.split,
                    case_id=item.case.case_id,
                    claim_id=item.claim.claim_id,
                    message="claim review is stale because the bound evidence changed",
                )
            )
            continue
        if item.review is None:
            complete = False
            continue
        if item.review.decision == "correct":
            complete = False
            findings.append(
                ReviewCompletionFinding(
                    code="correct_review",
                    split=item.case.split,
                    case_id=item.case.case_id,
                    claim_id=item.claim.claim_id,
                    message="claim review requires correction before completion",
                )
            )
        elif item.review.decision == "reject":
            complete = False
            findings.append(
                ReviewCompletionFinding(
                    code="rejected_review",
                    split=item.case.split,
                    case_id=item.case.case_id,
                    claim_id=item.claim.claim_id,
                    message="claim review is rejected and does not satisfy completion",
                )
            )
    ordered_findings = tuple(
        sorted(
            findings,
            key=lambda finding: (
                _completion_finding_rank(finding.code),
                finding.case_id,
                finding.claim_id,
            ),
        )
    )
    return ReviewCompletionReport(
        dataset_version=queue.dataset_version,
        schema_version=queue.schema_version,
        splits=tuple(selected_splits),
        total_claims=queue.total_claims,
        reviewed_claims=queue.reviewed_claims,
        current_claims=queue.current_claims,
        stale_claims=queue.stale_claims,
        approved_claims=queue.approved_claims,
        corrected_claims=queue.corrected_claims,
        rejected_claims=queue.rejected_claims,
        missing_claims=queue.missing_claims,
        findings=ordered_findings,
        complete=complete,
    )


def assert_review_complete(
    cases: Iterable[EvaluationCase],
    ledger: ReviewLedger,
    *,
    split: Split,
) -> None:
    report = review_completion(cases, ledger, splits=(split,))
    if report.complete:
        return
    raise ValueError(
        f"{split} review incomplete: "
        f"total={report.total_claims} current={report.current_claims} stale={report.stale_claims} "
        f"approved={report.approved_claims} corrected={report.corrected_claims} "
        f"rejected={report.rejected_claims} missing={report.missing_claims}"
    )


def load_review_ledger(path: str | Path) -> ReviewLedger:
    ledger_path = Path(path)
    try:
        raw_bytes = ledger_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"unable to read review ledger {ledger_path}") from exc
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"review ledger {ledger_path} is not valid JSON") from exc
    try:
        ledger = ReviewLedger.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"review ledger {ledger_path} is invalid") from exc
    if raw_bytes != canonical_json_bytes(ledger):
        raise ValueError(f"review ledger {ledger_path} must use canonical JSON ordering")
    return ledger


def write_review_ledger(path: str | Path, ledger: ReviewLedger) -> None:
    ledger_path = Path(path)
    parent = ledger_path.parent
    if not parent.exists():
        raise FileNotFoundError(f"review ledger parent directory does not exist: {parent}")
    canonical_bytes = canonical_json_bytes(ledger)
    temp_fd, temp_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{ledger_path.name}.tmp.",
    )
    try:
        with os.fdopen(temp_fd, "wb") as handle:
            handle.write(canonical_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, ledger_path)
        _fsync_directory(parent)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.review")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fixture_parser = subparsers.add_parser("render-fixture")
    fixture_parser.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "render-fixture":
        output = Path(args.output)
        queue = build_review_queue((_fixture_case(),))
        html = render_review_queue_html(queue)
        output.write_text(html, encoding="utf-8")
        return 0
    parser.error(f"unknown command {args.command}")
    return 2


def _fixture_case() -> EvaluationCase:
    hostile_source = 'Alpha <script>alert("x")</script> & "Beta" and Gamma.'
    answer = 'Alpha and "Beta"'
    return EvaluationCase.model_validate(
        {
            "case_id": "fixture-case",
            "dataset_version": "1.0.0",
            "split": "dev",
            "document_family_id": 'fixture-family<danger>&"',
            "transformation_family_id": "fixture-render",
            "provenance": _fixture_provenance(),
            "sources": (
                {
                    "source_id": "source-1",
                    "text": hostile_source,
                },
            ),
            "answer": answer,
            "evaluation_units": (
                {
                    "unit_id": "unit-1",
                    "answer_span": {"start": 0, "end": len(answer)},
                    "text": answer,
                    "claims": (
                        {
                            "claim_id": "claim-1",
                            "answer_span": {"start": 0, "end": len(answer)},
                            "text": answer,
                            "label": "entailed",
                            "citation_requirements": (
                                {
                                    "requirement_id": "req-1",
                                    "alternatives": (
                                        {
                                            "source_id": "source-1",
                                            "spans": (
                                                {"start": 0, "end": 5},
                                                {"start": 28, "end": 32},
                                            ),
                                        },
                                    ),
                                },
                            ),
                            "acceptable_retrieval_source_ids": ("source-1",),
                        },
                    ),
                },
            ),
            "difficulty_tags": ("science", "fixture"),
            "generation": _fixture_generation(),
            "review": None,
        }
    )


def _fixture_provenance() -> Provenance:
    return Provenance(
        kind="authored",
        title='Fixture "Title" <unsafe>',
        origin='https://example.com/query?a=1&b="two"',
        publisher='Fixture & Co.',
        license="CC-BY-4.0",
        retrieval_date=date(2026, 7, 17),
        snapshot_hash="fixture-snapshot",
    )


def _fixture_generation() -> GenerationRecipe:
    return GenerationRecipe(
        recipe_id='recipe<fixture>',
        generator_name='generator "fixture"',
        prompt_version="v<1>",
        seed=17,
        notes='notes & "<unsafe>"',
    )


def _normalize_cases(cases: Iterable[EvaluationCase]) -> tuple[EvaluationCase, ...]:
    ordered_cases = tuple(cases)
    if not ordered_cases:
        raise ValueError("cases must not be empty")
    seen_case_ids: set[str] = set()
    dataset_versions = {case.dataset_version for case in ordered_cases}
    if len(dataset_versions) != 1:
        raise ValueError("cases must all share one dataset_version")
    for case in ordered_cases:
        if case.case_id in seen_case_ids:
            raise ValueError(f"duplicate case id {case.case_id!r}")
        seen_case_ids.add(case.case_id)
        seen_claim_ids: set[str] = set()
        for unit in case.evaluation_units:
            for claim in unit.claims:
                if claim.claim_id in seen_claim_ids:
                    raise ValueError(
                        f"case {case.case_id!r} reuses claim_id {claim.claim_id!r} across units"
                    )
                seen_claim_ids.add(claim.claim_id)
    return tuple(sorted(ordered_cases, key=lambda case: case.case_id))


def _single_dataset_version(cases: Sequence[EvaluationCase]) -> str:
    dataset_versions = {case.dataset_version for case in cases}
    if len(dataset_versions) != 1:
        raise ValueError("cases must all share one dataset_version")
    return next(iter(dataset_versions))


def _normalize_splits(splits: Sequence[Split] | None) -> tuple[Split, ...] | None:
    if splits is None:
        return None
    unique = tuple(dict.fromkeys(splits))
    for split in unique:
        if split not in {"train", "dev", "holdout"}:
            raise ValueError(f"unsupported split {split!r}")
    return unique


def _shard_cases(
    cases: Sequence[EvaluationCase],
    *,
    shard_index: int,
    shard_count: int,
) -> tuple[EvaluationCase, ...]:
    family_ids = tuple(sorted({case.document_family_id for case in cases}))
    family_to_shard = {
        family_id: index % shard_count for index, family_id in enumerate(family_ids)
    }
    return tuple(
        case
        for case in cases
        if family_to_shard[case.document_family_id] == shard_index
    )


def _queue_items(
    cases: Sequence[EvaluationCase],
    ledger: ReviewLedger,
) -> tuple[ReviewQueueItem, ...]:
    review_by_key = {(entry.case_id, entry.claim_id): entry for entry in ledger.entries}
    items: list[ReviewQueueItem] = []
    for case in cases:
        for unit in sorted(case.evaluation_units, key=lambda value: value.unit_id):
            for claim in sorted(unit.claims, key=lambda value: value.claim_id):
                binding = claim_review_binding(case, claim)
                review = review_by_key.get((case.case_id, claim.claim_id))
                review_state: Literal["missing", "current", "stale"]
                if review is None:
                    review_state = "missing"
                elif review.binding_sha256 == binding:
                    review_state = "current"
                else:
                    review_state = "stale"
                items.append(
                    ReviewQueueItem(
                        case=case,
                        unit=unit,
                        claim=claim,
                        binding_sha256=binding,
                        review=review,
                        review_state=review_state,
                    )
                )
    return tuple(items)


def _queue_counts(items: Sequence[ReviewQueueItem]) -> dict[str, int]:
    counts = {
        "reviewed": 0,
        "current": 0,
        "stale": 0,
        "missing": 0,
        "approve": 0,
        "correct": 0,
        "reject": 0,
    }
    for item in items:
        if item.review is None:
            counts["missing"] += 1
            continue
        counts["reviewed"] += 1
        counts[item.review.decision] += 1
        if item.review_state == "current" and item.case.split in {"dev", "holdout"}:
            counts["current"] += 1
        elif item.review_state == "stale":
            counts["stale"] += 1
    return counts


def _render_item(item: ReviewQueueItem) -> list[str]:
    case = item.case
    review_badges = [
        f"<span class=\"pill\">split {escape(case.split)}</span>",
        f"<span class=\"pill\">review {escape(item.review_state)}</span>",
    ]
    if item.review is not None:
        review_badges.append(
            f"<span class=\"pill\">decision {escape(item.review.decision)}</span>"
        )
    lines = [
        (
            f"<section class=\"item\" data-case-id=\"{escape(case.case_id)}\" "
            f"data-family-id=\"{escape(case.document_family_id)}\">"
        ),
        f"<h2>{escape(case.case_id)}</h2>",
        "<div>" + "".join(review_badges) + "</div>",
        "<div class=\"meta\">",
        _meta_row("Family", case.document_family_id),
        _meta_row("Transformation", case.transformation_family_id),
        _meta_row("Provenance kind", case.provenance.kind),
        _meta_row("Provenance title", case.provenance.title),
        _meta_row("Provenance origin", case.provenance.origin),
        _meta_row("Publisher", case.provenance.publisher),
        _meta_row("License", case.provenance.license),
        _meta_row(
            "Retrieval date",
            None if case.provenance.retrieval_date is None else str(case.provenance.retrieval_date),
        ),
        _meta_row("Snapshot hash", case.provenance.snapshot_hash),
        _meta_row("Generation", _generation_summary(case.generation)),
        "</div>",
        "<h3>Full Answer</h3>",
        f"<pre>{escape(case.answer)}</pre>",
        _render_unit(item.unit, item.claim),
    ]
    for source in case.sources:
        lines.extend(_render_source_blocks(source, item.claim))
    lines.append("</section>")
    return lines


def _render_unit(unit: EvaluationUnit, claim: ClaimAnnotation) -> str:
    retrieval = (
        ", ".join(claim.acceptable_retrieval_source_ids)
        if claim.acceptable_retrieval_source_ids
        else "None"
    )
    return "\n".join(
        (
            "<section class=\"unit\">",
            f"<h3>Unit {escape(unit.unit_id)}</h3>",
            f"<pre>{escape(unit.text)}</pre>",
            "<section class=\"claim\">",
            f"<div class=\"label\">Claim {escape(claim.claim_id)}</div>",
            f"<pre>{escape(claim.text)}</pre>",
            f"<div><span class=\"pill\">label {escape(claim.label)}</span>"
            f"<span class=\"pill\">retrieval {escape(retrieval)}</span></div>",
            "</section>",
            "</section>",
        )
    )


def _render_source_blocks(source: Source, claim: ClaimAnnotation) -> list[str]:
    lines = [
        "<section class=\"source-block\">",
        "<div class=\"source-title\">",
        f"<strong>Source {escape(source.source_id)}</strong>",
        "</div>",
        f"<pre>{escape(source.text)}</pre>",
    ]
    if claim.citation_requirements:
        for requirement in claim.citation_requirements:
            for alternative_index, alternative in enumerate(requirement.alternatives, start=1):
                if alternative.source_id != source.source_id:
                    continue
                lines.extend(
                    (
                        (
                            "<div class=\"source-block\">"
                            f"<div class=\"label\">Requirement {escape(requirement.requirement_id)}"
                            f" | Alternative {alternative_index} | Source {escape(source.source_id)}</div>"
                        ),
                        f"<pre>{_highlight_source_text(source.text, alternative.spans)}</pre>",
                        "</div>",
                    )
                )
    lines.append("</section>")
    return lines


def _highlight_source_text(text: str, spans: Sequence[CharSpan]) -> str:
    merged = _merge_spans(tuple((span.start, span.end) for span in spans))
    if not merged:
        return escape(text)
    parts: list[str] = []
    cursor = 0
    for index, (start, end) in enumerate(merged, start=1):
        parts.append(escape(text[cursor:start]))
        parts.append(
            f"<mark class=\"target-span\" data-span-index=\"{index}\">{escape(text[start:end])}</mark>"
        )
        cursor = end
    parts.append(escape(text[cursor:]))
    return "".join(parts)


def _merge_spans(spans: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    if not spans:
        return ()
    ordered = sorted(spans)
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
            continue
        merged.append((start, end))
    return tuple(merged)


def _meta_row(label: str, value: object) -> str:
    rendered = "" if value is None else escape(str(value))
    return (
        f"<div><div class=\"label\">{escape(label)}</div><div>{rendered}</div></div>"
    )


def _generation_summary(generation: GenerationRecipe | None) -> str:
    if generation is None:
        return "None"
    pieces = [generation.recipe_id, generation.generator_name, generation.prompt_version]
    if generation.seed is not None:
        pieces.append(f"seed={generation.seed}")
    if generation.notes is not None:
        pieces.append(generation.notes)
    return " | ".join(pieces)


def _unit_for_claim(case: EvaluationCase, claim_id: str) -> EvaluationUnit:
    matching_units = [
        unit for unit in case.evaluation_units if any(claim.claim_id == claim_id for claim in unit.claims)
    ]
    if len(matching_units) != 1:
        raise ValueError(
            f"case {case.case_id!r} must contain exactly one claim named {claim_id!r}"
        )
    return matching_units[0]


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _completion_finding_rank(
    code: Literal["missing_review", "stale_review", "correct_review", "rejected_review"]
) -> int:
    order = {
        "stale_review": 0,
        "missing_review": 1,
        "correct_review": 2,
        "rejected_review": 3,
    }
    return order[code]


if __name__ == "__main__":
    raise SystemExit(main())
