from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import date
from html import unescape
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from evaluation.builders.cases import generate_all_authored_cases
from evaluation.builders.real_sources import generate_real_cases
from evaluation.canonical import authoritative_case_id, canonical_json_bytes
from evaluation.review import (
    ClaimReviewRecord,
    ReviewLedger,
    ReviewQueue,
    assert_review_complete,
    build_review_queue,
    claim_review_binding,
    load_review_ledger,
    make_review_record,
    render_review_queue_html,
    review_completion,
    write_review_ledger,
)
from evaluation.schema import (
    CharSpan,
    CitationRequirement,
    CitationTarget,
    ClaimAnnotation,
    EvaluationCase,
    EvaluationUnit,
    GenerationRecipe,
    Provenance,
    ProvenanceKind,
    ReviewRecord,
    Source,
    Split,
    SupportLabel,
)
from evaluation.splitting import apply_split_assignments, assign_splits


def test_claim_review_binding_tracks_review_relevant_fields_and_ignores_split_and_case_review() -> (
    None
):
    base_case = _build_case(
        family_id="family-binding",
        transformation_id="binding",
        answer='Alpha "quoted" & beta',
        source_texts=("Alpha evidence and beta evidence.", "Fallback evidence."),
    )
    base_claim = base_case.evaluation_units[0].claims[0]
    baseline = claim_review_binding(base_case, base_claim)

    changed_answer = _copy_case(
        base_case,
        answer="Updated answer text",
        evaluation_units=(
            _build_unit(
                unit_id="unit-1",
                answer="Updated answer text",
                claims=(
                    _build_claim(
                        claim_id=base_claim.claim_id,
                        answer="Updated answer text",
                        label=base_claim.label,
                        requirement_id="req-1",
                        source_id="source-1",
                        spans=((0, 7),),
                        acceptable_retrieval_source_ids=("source-1",),
                    ),
                ),
            ),
        ),
    )
    assert (
        claim_review_binding(
            changed_answer, changed_answer.evaluation_units[0].claims[0]
        )
        != baseline
    )

    changed_source = _copy_case(
        base_case,
        sources=(
            Source(source_id="source-1", text="Changed source text."),
            base_case.sources[1],
        ),
    )
    assert (
        claim_review_binding(
            changed_source, changed_source.evaluation_units[0].claims[0]
        )
        != baseline
    )

    changed_label = _copy_case(
        base_case,
        evaluation_units=(
            _build_unit(
                unit_id="unit-1",
                answer=base_case.answer,
                claims=(
                    _build_claim(
                        claim_id=base_claim.claim_id,
                        answer=base_case.answer,
                        label="contradicted",
                    ),
                ),
            ),
        ),
    )
    assert (
        claim_review_binding(changed_label, changed_label.evaluation_units[0].claims[0])
        != baseline
    )

    changed_target = _copy_case(
        base_case,
        evaluation_units=(
            _build_unit(
                unit_id="unit-1",
                answer=base_case.answer,
                claims=(
                    _build_claim(
                        claim_id=base_claim.claim_id,
                        answer=base_case.answer,
                        label=base_claim.label,
                        requirement_id="req-1",
                        source_id="source-1",
                        spans=((6, 14),),
                        acceptable_retrieval_source_ids=("source-1",),
                    ),
                ),
            ),
        ),
    )
    assert (
        claim_review_binding(
            changed_target, changed_target.evaluation_units[0].claims[0]
        )
        != baseline
    )

    changed_retrieval = _copy_case(
        base_case,
        evaluation_units=(
            _build_unit(
                unit_id="unit-1",
                answer=base_case.answer,
                claims=(
                    _build_claim(
                        claim_id=base_claim.claim_id,
                        answer=base_case.answer,
                        label=base_claim.label,
                        requirement_id="req-1",
                        source_id="source-1",
                        spans=((0, 5),),
                        acceptable_retrieval_source_ids=("source-2",),
                    ),
                ),
            ),
        ),
    )
    assert (
        claim_review_binding(
            changed_retrieval, changed_retrieval.evaluation_units[0].claims[0]
        )
        != baseline
    )

    changed_split = _copy_case(base_case, split="holdout")
    assert (
        claim_review_binding(changed_split, changed_split.evaluation_units[0].claims[0])
        == baseline
    )

    changed_review_metadata = _copy_case(
        base_case,
        review=ReviewRecord(
            state="approved",
            reviewer="internal-reviewer",
            reviewed_at=date(2026, 7, 17),
            notes="Operational metadata only.",
        ),
    )
    assert (
        claim_review_binding(
            changed_review_metadata,
            changed_review_metadata.evaluation_units[0].claims[0],
        )
        == baseline
    )


def test_make_review_record_and_ledger_validation_contracts() -> None:
    case = _build_case(family_id="family-record", transformation_id="record")
    claim = case.evaluation_units[0].claims[0]

    approve = make_review_record(
        case,
        claim,
        reviewer="reviewer-1",
        reviewed_at="2026-07-17",
        decision="approve",
        notes="Looks good.",
    )
    correct = make_review_record(
        case,
        claim,
        reviewer="reviewer-2",
        reviewed_at="2026-07-17T12:34:56+00:00",
        decision="correct",
        correction_summary="Needs one correction.",
    )
    reject = make_review_record(
        case,
        claim,
        reviewer="reviewer-3",
        reviewed_at="2026-07-17T14:30:00Z",
        decision="reject",
    )

    assert approve.decision == "approve"
    assert correct.decision == "correct"
    assert reject.decision == "reject"
    assert approve.binding_sha256 == claim_review_binding(case, claim)
    assert len(approve.binding_sha256) == 64

    with pytest.raises(ValidationError):
        ClaimReviewRecord.model_validate(
            {
                **approve.model_dump(mode="json"),
                "reviewer": "   ",
            }
        )
    with pytest.raises(ValidationError):
        ClaimReviewRecord.model_validate(
            {
                **approve.model_dump(mode="json"),
                "reviewed_at": "2026/07/17",
            }
        )
    with pytest.raises(ValidationError):
        ClaimReviewRecord.model_validate(
            {
                **approve.model_dump(mode="json"),
                "binding_sha256": "not-a-sha256",
            }
        )

    ordered = ReviewLedger(
        dataset_version=case.dataset_version,
        schema_version="1.0.0",
        entries=(approve,),
    )
    assert ordered.entries == (approve,)

    with pytest.raises(ValidationError):
        ReviewLedger(
            dataset_version=case.dataset_version,
            schema_version="1.0.0",
            entries=(
                make_review_record(
                    _copy_case(case, case_id="case-z"),
                    _copy_case(case, case_id="case-z").evaluation_units[0].claims[0],
                    reviewer="reviewer-z",
                    reviewed_at="2026-07-17",
                    decision="approve",
                ),
                make_review_record(
                    _copy_case(case, case_id="case-a"),
                    _copy_case(case, case_id="case-a").evaluation_units[0].claims[0],
                    reviewer="reviewer-a",
                    reviewed_at="2026-07-17",
                    decision="approve",
                ),
            ),
        )

    with pytest.raises(ValidationError):
        ReviewLedger(
            dataset_version=case.dataset_version,
            schema_version="1.0.0",
            entries=(approve, approve),
        )


def test_build_review_queue_is_order_invariant_and_family_aware_across_shards() -> None:
    family_a_first = _build_case(
        family_id="family-a",
        transformation_id="first",
        case_id="case-a-1",
        split="dev",
    )
    family_a_second = _build_case(
        family_id="family-a",
        transformation_id="second",
        case_id="case-a-2",
        split="dev",
    )
    family_b = _build_case(
        family_id="family-b",
        transformation_id="only",
        case_id="case-b-1",
        split="dev",
    )
    family_c = _build_case(
        family_id="family-c",
        transformation_id="only",
        case_id="case-c-1",
        split="holdout",
    )
    cases = (family_b, family_a_second, family_c, family_a_first)

    shard_zero = build_review_queue(cases, shard_count=2, shard_index=0)
    reversed_shard_zero = build_review_queue(
        tuple(reversed(cases)),
        shard_count=2,
        shard_index=0,
    )
    shard_one = build_review_queue(cases, shard_count=2, shard_index=1)

    assert _queue_keys(shard_zero) == _queue_keys(reversed_shard_zero)
    assert {item.case.document_family_id for item in shard_zero.items}.isdisjoint(
        {item.case.document_family_id for item in shard_one.items}
    )
    assert {
        item.case.document_family_id for item in shard_zero.items + shard_one.items
    } == {"family-a", "family-b", "family-c"}
    assert _queue_keys(build_review_queue(cases)) == (
        ("case-a-1", "unit-1", "claim-1"),
        ("case-a-2", "unit-1", "claim-1"),
        ("case-b-1", "unit-1", "claim-1"),
        ("case-c-1", "unit-1", "claim-1"),
    )

    with pytest.raises(ValueError):
        build_review_queue(cases, shard_count=0, shard_index=0)
    with pytest.raises(ValueError):
        build_review_queue(cases, shard_count=2, shard_index=2)
    with pytest.raises(ValueError):
        build_review_queue(
            (family_a_first, family_a_first), shard_count=1, shard_index=0
        )


def test_render_review_queue_html_escapes_all_data_and_avoids_script_injection() -> (
    None
):
    hostile_case = _build_case(
        family_id='family<danger>&"x"',
        transformation_id="render",
        case_id="case-render",
        split="dev",
        answer='<script>alert("x")</script> answer & tail',
        source_texts=('Source says <script>alert("x")</script> & more.',),
        provenance_origin='https://example.com/query?a=1&b="two"',
        publisher='ACME & "Partners"',
        generation=GenerationRecipe(
            recipe_id="recipe<1>",
            generator_name='generator "unsafe"',
            prompt_version="v<2>",
            seed=7,
            notes='notes & "<danger>"',
        ),
    )
    hostile_case = _copy_case(
        hostile_case,
        sources=(
            Source(
                source_id="source-1",
                text='Source says <script>alert("x")</script> & more.',
                chunk_id='chunk<danger>&"1"',
                chunk_char_start=100,
                chunk_char_end=145,
            ),
        ),
    )

    html = render_review_queue_html(build_review_queue((hostile_case,)))

    assert "<script" not in html
    assert "<link" not in html
    assert "src=" not in html
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; answer &amp; tail" in html
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; more." in html
    assert "family&lt;danger&gt;&amp;&quot;x&quot;" in html
    assert "https://example.com/query?a=1&amp;b=&quot;two&quot;" in html
    assert "chunk&lt;danger&gt;&amp;&quot;1&quot;" in html
    assert ">100<" in html
    assert ">145<" in html
    assert 'data-case-id="case-render"' in html
    assert 'data-family-id="family&lt;danger&gt;&amp;&quot;x&quot;"' in html


def test_render_review_queue_html_highlights_exact_targets_without_losing_source_text() -> (
    None
):
    case = _build_case(
        family_id="family-highlight",
        transformation_id="highlight",
        source_texts=('Alpha <tag> and "Beta" & Gamma.',),
        answer='Alpha and "Beta"',
        split="dev",
        claims=(
            _build_claim(
                claim_id="claim-1",
                answer='Alpha and "Beta"',
                requirement_id="req-1",
                source_id="source-1",
                spans=((0, 5), (5, 11), (17, 21)),
                acceptable_retrieval_source_ids=("source-1",),
            ),
        ),
    )

    html = render_review_queue_html(build_review_queue((case,)))
    normalized = _normalize_space(_strip_tags(html))

    assert 'Alpha <tag> and "Beta" & Gamma.' in normalized
    assert "Requirement req-1" in normalized
    assert "Alternative 1" in normalized
    assert "Alpha &lt;tag&gt; and &quot;Beta&quot; &amp; Gamma." in html
    assert html.count('class="target-span"') == 3
    assert "</mark><mark" in html
    assert 'data-span-start="0"' in html
    assert 'data-span-end="5"' in html
    assert 'data-span-start="5"' in html
    assert 'data-span-end="11"' in html
    assert 'data-span-start="17"' in html
    assert 'data-span-end="21"' in html


def test_render_review_queue_html_keeps_overlapping_alternatives_independently_auditable() -> (
    None
):
    case = _build_case(
        family_id="family-overlap",
        transformation_id="overlap",
        source_texts=('Alpha <tag> and "Beta" & Gamma.',),
        answer='Alpha and "Beta"',
        split="dev",
        claims=(
            ClaimAnnotation(
                claim_id="claim-1",
                answer_span=CharSpan(start=0, end=len('Alpha and "Beta"')),
                text='Alpha and "Beta"',
                label="entailed",
                citation_requirements=(
                    CitationRequirement(
                        requirement_id="req-1",
                        alternatives=(
                            CitationTarget(
                                source_id="source-1",
                                spans=(
                                    CharSpan(start=0, end=5),
                                    CharSpan(start=5, end=11),
                                ),
                            ),
                            CitationTarget(
                                source_id="source-1",
                                spans=(
                                    CharSpan(start=0, end=11),
                                    CharSpan(start=17, end=21),
                                ),
                            ),
                        ),
                    ),
                ),
                acceptable_retrieval_source_ids=("source-1",),
            ),
        ),
    )

    html = render_review_queue_html(build_review_queue((case,)))
    normalized = _normalize_space(_strip_tags(html))

    assert normalized.count('Alpha <tag> and "Beta" & Gamma.') >= 3
    assert "Alternative 1" in normalized
    assert "Alternative 2" in normalized
    assert html.count('data-span-start="0"') >= 2
    assert html.count('data-span-end="11"') >= 2
    assert 'data-span-start="17"' in html
    assert 'data-span-end="21"' in html


def test_review_completion_counts_and_gate_rules_distinguish_current_stale_and_incomplete_states() -> (
    None
):
    train_case = _build_case(
        family_id="family-train",
        transformation_id="train",
        case_id="case-train",
        split="train",
    )
    dev_approved = _build_case(
        family_id="family-dev-approved",
        transformation_id="dev-approved",
        case_id="case-dev-approved",
        split="dev",
    )
    dev_corrected = _build_case(
        family_id="family-dev-corrected",
        transformation_id="dev-corrected",
        case_id="case-dev-corrected",
        split="dev",
    )
    holdout_stale = _build_case(
        family_id="family-holdout-stale",
        transformation_id="holdout-stale",
        case_id="case-holdout-stale",
        split="holdout",
    )
    holdout_missing = _build_case(
        family_id="family-holdout-missing",
        transformation_id="holdout-missing",
        case_id="case-holdout-missing",
        split="holdout",
    )

    stale_record = make_review_record(
        holdout_stale,
        holdout_stale.evaluation_units[0].claims[0],
        reviewer="reviewer-stale",
        reviewed_at="2026-07-17",
        decision="approve",
    )
    mutated_holdout = _copy_case(
        holdout_stale,
        sources=(
            Source(
                source_id="source-1",
                text="Changed stale source text that stays long enough for the original spans.",
            ),
        ),
    )
    ledger = ReviewLedger(
        dataset_version="1.0.0",
        schema_version="1.0.0",
        entries=(
            make_review_record(
                dev_approved,
                dev_approved.evaluation_units[0].claims[0],
                reviewer="reviewer-dev-approve",
                reviewed_at="2026-07-17",
                decision="approve",
            ),
            make_review_record(
                dev_corrected,
                dev_corrected.evaluation_units[0].claims[0],
                reviewer="reviewer-dev-correct",
                reviewed_at="2026-07-17T12:00:00+00:00",
                decision="correct",
                correction_summary="Need a better citation.",
            ),
            stale_record,
            make_review_record(
                train_case,
                train_case.evaluation_units[0].claims[0],
                reviewer="reviewer-train",
                reviewed_at="2026-07-17",
                decision="reject",
            ),
        ),
    )

    report = review_completion(
        (train_case, dev_approved, dev_corrected, mutated_holdout, holdout_missing),
        ledger,
    )
    dev_report = review_completion(
        (train_case, dev_approved, dev_corrected), ledger, splits=("dev",)
    )
    holdout_report = review_completion(
        (mutated_holdout, holdout_missing), ledger, splits=("holdout",)
    )

    assert report.total_claims == 5
    assert report.reviewed_claims == 4
    assert report.current_claims == 2
    assert report.stale_claims == 1
    assert report.approved_claims == 2
    assert report.corrected_claims == 1
    assert report.rejected_claims == 1
    assert report.missing_claims == 1
    assert report.complete is False
    assert tuple(finding.code for finding in holdout_report.findings) == (
        "stale_review",
        "missing_review",
    )

    assert dev_report.complete is False
    with pytest.raises(ValueError):
        assert_review_complete(
            (train_case, dev_approved, dev_corrected),
            ledger,
            split="dev",
        )
    with pytest.raises(ValueError):
        assert_review_complete(
            (mutated_holdout, holdout_missing), ledger, split="holdout"
        )

    dev_clean_ledger = ReviewLedger(
        dataset_version="1.0.0",
        schema_version="1.0.0",
        entries=(
            make_review_record(
                dev_approved,
                dev_approved.evaluation_units[0].claims[0],
                reviewer="reviewer-dev-approve",
                reviewed_at="2026-07-17",
                decision="approve",
            ),
            make_review_record(
                dev_corrected,
                dev_corrected.evaluation_units[0].claims[0],
                reviewer="reviewer-dev-corrected",
                reviewed_at="2026-07-17",
                decision="approve",
            ),
        ),
    )
    dev_clean_report = review_completion(
        (train_case, dev_approved, dev_corrected),
        dev_clean_ledger,
        splits=("dev",),
    )
    assert dev_clean_report.complete is True
    assert_review_complete(
        (train_case, dev_approved, dev_corrected),
        dev_clean_ledger,
        split="dev",
    )


def test_load_and_write_review_ledger_are_canonical_and_atomic_on_replace_failure(
    tmp_path: Path,
) -> None:
    case = _build_case(
        family_id="family-write",
        transformation_id="write",
        case_id="case-write",
        split="dev",
    )
    ledger = ReviewLedger(
        dataset_version=case.dataset_version,
        schema_version="1.0.0",
        entries=(
            make_review_record(
                case,
                case.evaluation_units[0].claims[0],
                reviewer="reviewer-write",
                reviewed_at="2026-07-17",
                decision="approve",
            ),
        ),
    )
    target = tmp_path / "ledger.json"
    target.parent.mkdir(parents=True, exist_ok=True)

    write_review_ledger(target, ledger)
    written_bytes = target.read_bytes()
    assert written_bytes == canonical_json_bytes(ledger)
    assert load_review_ledger(target) == ledger

    previous_bytes = target.read_bytes()

    from evaluation import review as review_module

    def explode_replace(source: str, destination: str) -> None:
        raise OSError("replace failed")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(review_module.os, "replace", explode_replace)
    try:
        with pytest.raises(OSError, match="replace failed"):
            write_review_ledger(target, ledger)
    finally:
        monkeypatch.undo()

    assert target.read_bytes() == previous_bytes


def test_load_review_ledger_rejects_malformed_noncanonical_unknown_fields_and_duplicate_keys(
    tmp_path: Path,
) -> None:
    case = _build_case(
        family_id="family-load",
        transformation_id="load",
        case_id="case-load",
        split="dev",
    )
    record = make_review_record(
        case,
        case.evaluation_units[0].claims[0],
        reviewer="reviewer-load",
        reviewed_at="2026-07-17",
        decision="approve",
    )
    good_ledger = ReviewLedger(
        dataset_version="1.0.0",
        schema_version="1.0.0",
        entries=(record,),
    )
    root = tmp_path
    malformed = root / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")

    unsorted = root / "unsorted.json"
    unsorted_payload = {
        "dataset_version": "1.0.0",
        "entries": [
            {
                **record.model_dump(mode="json"),
                "case_id": "case-z",
            },
            {
                **record.model_dump(mode="json"),
                "case_id": "case-a",
            },
        ],
        "schema_version": "1.0.0",
    }
    unsorted.write_bytes(canonical_json_bytes(unsorted_payload))

    duplicate = root / "duplicate.json"
    duplicate.write_bytes(
        canonical_json_bytes(
            {
                "dataset_version": "1.0.0",
                "schema_version": "1.0.0",
                "entries": [
                    record.model_dump(mode="json"),
                    record.model_dump(mode="json"),
                ],
            }
        )
    )

    unknown = root / "unknown.json"
    unknown.write_bytes(
        canonical_json_bytes(
            {
                **good_ledger.model_dump(mode="json"),
                "extra_field": "not allowed",
            }
        )
    )

    noncanonical = root / "noncanonical.json"
    noncanonical.write_text(
        json.dumps(good_ledger.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    bad_hash = root / "bad-hash.json"
    bad_hash.write_bytes(
        canonical_json_bytes(
            {
                "dataset_version": "1.0.0",
                "schema_version": "1.0.0",
                "entries": [
                    {
                        **record.model_dump(mode="json"),
                        "binding_sha256": "xyz",
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError):
        load_review_ledger(malformed)
    with pytest.raises(ValueError):
        load_review_ledger(unsorted)
    with pytest.raises(ValueError):
        load_review_ledger(duplicate)
    with pytest.raises(ValueError):
        load_review_ledger(unknown)
    with pytest.raises(ValueError):
        load_review_ledger(noncanonical)
    with pytest.raises(ValueError):
        load_review_ledger(bad_hash)


def test_checked_in_dev_ledger_is_canonical_complete_and_excludes_holdout() -> None:
    scaffold_path = Path("evaluation/data/v1/dev_reviews.json")
    ledger = load_review_ledger(scaffold_path)

    assert ledger.entries
    assert scaffold_path.read_bytes() == canonical_json_bytes(ledger)

    corpus = generate_all_authored_cases(seed=20260717) + generate_real_cases()
    assignment_report = assign_splits(corpus, seed=20260717)
    assigned = apply_split_assignments(corpus, assignment_report.assignment_by_case_id)

    dev_cases = tuple(case for case in assigned if case.split == "dev")
    holdout_cases = tuple(case for case in assigned if case.split == "holdout")
    dev_claim_count = sum(
        len(unit.claims) for case in dev_cases for unit in case.evaluation_units
    )
    holdout_claim_count = sum(
        len(unit.claims) for case in holdout_cases for unit in case.evaluation_units
    )

    dev_report = review_completion(assigned, ledger, splits=("dev",))
    holdout_report = review_completion(assigned, ledger, splits=("holdout",))

    assert len(assigned) == 750
    assert dev_claim_count > 0
    assert holdout_claim_count > 0
    assert dev_report.total_claims == dev_claim_count
    assert dev_report.approved_claims == dev_claim_count
    assert dev_report.missing_claims == 0
    assert dev_report.complete is True
    assert holdout_report.total_claims == holdout_claim_count
    assert holdout_report.missing_claims == holdout_claim_count
    assert holdout_report.complete is False

    assert_review_complete(assigned, ledger, split="dev")
    with pytest.raises(ValueError):
        assert_review_complete(assigned, ledger, split="holdout")


def test_render_fixture_cli_output_is_deterministic_and_inspectable(
    tmp_path: Path,
) -> None:
    first = tmp_path / "fixture-first.html"
    second = tmp_path / "fixture-second.html"

    first_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.review",
            "render-fixture",
            "--output",
            str(first),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    second_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.review",
            "render-fixture",
            "--output",
            str(second),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr
    assert first.read_bytes() == second.read_bytes()
    assert "<script" not in first.read_text(encoding="utf-8")
    assert 'class="target-span"' in first.read_text(encoding="utf-8")
    assert "fixture-case" in first.read_text(encoding="utf-8")


def _build_case(
    *,
    family_id: str,
    transformation_id: str,
    case_id: str | None = None,
    split: str = "train",
    answer: str | None = None,
    source_texts: tuple[str, ...] | None = None,
    provenance_kind: str = "authored",
    provenance_origin: str = "https://example.com/source",
    publisher: str = "Cite Right",
    generation: GenerationRecipe | None = None,
    claims: tuple[ClaimAnnotation, ...] | None = None,
    review: ReviewRecord | None = None,
) -> EvaluationCase:
    answer_text = answer or f"{family_id} answer for {transformation_id}."
    source_values = source_texts or (answer_text,)
    sources = tuple(
        Source(source_id=f"source-{index}", text=text)
        for index, text in enumerate(source_values, start=1)
    )
    resolved_claims = claims or (
        _build_claim(
            claim_id="claim-1",
            answer=answer_text,
            requirement_id="req-1",
            source_id=sources[0].source_id,
            spans=((0, min(len(sources[0].text), max(1, len(sources[0].text) // 2))),),
            acceptable_retrieval_source_ids=(sources[0].source_id,),
        ),
    )
    unit = _build_unit(unit_id="unit-1", answer=answer_text, claims=resolved_claims)
    provenance = Provenance(
        kind=cast(ProvenanceKind, provenance_kind),
        title=f"{family_id} title",
        origin=provenance_origin,
        publisher=publisher,
        license="CC-BY-4.0",
        retrieval_date=date(2026, 7, 17),
        snapshot_hash=f"snapshot-{family_id}",
    )
    pending_case = EvaluationCase(
        case_id="case-pending",
        dataset_version="1.0.0",
        split=cast(Split, split),
        document_family_id=family_id,
        transformation_family_id=transformation_id,
        provenance=provenance,
        sources=sources,
        answer=answer_text,
        evaluation_units=(unit,),
        difficulty_tags=("science", transformation_id),
        generation=generation,
        review=review,
    )
    resolved_case_id = case_id or authoritative_case_id(pending_case)
    return pending_case.model_copy(update={"case_id": resolved_case_id})


def _build_unit(
    *,
    unit_id: str,
    answer: str,
    claims: tuple[ClaimAnnotation, ...],
) -> EvaluationUnit:
    return EvaluationUnit(
        unit_id=unit_id,
        answer_span=CharSpan(start=0, end=len(answer)),
        text=answer,
        claims=claims,
    )


def _build_claim(
    *,
    claim_id: str,
    answer: str,
    label: str = "entailed",
    requirement_id: str = "req-1",
    source_id: str = "source-1",
    spans: tuple[tuple[int, int], ...] = ((0, 5),),
    acceptable_retrieval_source_ids: tuple[str, ...] = ("source-1",),
) -> ClaimAnnotation:
    requirements = (
        CitationRequirement(
            requirement_id=requirement_id,
            alternatives=(
                CitationTarget(
                    source_id=source_id,
                    spans=tuple(CharSpan(start=start, end=end) for start, end in spans),
                ),
            ),
        ),
    )
    return ClaimAnnotation(
        claim_id=claim_id,
        answer_span=CharSpan(start=0, end=len(answer)),
        text=answer,
        label=cast(SupportLabel, label),
        citation_requirements=requirements if label == "entailed" else (),
        acceptable_retrieval_source_ids=acceptable_retrieval_source_ids,
    )


def _copy_case(case: EvaluationCase, **updates: object) -> EvaluationCase:
    payload = case.model_dump(mode="python", round_trip=True)
    payload.update(updates)
    return EvaluationCase.model_validate(payload)


def _queue_keys(queue: ReviewQueue) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (item.case.case_id, item.unit.unit_id, item.claim.claim_id)
        for item in queue.items
    )


def _strip_tags(value: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", value))


def _normalize_space(value: str) -> str:
    return " ".join(value.split())
