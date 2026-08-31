#!/usr/bin/env python3
"""Drive a multi-turn RAG citation session the way a product would.

This is the verification harness, not a unit test. It:

1. Retrieves a per-turn source set from a small in-process corpus (stand-in
   for a vector store).
2. Aligns a generated answer against *that turn's* retrieved sources only.
3. Checks character-offset invariants and source isolation across turns.
4. Writes JSON evidence under VERIFY_EVIDENCE_DIR (default:
   /tmp/cite-right-verify-$RUN_ID).

It never starts a server. Cleanup must not delete the evidence directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]


def _ensure_src_path() -> None:
    src = str(REPO_ROOT / "src")
    if (REPO_ROOT / "pyproject.toml").is_file() and src not in sys.path:
        sys.path.insert(0, src)


_ensure_src_path()

from cite_right import (  # noqa: E402
    PreparedCitationCorpus,
    SourceDocument,
    align_citations,
    annotate_answer,
    format_with_citations,
    get_citation_summary,
)


CORPUS: list[SourceDocument] = [
    SourceDocument(
        id="q4_earnings",
        text=(
            "During the Q4 earnings call, CEO Jane Smith noted that revenue "
            "reached 5.2 billion dollars, exceeding analyst expectations."
        ),
    ),
    SourceDocument(
        id="press_release",
        text=(
            "Acme Corporation today announced fourth quarter revenue of "
            "5.2 billion dollars, a new company record."
        ),
    ),
    SourceDocument(
        id="europe_sales",
        text="Sales in Europe surpassed all projections by 15 percent in the quarter.",
    ),
    SourceDocument(
        id="product_line",
        text="The new product line launched in March drove significant growth.",
    ),
]


def retrieve(query: str, *, k: int = 2) -> list[SourceDocument]:
    """Tiny lexical retriever. Product RAG would call a vector store here."""
    tokens = {t.lower() for t in query.split() if len(t) > 3}

    def score(doc: SourceDocument) -> int:
        hay = set(doc.text.lower().split()) | set(doc.id.lower().replace("_", " ").split())
        return sum(1 for t in tokens if t in hay or t in doc.id.lower())

    ranked = sorted(CORPUS, key=score, reverse=True)
    return [d for d in ranked if score(d) > 0][:k] or ranked[:k]


@dataclass
class TurnSpec:
    turn_id: str
    question: str
    answer: str
    retrieve_query: str | None = None
    expected_status: str | None = None
    required_source_ids: frozenset[str] | None = None
    forbidden_source_ids: frozenset[str] | None = None
    expect_empty_citations: bool = False


TURNS: list[TurnSpec] = [
    TurnSpec(
        turn_id="turn1_q4_revenue",
        question="What was Acme Q4 revenue?",
        answer="Revenue reached 5.2 billion dollars, exceeding analyst expectations.",
        retrieve_query="Acme Q4 revenue earnings call Jane Smith",
        expected_status="supported",
        required_source_ids=frozenset({"q4_earnings", "press_release"}),
        forbidden_source_ids=frozenset({"europe_sales"}),
    ),
    TurnSpec(
        turn_id="turn2_europe_followup",
        question="What about European sales?",
        answer="Sales in Europe surpassed all projections by 15 percent.",
        retrieve_query="European sales Europe projections",
        expected_status="supported",
        required_source_ids=frozenset({"europe_sales"}),
        forbidden_source_ids=frozenset({"q4_earnings", "press_release"}),
    ),
    TurnSpec(
        turn_id="turn3_hallucinated_followup",
        question="Did they colonize Mars?",
        answer="The company announced plans to colonize Mars this year.",
        retrieve_query="Acme Q4 revenue earnings",
        expected_status="unsupported",
        expect_empty_citations=True,
        forbidden_source_ids=frozenset({"q4_earnings", "press_release", "europe_sales"}),
    ),
    TurnSpec(
        turn_id="turn4_conversational_wrapper",
        question="Can you recap that?",
        answer="Sure!",
        retrieve_query="Acme Q4 revenue earnings",
        expected_status="unsupported",
        expect_empty_citations=True,
    ),
]


def _serialize_span(span_citations: Any, source_by_id: dict[str, SourceDocument]) -> dict[str, Any]:
    citations = []
    for c in span_citations.citations:
        source = source_by_id.get(c.source_id)
        sliced = source.text[c.char_start : c.char_end] if source else None
        citations.append(
            {
                "source_id": c.source_id,
                "score": c.score,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "evidence": c.evidence,
                "offset_invariant": sliced == c.evidence if source is not None else False,
                "answer_coverage": c.components.get("answer_coverage"),
            }
        )
    span = span_citations.answer_span
    return {
        "text": span.text,
        "char_start": span.char_start,
        "char_end": span.char_end,
        "status": span_citations.status,
        "citations": citations,
        "retrieval_support_count": len(span_citations.retrieval_support),
    }


def _check_invariants(
    answer: str,
    sources: list[SourceDocument],
    results: list[Any],
) -> list[str]:
    failures: list[str] = []
    source_by_id = {s.id: s for s in sources}
    for result in results:
        span = result.answer_span
        if answer[span.char_start : span.char_end] != span.text:
            failures.append(
                f"answer span mismatch at [{span.char_start}:{span.char_end}]"
            )
        for citation in result.citations:
            source = source_by_id.get(citation.source_id)
            if source is None:
                failures.append(f"citation source_id {citation.source_id!r} not in this turn's sources")
                continue
            sliced = source.text[citation.char_start : citation.char_end]
            if sliced != citation.evidence:
                failures.append(
                    f"{citation.source_id} offset invariant failed: "
                    f"{sliced!r} != {citation.evidence!r}"
                )
    return failures


def drive_turn(spec: TurnSpec) -> dict[str, Any]:
    query = spec.retrieve_query or spec.question
    sources = retrieve(query)
    source_ids = [s.id for s in sources]
    results = align_citations(spec.answer, sources)
    source_by_id = {s.id: s for s in sources}
    cited_ids = {c.source_id for r in results for c in r.citations}
    statuses = [r.status for r in results]
    failures = _check_invariants(spec.answer, sources, results)

    if spec.expected_status and spec.expected_status not in statuses:
        if spec.expected_status == "unsupported":
            if any(s != "unsupported" for s in statuses):
                failures.append(f"expected unsupported, got {statuses}")
        else:
            failures.append(f"expected status {spec.expected_status!r} in {statuses}")

    if spec.expect_empty_citations and cited_ids:
        failures.append(f"expected no citations, got {sorted(cited_ids)}")

    if spec.required_source_ids is not None:
        # At least one required source must be *retrievable* and cited when we
        # expect support. Retrieval is a stand-in; fail loudly if the retriever
        # did not even return a required doc.
        retrieved = set(source_ids)
        if not (spec.required_source_ids & retrieved):
            failures.append(
                f"retriever missed required sources {sorted(spec.required_source_ids)}; "
                f"got {source_ids}"
            )
        if spec.expected_status == "supported" and not (cited_ids & spec.required_source_ids):
            failures.append(
                f"citations {sorted(cited_ids)} did not include required "
                f"{sorted(spec.required_source_ids)}"
            )

    if spec.forbidden_source_ids is not None and (cited_ids & spec.forbidden_source_ids):
        failures.append(
            f"cited forbidden leftover sources {sorted(cited_ids & spec.forbidden_source_ids)}"
        )

    annotated = annotate_answer(spec.answer, sources, format="markdown")
    formatted = format_with_citations(spec.answer, results, format="markdown")
    summary = get_citation_summary(results)

    return {
        "turn_id": spec.turn_id,
        "question": spec.question,
        "answer": spec.answer,
        "retrieved_source_ids": source_ids,
        "cited_source_ids": sorted(cited_ids),
        "statuses": statuses,
        "spans": [_serialize_span(r, source_by_id) for r in results],
        "annotated_answer": annotated,
        "formatted_answer": formatted,
        "summary": summary,
        "failures": failures,
        "ok": not failures,
    }


def drive_prepared_corpus() -> dict[str, Any]:
    corpus = PreparedCitationCorpus.from_sources(CORPUS)
    answers = [
        "Revenue reached 5.2 billion dollars, exceeding analyst expectations.",
        "Sales in Europe surpassed all projections by 15 percent.",
    ]
    turns = []
    failures: list[str] = []
    for i, answer in enumerate(answers, start=1):
        results = corpus.align(answer)
        source_by_id = {s.id: s for s in CORPUS}
        inv = _check_invariants(answer, list(CORPUS), results)
        failures.extend(inv)
        statuses = [r.status for r in results]
        if "supported" not in statuses:
            failures.append(f"prepared corpus answer {i} not supported: {statuses}")
        turns.append(
            {
                "answer": answer,
                "statuses": statuses,
                "spans": [_serialize_span(r, source_by_id) for r in results],
            }
        )
    return {"turns": turns, "failures": failures, "ok": not failures}


def evidence_dir(run_id: str) -> Path:
    base = os.environ.get("VERIFY_EVIDENCE_DIR")
    if base:
        path = Path(base)
    else:
        path = Path(f"/tmp/cite-right-verify-{run_id}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature",
        choices=[
            "per-turn-alignment",
            "follow-up-source-isolation",
            "prepared-corpus-session",
            "ungrounded-follow-up",
            "annotated-footnotes",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--run-id", default=os.environ.get("VERIFY_RUN_ID") or str(int(time.time())))
    args = parser.parse_args()

    out_dir = evidence_dir(args.run_id)
    started = time.time()

    turn_results = [drive_turn(spec) for spec in TURNS]
    prepared = drive_prepared_corpus()

    feature_ok: dict[str, bool] = {
        "per-turn-alignment": turn_results[0]["ok"],
        "follow-up-source-isolation": turn_results[1]["ok"] and turn_results[0]["ok"],
        "ungrounded-follow-up": turn_results[2]["ok"] and turn_results[3]["ok"],
        "annotated-footnotes": all(
            isinstance(t["annotated_answer"], str)
            and "[1]" in t["annotated_answer"]
            and t["formatted_answer"] == t["annotated_answer"]
            for t in turn_results[:2]
        )
        and turn_results[0]["ok"],
        "prepared-corpus-session": prepared["ok"],
    }

    if args.feature != "all":
        selected = {
            "per-turn-alignment": [turn_results[0]],
            "follow-up-source-isolation": turn_results[:2],
            "ungrounded-follow-up": turn_results[2:],
            "annotated-footnotes": turn_results[:2],
            "prepared-corpus-session": [],
        }[args.feature]
        ok = feature_ok[args.feature]
        payload = {
            "feature": args.feature,
            "run_id": args.run_id,
            "ok": ok,
            "turns": selected,
            "prepared_corpus": prepared if args.feature == "prepared-corpus-session" else None,
        }
    else:
        ok = all(feature_ok.values())
        payload = {
            "feature": "all",
            "run_id": args.run_id,
            "ok": ok,
            "feature_ok": feature_ok,
            "turns": turn_results,
            "prepared_corpus": prepared,
        }

    payload["elapsed_ms"] = round((time.time() - started) * 1000, 2)
    payload["evidence_dir"] = str(out_dir)

    out_path = out_dir / f"{args.feature}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary_path = out_dir / "session-summary.txt"
    lines = [
        f"run_id={args.run_id}",
        f"feature={args.feature}",
        f"ok={ok}",
        f"evidence={out_path}",
    ]
    for t in payload.get("turns") or []:
        lines.append(
            f"{t['turn_id']}: statuses={t['statuses']} cited={t['cited_source_ids']} ok={t['ok']}"
        )
        if t.get("failures"):
            lines.append(f"  failures: {t['failures']}")
    if payload.get("prepared_corpus"):
        lines.append(f"prepared_corpus ok={prepared['ok']} failures={prepared['failures']}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"ok": ok, "evidence": str(out_path), "summary": str(summary_path)}, indent=2))
    if not ok:
        print(summary_path.read_text(encoding="utf-8"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
