"""Grade cite-right's document attribution against labeled agent answers.

Each case is one answer sentence with a label:

- ``must_cite``: the sentence restates a fact from the listed document(s).
  No citation is a recall miss.
- ``may_cite``: inference, meta, or mixed sentence. A citation to a listed
  document is fine and a missing one is not penalised.
- ``must_not_cite``: no document supports the sentence (absent fact, swapped
  number or entity, contradiction).

A citation to a document outside ``docs`` is ``wrong`` under every label. The
suite is precision-first: tune for recall only while ``wrong`` stays zero.

Only spans whose status is ``supported`` count as citations. ``partial``
includes contradiction downgrades, so counting it would cite contradicted
claims.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cite_right import CitationConfig, SourceDocument, align_citations
from cite_right.core.interfaces import Tokenizer
from cite_right.models.base import Embedder

CASES_PATH = Path(__file__).with_name("cases.json")

Label = Literal["must_cite", "may_cite", "must_not_cite"]
Grade = Literal["hit", "miss", "ok", "wrong"]
Split = Literal["train", "dev"]

CONFIGS: dict[str, Callable[[], CitationConfig]] = {
    "default": CitationConfig,
    "strict": CitationConfig.strict,
}


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    group: str
    split: Split
    label: Label
    docs: tuple[str, ...]
    answer: str
    origin: str


@dataclass(frozen=True, slots=True)
class CaseResult:
    id: str
    group: str
    split: Split
    label: Label
    expected: tuple[str, ...]
    cited: tuple[str, ...]
    statuses: tuple[str, ...]
    grade: Grade
    latency_ms: float


@dataclass(frozen=True, slots=True)
class Summary:
    cases: int
    wrong: int
    hits: int
    must_cite: int
    must_not_cite_clean: int
    must_not_cite: int
    recall: float


def load_cases(path: Path = CASES_PATH) -> tuple[dict[str, str], tuple[Case, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = tuple(
        Case(
            id=case["id"],
            group=case["group"],
            split=case["split"],
            label=case["label"],
            docs=tuple(case["docs"]),
            answer=case["answer"],
            origin=case["origin"],
        )
        for case in payload["cases"]
    )
    return dict(payload["documents"]), cases


def grade_case(case: Case, cited: tuple[str, ...]) -> Grade:
    if any(document_id not in case.docs for document_id in cited):
        return "wrong"
    if case.label == "must_cite":
        return "hit" if cited else "miss"
    return "ok"


# A predictor returns the supported document ids and the per-span statuses.
Predictor = Callable[[Case], tuple[tuple[str, ...], tuple[str, ...]]]


def cite_right_predictor(
    documents: dict[str, str],
    *,
    config: CitationConfig,
    tokenizer: Tokenizer | None = None,
    embedder: Embedder | None = None,
) -> Predictor:
    sources = [
        SourceDocument(id=document_id, text=text)
        for document_id, text in sorted(documents.items())
    ]

    def predict(case: Case) -> tuple[tuple[str, ...], tuple[str, ...]]:
        spans = align_citations(
            case.answer,
            sources,
            config=config,
            tokenizer=tokenizer,
            embedder=embedder,
        )
        cited = {
            citation.source_id
            for span in spans
            if span.status == "supported"
            for citation in span.citations
        }
        return tuple(sorted(cited)), tuple(span.status for span in spans)

    return predict


def run_cases(cases: tuple[Case, ...], predict: Predictor) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in cases:
        started = time.perf_counter()
        cited, statuses = predict(case)
        latency_ms = (time.perf_counter() - started) * 1000
        results.append(
            CaseResult(
                id=case.id,
                group=case.group,
                split=case.split,
                label=case.label,
                expected=case.docs,
                cited=cited,
                statuses=statuses,
                grade=grade_case(case, cited),
                latency_ms=round(latency_ms, 1),
            )
        )
    return results


def summarize(results: list[CaseResult]) -> Summary:
    must_cite = [r for r in results if r.label == "must_cite"]
    must_not = [r for r in results if r.label == "must_not_cite"]
    hits = sum(r.grade == "hit" for r in must_cite)
    return Summary(
        cases=len(results),
        wrong=sum(r.grade == "wrong" for r in results),
        hits=hits,
        must_cite=len(must_cite),
        must_not_cite_clean=sum(r.grade == "ok" for r in must_not),
        must_not_cite=len(must_not),
        recall=round(hits / len(must_cite), 3) if must_cite else 0.0,
    )
