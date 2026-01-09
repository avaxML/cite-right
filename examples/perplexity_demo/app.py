from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from cite_right import CitationConfig, PySBDSegmenter, SpacyAnswerSegmenter, align_citations

from .example_data import ANSWER, QUESTION, SOURCES


app = FastAPI(title="Cite-Right Perplexity Demo")


def _init_segmenters() -> tuple[SpacyAnswerSegmenter | None, PySBDSegmenter | None]:
    try:
        return SpacyAnswerSegmenter(split_clauses=True), PySBDSegmenter()
    except RuntimeError as exc:
        print(f"Segmenter fallback: {exc}")
        return None, None


ANSWER_SEGMENTER, SOURCE_SEGMENTER = _init_segmenters()


def _build_citations_payload() -> dict[str, Any]:
    config = CitationConfig(top_k=2, allow_embedding_only=False)
    results = align_citations(
        ANSWER,
        SOURCES,
        config=config,
        answer_segmenter=ANSWER_SEGMENTER,
        source_segmenter=SOURCE_SEGMENTER,
    )

    spans: list[dict[str, Any]] = []
    for span_index, span in enumerate(results):
        citations = [
            {
                "source_id": SOURCES[citation.source_index].id,
                "evidence": citation.evidence,
                "char_start": citation.char_start,
                "char_end": citation.char_end,
                "score": citation.score,
            }
            for citation in span.citations
        ]
        spans.append(
            {
                "id": span_index,
                "text": span.answer_span.text,
                "status": span.status,
                "citations": citations,
            }
        )

    sources = [
        {"id": source.id, "text": source.text, "index": index}
        for index, source in enumerate(SOURCES)
    ]

    return {"question": QUESTION, "answer": ANSWER, "spans": spans, "sources": sources}


@app.get("/api/citations")
def get_citations() -> JSONResponse:
    payload = _build_citations_payload()
    return JSONResponse(payload)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = Path(__file__).with_name("index.html")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


def main() -> None:
    print(json.dumps(_build_citations_payload(), indent=2))


if __name__ == "__main__":
    main()
