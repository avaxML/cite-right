"""Run the document-attribution suite.

uv run python -m evaluation.doc_attribution --config default
uv run python -m evaluation.doc_attribution --config strict --split dev
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from evaluation.doc_attribution.harness import (
    CONFIGS,
    cite_right_predictor,
    load_cases,
    run_cases,
    summarize,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.doc_attribution")
    parser.add_argument("--config", choices=sorted(CONFIGS), default="default")
    parser.add_argument("--split", choices=["train", "dev", "all"], default="all")
    parser.add_argument(
        "--embedder",
        action="store_true",
        help="use SentenceTransformerEmbedder (all-MiniLM-L6-v2) for retrieval",
    )
    parser.add_argument("--output", type=Path, help="write per-case results JSONL")
    parser.add_argument(
        "--fail-on-wrong",
        action="store_true",
        help="exit 1 when any case cites a document outside its labels",
    )
    args = parser.parse_args(argv)

    documents, cases = load_cases()
    if args.split != "all":
        cases = tuple(case for case in cases if case.split == args.split)
    embedder = None
    if args.embedder:
        from cite_right import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    predict = cite_right_predictor(
        documents, config=CONFIGS[args.config](), embedder=embedder
    )
    results = run_cases(cases, predict)
    summary = summarize(results)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(asdict(r)) + "\n" for r in results), encoding="utf-8"
        )
    print(
        f"config={args.config} split={args.split} embedder={args.embedder} "
        f"cases={summary.cases} wrong={summary.wrong} "
        f"recall={summary.hits}/{summary.must_cite} ({summary.recall}) "
        f"must_not_cite_clean={summary.must_not_cite_clean}/{summary.must_not_cite}"
    )
    return 1 if args.fail_on_wrong and summary.wrong else 0


if __name__ == "__main__":
    sys.exit(main())
