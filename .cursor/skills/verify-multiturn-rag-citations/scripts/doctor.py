#!/usr/bin/env python3
"""Read-only health check for Cite-Right multi-turn RAG citation verification.

Answers: is this checkout worth driving? Prints a JSON report and exits 0 only
when cite_right imports from this repo and the offset contract can be checked
with a one-span self-test. Does not start servers or mutate sources.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC = REPO_ROOT / "src"


def _repo_root() -> Path:
    if (REPO_ROOT / "pyproject.toml").is_file() and SRC.is_dir():
        return REPO_ROOT
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").is_file():
        return cwd
    raise SystemExit("Could not find cite-right repo root (missing pyproject.toml).")


def main() -> int:
    root = _repo_root()
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    report: dict[str, object] = {
        "repo_root": str(root),
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
    }

    try:
        import cite_right
        from cite_right import SourceDocument, align_citations
        from cite_right.citations import HAS_RUST_CORE
    except Exception as exc:  # noqa: BLE001 — doctor must report any import failure
        report["ok"] = False
        report["error"] = f"cite_right import failed: {exc}"
        print(json.dumps(report, indent=2))
        return 1

    report["cite_right_version"] = cite_right.__version__
    report["cite_right_file"] = cite_right.__file__
    report["has_rust_core"] = bool(HAS_RUST_CORE)
    from_checkout = Path(cite_right.__file__).resolve().is_relative_to(root.resolve())
    report["imported_from_checkout"] = from_checkout

    source = SourceDocument(
        id="doctor_source",
        text="Acme Corporation reported revenue of 5.2 billion dollars in 2024.",
    )
    answer = "Acme Corporation reported revenue of 5.2 billion dollars in 2024."
    results = align_citations(answer, [source])
    if not results or results[0].status != "supported" or not results[0].citations:
        report["ok"] = False
        report["error"] = "self-test align_citations did not return a supported citation"
        report["statuses"] = [r.status for r in results]
        print(json.dumps(report, indent=2))
        return 1

    citation = results[0].citations[0]
    sliced = source.text[citation.char_start : citation.char_end]
    offset_ok = sliced == citation.evidence
    span = results[0].answer_span
    span_ok = answer[span.char_start : span.char_end] == span.text
    report["self_test"] = {
        "status": results[0].status,
        "source_id": citation.source_id,
        "offset_invariant": offset_ok,
        "answer_span_invariant": span_ok,
    }
    if not offset_ok or not span_ok:
        report["ok"] = False
        report["error"] = "self-test offset invariant failed"
        print(json.dumps(report, indent=2))
        return 1

    demo_url = os.environ.get("CITE_RIGHT_VERIFY_DEMO_URL")
    if demo_url:
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(demo_url.rstrip("/") + "/api/citations", timeout=3) as resp:
                report["demo"] = {"url": demo_url, "status": resp.status, "reachable": True}
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            report["ok"] = False
            report["demo"] = {"url": demo_url, "reachable": False, "error": str(exc)}
            print(json.dumps(report, indent=2))
            return 1

    report["ok"] = True
    if not from_checkout:
        report["warning"] = (
            "cite_right did not import from this checkout. "
            "Install the local package before driving (see Launch in SKILL.md)."
        )
    print(json.dumps(report, indent=2))
    return 0 if from_checkout else 1


if __name__ == "__main__":
    raise SystemExit(main())
