#!/usr/bin/env python3
"""Pick free OpenRouter models for OpenWiki CI.

OpenWiki is a tool-calling agent, so this prefers models whose
``supported_parameters`` include ``tools``. Only ids containing ``:free``
are considered (prompt/completion price is usually 0 on those).

Ranking, highest first:
  1. tool-calling support
  2. Artificial Analysis ``coding_index`` when present
  3. otherwise ``context_length``
  4. ``created`` as a tie-break

Prints the top N ids on stdout (default 8), one per line. Ranking details
go to stderr so CI logs show why a model was chosen.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_TOP_N = 8


def _coding_index(model: dict[str, Any]) -> float | None:
    benchmarks = model.get("benchmarks") or {}
    analysis = benchmarks.get("artificial_analysis") or {}
    value = analysis.get("coding_index")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _supports_tools(model: dict[str, Any]) -> bool:
    params = model.get("supported_parameters") or []
    return "tools" in params


def _sort_key(model: dict[str, Any]) -> tuple[int, int, float, int, int]:
    coding = _coding_index(model)
    context = int(model.get("context_length") or 0)
    created = int(model.get("created") or 0)
    return (
        1 if _supports_tools(model) else 0,
        1 if coding is not None else 0,
        coding if coding is not None else 0.0,
        context,
        created,
    )


def select_free_models(payload: dict[str, Any], top_n: int) -> list[dict[str, Any]]:
    models = payload.get("data") or []
    free = [
        m for m in models if isinstance(m, dict) and ":free" in str(m.get("id") or "")
    ]
    free.sort(key=_sort_key, reverse=True)
    return free[:top_n]


def fetch_models() -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "cite-right-openwiki-ci",
    }
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(MODELS_URL, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"OpenRouter models request failed: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"OpenRouter models request failed: {exc.reason}") from exc
    return json.loads(body.decode("utf-8"))


def load_payload(argv: list[str]) -> dict[str, Any]:
    if len(argv) >= 3 and argv[1] == "--from-json":
        with open(argv[2], encoding="utf-8") as handle:
            return json.load(handle)
    if len(argv) >= 2 and argv[1] == "--from-stdin":
        return json.load(sys.stdin)
    return fetch_models()


def main(argv: list[str]) -> int:
    top_n = int(os.environ.get("OPENWIKI_FREE_MODEL_COUNT", DEFAULT_TOP_N))
    payload = load_payload(argv)
    chosen = select_free_models(payload, top_n)
    if not chosen:
        print("No OpenRouter models with ':free' in the id.", file=sys.stderr)
        return 1
    print(
        "Free OpenRouter rotation (prefer tools, then coding_index/context/created):",
        file=sys.stderr,
    )
    for index, model in enumerate(chosen, start=1):
        coding = _coding_index(model)
        coding_text = f"{coding:.1f}" if coding is not None else "n/a"
        print(
            f"  {index}. {model.get('id')}  tools={int(_supports_tools(model))}  "
            f"coding_index={coding_text}  context_length={model.get('context_length')}  "
            f"created={model.get('created')}",
            file=sys.stderr,
        )
    for model in chosen:
        model_id = model.get("id")
        if model_id:
            print(model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
