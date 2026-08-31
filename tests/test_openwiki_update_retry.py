"""Retry classification for scripts/openwiki_update.sh."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "openwiki_update.sh"


def _check(tmp_path: Path, text: str) -> int:
    log = tmp_path / "openwiki.log"
    log.write_text(text, encoding="utf-8")
    result = subprocess.run(
        ["bash", str(SCRIPT), "--check-retryable", str(log)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode


@pytest.mark.parametrize(
    "text",
    [
        "Cannot read properties of undefined (reading '0')\n",
        "TypeError: Cannot read properties of undefined (reading '0')\n",
        "Cannot read property '0' of undefined\n",
        "undefined is not an object (evaluating 'choices[0]')\n",
        "Provider returned error | metadata: {\"retry_after_seconds\":5}\n"
        "rate-limited upstream\n",
        "HTTP 429 Too Many Requests\n",
        "Error: 403 Forbidden\n only available on agentic harnesses\n",
    ],
)
def test_retryable_logs(tmp_path: Path, text: str) -> None:
    assert _check(tmp_path, text) == 0


@pytest.mark.parametrize(
    "text",
    [
        "ENOENT: no such file or directory, open 'openwiki/INSTRUCTIONS.md'\n",
        "SyntaxError: Unexpected token\n",
        "Error: 500 Internal Server Error\n",
        "",
    ],
)
def test_non_retryable_logs(tmp_path: Path, text: str) -> None:
    assert _check(tmp_path, text) != 0
