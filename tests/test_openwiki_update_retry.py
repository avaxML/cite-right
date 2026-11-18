"""Retry classification for scripts/openwiki_update.sh."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "openwiki_update.sh"


def _bash() -> str:
    if sys.platform != "win32":
        return "bash"
    git_exec_path = subprocess.check_output(
        ["git", "--exec-path"], text=True, encoding="utf-8"
    ).strip()
    return str(Path(git_exec_path).parents[2] / "bin" / "bash.exe")


def _check(tmp_path: Path, text: str) -> subprocess.CompletedProcess[str]:
    log = tmp_path / "openwiki.log"
    log.write_text(text, encoding="utf-8")
    result = subprocess.run(
        [_bash(), SCRIPT.as_posix(), "--check-retryable", log.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )
    return result


@pytest.mark.parametrize(
    "text",
    [
        "Cannot read properties of undefined (reading '0')\n",
        "TypeError: Cannot read properties of undefined (reading '0')\n",
        "Cannot read property '0' of undefined\n",
        'Cannot read properties of undefined (reading "0")\n',
        "undefined is not an object (evaluating 'choices[0]')\n",
        "undefined is not an object (evaluating 'content[0]')\n",
        (
            "Provider returned error | metadata:"
            ' {"retry_after_seconds":5}\n'
            "rate-limited upstream\n"
        ),
        "HTTP 429 Too Many Requests\n",
        "Error: 403 Forbidden\n only available on agentic harnesses\n",
    ],
)
def test_retryable_logs(tmp_path: Path, text: str) -> None:
    result = _check(tmp_path, text)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "text",
    [
        "ENOENT: no such file or directory, open 'openwiki/INSTRUCTIONS.md'\n",
        "SyntaxError: Unexpected token\n",
        "Error: 500 Internal Server Error\n",
        "Cannot read properties of undefined (reading 'config')\n",
        "Cannot read properties of undefined (reading 'length')\n",
        "undefined is not an object (evaluating 'config.model')\n",
        "",
    ],
)
def test_non_retryable_logs(tmp_path: Path, text: str) -> None:
    result = _check(tmp_path, text)
    assert result.returncode == 1, result.stderr
