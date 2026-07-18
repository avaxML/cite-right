"""Minimal isolated worker entrypoint for train/dev tuning bundles."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

from evaluation.canonical import canonical_json_bytes
from evaluation.tuning_bundle import load_tuning_bundle

_SCRUBBED_ENVIRONMENT_VARIABLES = (
    "CITE_RIGHT_ATTESTATION_KEY_FILE",
    "CITE_RIGHT_HOLDOUT_KEY_FILE",
)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        print("usage: python -m evaluation.worker", file=sys.stderr)
        print("evaluation.worker accepts no positional or option arguments", file=sys.stderr)
        return 2

    leaked = [name for name in _SCRUBBED_ENVIRONMENT_VARIABLES if name in os.environ]
    if leaked:
        print("sensitive holdout environment variables were exposed to the tuning worker", file=sys.stderr)
        return 1

    bundle = load_tuning_bundle(Path.cwd())
    payload = {
        "dev_case_count": len(bundle.dev_cases),
        "ok": True,
        "train_case_count": len(bundle.train_cases),
    }
    sys.stdout.write(canonical_json_bytes(payload).decode("utf-8"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
