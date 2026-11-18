"""Print citation status-reason counts for the public train and dev splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import cast, get_args

from cite_right import CitationConfig
from cite_right.core.results import StatusReason
from evaluation.runner import Backend, execute_case
from evaluation.tuning_bundle import load_tuning_bundle


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.doc_attribution")
    parser.add_argument("--tuning-bundle", required=True, type=Path)
    parser.add_argument("--backend", choices=("python", "rust"), default="python")
    args = parser.parse_args(argv)
    backend = cast(Backend, args.backend)

    bundle = load_tuning_bundle(args.tuning_bundle)
    config = CitationConfig.strict().model_copy(
        update={
            "require_all_answer_tokens_in_evidence": False,
            "report_possible_source": True,
        }
    )
    histograms: dict[str, Counter[str]] = {}
    for split, cases in (("train", bundle.train_cases), ("dev", bundle.dev_cases)):
        counts: Counter[str] = Counter(dict.fromkeys(get_args(StatusReason), 0))
        for case in cases:
            run = execute_case(case=case, backend=backend, config=config)
            if run.error is not None:
                raise RuntimeError(f"{case.case_id}: {run.error.message}")
            for span in run.outputs:
                if span.status_reason is None:
                    raise RuntimeError(f"{case.case_id}: missing span status reason")
                counts[span.status_reason] += 1
        histograms[split] = counts

    print(json.dumps(histograms, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
