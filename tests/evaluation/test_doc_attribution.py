import json
from pathlib import Path

from evaluation.doc_attribution import main


def test_status_reason_histogram_reports_each_split(capsys) -> None:
    bundle = Path(__file__).parent / "fixtures" / "tuning"
    assert main(["--tuning-bundle", str(bundle)]) == 0
    expected = {
        "supported": 1,
        "below_coverage": 0,
        "contradiction": 0,
        "no_candidate": 0,
    }
    assert json.loads(capsys.readouterr().out) == {"train": expected, "dev": expected}
