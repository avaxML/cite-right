from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
from collections import UserDict
from datetime import date
from types import MappingProxyType
from typing import Any, cast

import pytest
from pydantic import BaseModel

from evaluation.canonical import authoritative_case_id, canonical_json_bytes, sha256_hex
from evaluation.schema import EvaluationCase


def test_canonical_json_bytes_ignores_dictionary_insertion_order() -> None:
    first = {"b": 2, "a": {"delta": 4, "charlie": 3}}
    second = {"a": {"charlie": 3, "delta": 4}, "b": 2}

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_json_bytes(first) == b'{"a":{"charlie":3,"delta":4},"b":2}'


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_json_bytes_rejects_non_finite_floats(value: float) -> None:
    with pytest.raises(
        ValueError, match="Out of range float values are not JSON compliant"
    ):
        canonical_json_bytes({"value": value})


def test_canonical_json_bytes_keeps_finite_float_representation_stable() -> None:
    payload = {"small": 0.000001, "regular": 1.25, "large": 1000000.0}

    assert (
        canonical_json_bytes(payload)
        == b'{"large":1000000.0,"regular":1.25,"small":1e-06}'
    )


def test_canonical_json_bytes_emits_literal_utf8_without_ascii_escaping() -> None:
    payload = {"greeting": "Țară café 東京"}

    assert canonical_json_bytes(payload) == '{"greeting":"Țară café 東京"}'.encode(
        "utf-8"
    )


def test_canonical_json_bytes_normalizes_non_plain_mappings_recursively() -> None:
    nested = UserDict({"delta": 4, "charlie": 3})
    payload = MappingProxyType({"b": 2, "a": nested})

    assert canonical_json_bytes(payload) == b'{"a":{"charlie":3,"delta":4},"b":2}'


def test_canonical_json_bytes_normalizes_nested_lists_and_tuples_recursively() -> None:
    payload = {
        "items": (
            {"values": [3, 2, 1]},
            {"values": ({"beta": 2, "alpha": 1},)},
        )
    }

    assert (
        canonical_json_bytes(payload)
        == b'{"items":[{"values":[3,2,1]},{"values":[{"alpha":1,"beta":2}]}]}'
    )


def test_canonical_json_bytes_rejects_nested_non_finite_floats_after_normalization() -> (
    None
):
    payload = MappingProxyType({"outer": UserDict({"value": math.inf})})

    with pytest.raises(
        ValueError, match="Out of range float values are not JSON compliant"
    ):
        canonical_json_bytes(payload)


@pytest.mark.parametrize("value", ["text", b"bytes"])
def test_canonical_json_bytes_rejects_top_level_string_and_bytes_inputs(
    value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="canonical_json_bytes accepts BaseModel, mapping, list, or tuple inputs",
    ):
        canonical_json_bytes(cast(Any, value))


def test_canonical_json_bytes_rejects_nested_non_list_tuple_sequences() -> None:
    payload = {"items": range(3)}

    with pytest.raises(
        TypeError, match="canonical JSON arrays must be list or tuple instances"
    ):
        canonical_json_bytes(payload)


def test_authoritative_case_id_changes_when_authoritative_content_changes() -> None:
    original = _make_case_data()
    changed = _make_case_data()
    changed["dataset_version"] = "1.0.1"

    assert authoritative_case_id(original) != authoritative_case_id(changed)


def test_authoritative_case_id_ignores_operational_metadata() -> None:
    original = _make_case_data()
    changed = _make_case_data()
    changed["case_id"] = "case-operational"
    changed["split"] = "holdout"
    changed["review"] = {
        "state": "rejected",
        "reviewer": "auditor",
        "reviewed_at": date(2026, 7, 17),
        "notes": "Operational review metadata must not affect canonical identity.",
    }

    assert authoritative_case_id(original) == authoritative_case_id(changed)


def test_authoritative_case_id_supports_evaluation_case_instances() -> None:
    case = EvaluationCase.model_validate(_make_case_data())
    equivalent = _make_case_data()
    equivalent["case_id"] = "different-case-id"
    equivalent["split"] = "train"
    equivalent["review"] = None

    assert authoritative_case_id(case) == authoritative_case_id(equivalent)


def test_authoritative_case_id_rejects_unrelated_base_models() -> None:
    model = _UnrelatedModel(name="not-a-case")

    with pytest.raises(
        TypeError,
        match="authoritative_case_id accepts EvaluationCase or Mapping\\[str, object\\] inputs",
    ):
        authoritative_case_id(cast(Any, model))


def test_sha256_hex_returns_full_deterministic_digest() -> None:
    data = canonical_json_bytes({"answer": "Paris is in France."})

    digest = sha256_hex(data)

    assert digest == "1406dfad4151852b076aa512d670a2083718882ea0eda4125588e345687d1152"
    assert len(digest) == 64
    assert digest == digest.lower()


@pytest.mark.parametrize(
    ("payload_literal", "hash_seed"),
    [
        ('{"zeta": 1, "alpha": {"beta": 2, "aardvark": 3}}', "1"),
        ('{"alpha": {"aardvark": 3, "beta": 2}, "zeta": 1}', "777"),
    ],
)
def test_subprocess_hash_is_deterministic_across_fresh_processes(
    payload_literal: str, hash_seed: str
) -> None:
    digest = _hash_in_subprocess(payload_literal=payload_literal, hash_seed=hash_seed)

    assert digest == "3d33f41a9b6e2ea7fe9f825aadb088f23a685d180aba030a9a5e149588b892c9"


def _hash_in_subprocess(*, payload_literal: str, hash_seed: str) -> str:
    script = textwrap.dedent(
        f"""
        from evaluation.canonical import canonical_json_bytes, sha256_hex

        payload = {payload_literal}
        print(sha256_hex(canonical_json_bytes(payload)))
        """
    )
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = hash_seed
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return completed.stdout.strip()


class _UnrelatedModel(BaseModel):
    name: str


def _make_case_data() -> dict[str, Any]:
    source_text = "Paris is in France."
    answer = "Paris is in France, and it is the capital city."

    return {
        "case_id": "case-local",
        "dataset_version": "1.0.0",
        "split": "dev",
        "document_family_id": "family-paris",
        "transformation_family_id": "transformation-summary",
        "provenance": {
            "kind": "authored",
            "title": "Paris facts",
            "origin": "internal",
            "publisher": "Cite-Right",
            "license": "permissive",
            "retrieval_date": date(2026, 7, 17),
            "snapshot_hash": "snapshot-123",
        },
        "sources": (
            {
                "source_id": "source-paris",
                "text": source_text,
                "chunk_id": "chunk-1",
                "chunk_char_start": 0,
                "chunk_char_end": len(source_text),
            },
        ),
        "answer": answer,
        "evaluation_units": (
            {
                "unit_id": "unit-1",
                "answer_span": {"start": 0, "end": len(answer)},
                "text": answer,
                "claims": (
                    {
                        "claim_id": "claim-1",
                        "answer_span": {"start": 0, "end": len("Paris is in France")},
                        "text": "Paris is in France",
                        "label": "entailed",
                        "citation_requirements": (
                            {
                                "requirement_id": "req-1",
                                "alternatives": (
                                    {
                                        "source_id": "source-paris",
                                        "spans": (
                                            {"start": 0, "end": len(source_text)},
                                        ),
                                    },
                                ),
                            },
                        ),
                        "acceptable_retrieval_source_ids": ("source-paris",),
                        "requires_non_contiguous_evidence": False,
                    },
                ),
            },
        ),
        "difficulty_tags": ("single_source", "city_fact"),
        "generation": {
            "recipe_id": "recipe-123",
            "generator_name": "hand-authored",
            "prompt_version": "v1",
            "seed": 17,
            "notes": "Simple authored case.",
        },
        "review": {
            "state": "approved",
            "reviewer": "reviewer-1",
            "reviewed_at": date(2026, 7, 17),
            "notes": "Approved.",
        },
    }
