"""Deterministic canonical serialization helpers for evaluation datasets."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

from pydantic import BaseModel

from evaluation.schema import EvaluationCase


def canonical_json_bytes(value: BaseModel | Mapping[str, object]) -> bytes:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def authoritative_case_id(
    case_or_authoritative_mapping: EvaluationCase | Mapping[str, object],
) -> str:
    payload = authoritative_case_payload(case_or_authoritative_mapping)
    digest = sha256_hex(canonical_json_bytes(payload))
    return f"case-{digest[:20]}"


def authoritative_case_payload(
    case_or_authoritative_mapping: EvaluationCase | Mapping[str, object],
) -> dict[str, object]:
    payload = (
        case_or_authoritative_mapping.model_dump(mode="json")
        if isinstance(case_or_authoritative_mapping, BaseModel)
        else EvaluationCase.model_validate(case_or_authoritative_mapping).model_dump(
            mode="json"
        )
    )
    payload.pop("case_id", None)
    payload.pop("split", None)
    payload.pop("review", None)
    return payload
