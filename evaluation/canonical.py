"""Deterministic canonical serialization helpers for evaluation datasets."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

from pydantic import BaseModel

from evaluation.schema import EvaluationCase


def canonical_json_bytes(value: BaseModel | Mapping[str, object]) -> bytes:
    payload = (
        value.model_dump(mode="json")
        if isinstance(value, BaseModel)
        else _normalize_mapping(value)
    )
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
    if isinstance(case_or_authoritative_mapping, EvaluationCase):
        payload = case_or_authoritative_mapping.model_dump(mode="json")
    elif isinstance(case_or_authoritative_mapping, BaseModel):
        raise TypeError(
            "authoritative_case_id accepts EvaluationCase or Mapping[str, object] inputs"
        )
    else:
        payload = EvaluationCase.model_validate(case_or_authoritative_mapping).model_dump(
            mode="json"
        )
    payload.pop("case_id", None)
    payload.pop("split", None)
    payload.pop("review", None)
    return payload


def _normalize_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return {key: _normalize_json_value(item) for key, item in value.items()}


def _normalize_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _normalize_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    return value
