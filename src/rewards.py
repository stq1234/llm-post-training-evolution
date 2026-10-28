"""Strict JSON and evidence-grounding rewards for synthetic QA rows."""

from __future__ import annotations

import json
import re
from typing import Any


class _DuplicateJSONKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of silently keeping the last value."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(key)
        result[key] = value
    return result


def _json_object(text: str) -> dict[str, Any] | None:
    """Parse exactly one JSON object; surrounding text and duplicate keys fail."""
    if not isinstance(text, str):
        return None
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TypeError, _DuplicateJSONKey):
        return None
    return value if isinstance(value, dict) else None


def _schema_valid(value: dict[str, Any] | None) -> bool:
    if value is None or set(value) != {"answer", "evidence"}:
        return False
    if not isinstance(value["answer"], str) or not value["answer"]:
        return False
    evidence = value["evidence"]
    if not isinstance(evidence, list):
        return False
    # bool is a subclass of int in Python, but is not a citation ID.
    return (
        all(type(item) is int and item > 0 for item in evidence)
        and len(evidence) == len(set(evidence))
    )


def _format_valid(
    text: str,
    parsed: dict[str, Any] | None = None,
    max_evidence_id: int | None = None,
) -> bool:
    """Check the exact JSON schema, independent of whitespace and key order.

    A row-aware caller can also constrain citations to its numbered materials.
    A no-gold check cannot infer that upper bound.
    """
    if parsed is None:
        parsed = _json_object(text)
    if not _schema_valid(parsed):
        return False
    if max_evidence_id is not None and any(i > max_evidence_id for i in parsed["evidence"]):
        return False
    return True


def format_only_reward(text: str) -> float:
    """Binary JSON/schema-only objective, independent of the gold row."""
    return 1.0 if _format_valid(text) else 0.0


def lenient_answer(text: str) -> str | None:
    """Extract an answer for diagnostics only; never used by reward scoring."""
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
    try:
        value = json.loads(candidate)
        if isinstance(value, dict) and isinstance(value.get("answer"), str):
            return value["answer"]
    except json.JSONDecodeError:
        pass
    match = re.search(r'"answer"\s*:\s*"((?:\\.|[^"\\])*)"', candidate)
    if not match:
        return None
    try:
        return json.loads('"' + match.group(1) + '"')
    except json.JSONDecodeError:
        return match.group(1)


def score_response(text: str, row: dict[str, Any]) -> dict[str, Any]:
    """Score one model response against the answer and exact evidence labels.

    Reward is ``0.10 * format_valid + 0.45 * answer_correct +
    0.45 * (answer_correct and evidence_exact)``. Whitespace and key order
    are accepted. Code fences, surrounding text, duplicate keys, extra keys,
    malformed types, duplicate/nonpositive citations, and row-out-of-range IDs
    fail the format component. Parsed answer correctness remains available as
    a diagnosis when citations are invalid or out of range.
    """
    parsed = _json_object(text)
    fact_count = len(row.get("metadata", {}).get("latent", {}).get("facts", []))
    format_valid = _format_valid(text, parsed, max_evidence_id=fact_count or None)
    answer_correct = (
        parsed is not None
        and set(parsed) == {"answer", "evidence"}
        and isinstance(parsed.get("answer"), str)
        and parsed["answer"] == row["answer"]
    )

    gold_evidence = row["evidence"]
    candidate_evidence = parsed.get("evidence") if parsed is not None else None
    evidence_items_valid = (
        isinstance(candidate_evidence, list)
        and all(type(item) is int and item > 0 for item in candidate_evidence)
        and len(candidate_evidence) == len(set(candidate_evidence))
    )
    predicted_evidence: list[int] | None = candidate_evidence if evidence_items_valid else None
    if predicted_evidence is None:
        evidence_exact = False
        evidence_f1 = 0.0
    else:
        gold_set = set(gold_evidence)
        predicted_set = set(predicted_evidence)
        evidence_exact = (
            len(predicted_evidence) == len(predicted_set) and predicted_set == gold_set
        )
        if not predicted_set and not gold_set:
            evidence_f1 = 1.0
        elif not predicted_set or not gold_set:
            evidence_f1 = 0.0
        else:
            overlap = len(predicted_set & gold_set)
            if not overlap:
                evidence_f1 = 0.0
            else:
                precision = overlap / len(predicted_set)
                recall = overlap / len(gold_set)
                evidence_f1 = 2 * precision * recall / (precision + recall)

    joint_correct = bool(answer_correct and evidence_exact)
    abstain_correct: bool | None
    if row["task_type"] == "unanswerable" and row["answer"] == "未知":
        abstain_correct = bool(answer_correct and evidence_exact and row["answer"] == "未知")
    else:
        abstain_correct = None
    reward = (
        0.10 * float(format_valid)
        + 0.45 * float(answer_correct)
        + 0.45 * float(answer_correct and evidence_exact)
    )
    return {
        "format_valid": bool(format_valid),
        "answer_correct": bool(answer_correct),
        "evidence_exact": bool(evidence_exact),
        "evidence_f1": float(evidence_f1),
        "joint_correct": joint_correct,
        "abstain_correct": abstain_correct,
        "reward": float(reward),
    }
