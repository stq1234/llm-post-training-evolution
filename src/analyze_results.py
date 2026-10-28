"""Summarize post-training runs and draw reproducible result figures.

This module only reads evaluation/training artifacts. It never loads a model or
starts training/evaluation. Run it after all requested seeds and splits finish:

    python -m src.analyze_results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.data import TASK_TYPES
from src.rewards import _json_object, lenient_answer, score_response


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "results"
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "analysis"
DEFAULT_FIGURES_DIR = ROOT / "figures"
DEFAULT_REPORT = ROOT / "research" / "experiment_results.md"

METHODS = ("baseline", "sft", "sft_long", "dpo", "online_dpo", "grpo", "grpo_format")
TRAIN_METHODS = tuple(method for method in METHODS if method != "baseline")
FIXED_METHOD_SEEDS = {"sft_long": (17,)}
SEEDS = (17, 29)
EVAL_SPLITS = ("validation", "test_id", "test_ood", "test_counterfactual")
TEST_SPLITS = ("test_id", "test_ood", "test_counterfactual")
SPLIT_LABELS = {
    "validation": "验证集",
    "test_id": "同分布测试",
    "test_ood": "词汇/措辞留出测试",
    "test_counterfactual": "反事实成对测试",
}
METHOD_LABELS = {
    "baseline": "基线",
    "sft": "SFT",
    "sft_long": "SFT-256监督对照",
    "dpo": "DPO",
    "online_dpo": "在线DPO",
    "grpo": "GRPO",
    "grpo_format": "GRPO-格式奖励",
}
TASK_LABELS = {
    "lookup": "直接查找",
    "two_hop": "两步关系/加法",
    "unanswerable": "不可回答",
    "negation": "否定排除",
}
METRICS = (
    "format_valid",
    "answer_correct",
    "evidence_exact",
    "evidence_f1",
    "joint_correct",
    "abstain_correct",
)
DIAGNOSTIC_METRICS = (
    "lenient_answer_correct",
    "normalized_joint_correct",
)
RESOURCE_METRICS = (
    "wall_time_seconds",
    "generated_tokens",
    "updated_tokens",
    "peak_cuda_allocated_gib",
    "peak_cuda_reserved_gib",
    "training_updates",
    "warmstart_wall_time_seconds",
    "warmstart_generated_tokens",
    "warmstart_updated_tokens",
    "warmstart_training_updates",
    "end_to_end_wall_time_seconds",
    "end_to_end_generated_tokens",
    "end_to_end_updated_tokens",
    "end_to_end_training_updates",
    "end_to_end_peak_cuda_allocated_gib",
    "end_to_end_peak_cuda_reserved_gib",
    "rollout_prompts",
    "effective_groups",
    "tied_groups",
    "effective_group_rate",
    "preference_pairs",
)
OUTCOMES = ("joint_correct", "evidence_error", "answer_error", "format_invalid")
OUTCOME_LABELS = {
    "joint_correct": "联合正确",
    "evidence_error": "答案正确但证据错误",
    "answer_error": "答案错误",
    "format_invalid": "格式无效",
}
SPLIT_ORDER = {name: index for index, name in enumerate(EVAL_SPLITS)}
METHOD_ORDER = {name: index for index, name in enumerate(METHODS)}
ERROR_CATEGORIES = (
    "unanswerable_hallucination",
    "format_invalid",
    "answer_incorrect",
    "evidence_inexact",
)
ERROR_CATEGORY_LABELS = {
    "unanswerable_hallucination": "不可回答题仍给出答案",
    "format_invalid": "JSON/schema格式无效",
    "answer_incorrect": "答案错误",
    "evidence_inexact": "答案正确但证据不精确",
}
ERROR_CATEGORY_ORDER = {name: index for index, name in enumerate(ERROR_CATEGORIES)}


def _resolve_path(path: str | Path, base: Path = ROOT) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_data_index(data_dir: Path, splits: Iterable[str]) -> dict[str, dict[str, dict[str, Any]]]:
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for split in splits:
        path = data_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"canonical split is required for metric recomputation: {path}")
        rows = _read_jsonl(path)
        split_index: dict[str, dict[str, Any]] = {}
        for row in rows:
            row_id = row.get("id")
            if not isinstance(row_id, str):
                raise ValueError(f"row without string id in {path}")
            if row_id in split_index:
                raise ValueError(f"duplicate id {row_id!r} in {path}")
            if row.get("split") != split:
                raise ValueError(f"row {row_id} says split={row.get('split')!r}, expected {split!r}")
            split_index[row_id] = row
        index[split] = split_index
    return index


def _prediction_path(results_root: Path, method: str, seed: int | None, split: str) -> Path:
    if method == "baseline":
        return results_root / "eval" / "base" / split / "predictions.jsonl"
    return results_root / "eval" / f"{method}_seed{seed}" / split / "predictions.jsonl"


def _seeds_for_method(method: str, seeds: Iterable[int]) -> tuple[int | None, ...]:
    if method == "baseline":
        return (None,)
    if method in FIXED_METHOD_SEEDS:
        return FIXED_METHOD_SEEDS[method]
    return tuple(seeds)


def _strip_outer_json_fence(text: str) -> str:
    """Remove only a single outer ```json ... ``` fence for diagnosis."""
    candidate = text.strip()
    if len(candidate) < 10 or candidate[:7].casefold() != "```json" or not candidate.endswith("```"):
        return text
    return candidate[7:-3].strip()


def _load_evaluations(
    results_root: Path,
    data_index: dict[str, dict[str, dict[str, Any]]],
    methods: Iterable[str],
    seeds: Iterable[int],
    splits: Iterable[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    examples: list[dict[str, Any]] = []
    missing: list[str] = []
    for method in methods:
        current_seeds = _seeds_for_method(method, seeds)
        for seed in current_seeds:
            for split in splits:
                path = _prediction_path(results_root, method, seed, split)
                if not path.is_file():
                    missing.append(str(path.relative_to(results_root)) if path.is_relative_to(results_root) else str(path))
                    continue
                seen_ids: set[str] = set()
                for raw in _read_jsonl(path):
                    row_id = raw.get("id")
                    if not isinstance(row_id, str):
                        raise ValueError(f"prediction without string id in {path}")
                    if row_id in seen_ids:
                        raise ValueError(f"duplicate prediction id {row_id!r} in {path}")
                    seen_ids.add(row_id)
                    row = data_index.get(split, {}).get(row_id)
                    if row is None:
                        raise ValueError(f"prediction {row_id!r} in {path} has no canonical source row")
                    prediction = raw.get("prediction")
                    if not isinstance(prediction, str):
                        raise ValueError(f"prediction {row_id!r} in {path} lacks a string prediction")
                    score = score_response(prediction, row)
                    normalized_prediction = _strip_outer_json_fence(prediction)
                    normalized_score = score_response(normalized_prediction, row)
                    diagnostics = {
                        "lenient_answer_correct": lenient_answer(prediction) == row["answer"],
                        "normalized_joint_correct": normalized_score["joint_correct"],
                    }
                    examples.append({
                        "method": method,
                        "seed": seed,
                        "split": split,
                        "id": row_id,
                        "task_type": row["task_type"],
                        "pair_id": row.get("pair_id"),
                        "prompt": row["prompt"],
                        "answer": row["answer"],
                        "evidence": row["evidence"],
                        "prediction": prediction,
                        "score": score,
                        "diagnostics": diagnostics,
                        "generated_tokens": raw.get("generated_tokens"),
                        "truncated": raw.get("truncated"),
                    })
    if not examples:
        expected = _prediction_path(results_root, "sft", next(iter(seeds), 17), "test_id")
        raise FileNotFoundError(
            "no prediction files found; run training and evaluation first. "
            f"Expected paths such as {expected}"
        )
    return examples, missing


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := _numeric(value)) is not None]
    return statistics.fmean(numbers) if numbers else None


def _metric_row(
    method: str,
    seed: int | None,
    split: str,
    task_type: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    scores = [row["score"] for row in rows]
    truncated = [row["truncated"] for row in rows if isinstance(row.get("truncated"), bool)]
    generated = [row["generated_tokens"] for row in rows if _numeric(row.get("generated_tokens")) is not None]
    result: dict[str, Any] = {
        "method": method,
        "seed": seed,
        "split": split,
        "task_type": task_type,
        "n": len(rows),
        "truncation_observations": len(truncated),
        "truncation_rate": _mean(truncated),
        "mean_generated_tokens": _mean(generated),
        "generated_tokens_total": sum(int(value) for value in generated) if generated else None,
    }
    for metric in METRICS:
        result[metric] = _mean(score.get(metric) for score in scores)
    for metric in DIAGNOSTIC_METRICS:
        result[metric] = _mean(row["diagnostics"].get(metric) for row in rows)
    result["abstain_examples"] = sum(score.get("abstain_correct") is not None for score in scores)
    return result


def _metrics_by_seed(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bucket: dict[tuple[str, int | None, str, str], list[dict[str, Any]]] = defaultdict(list)
    overall: dict[tuple[str, int | None, str], list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        bucket[(example["method"], example["seed"], example["split"], example["task_type"])].append(example)
        overall[(example["method"], example["seed"], example["split"])].append(example)
    rows = [
        _metric_row(method, seed, split, task_type, members)
        for (method, seed, split, task_type), members in bucket.items()
    ]
    rows.extend(
        _metric_row(method, seed, split, "overall", members)
        for (method, seed, split), members in overall.items()
    )
    return sorted(rows, key=_metric_sort_key)


def _evaluation_coverage(
    results_root: Path,
    data_index: dict[str, dict[str, dict[str, Any]]],
    examples: list[dict[str, Any]],
    methods: Iterable[str],
    seeds: Iterable[int],
    splits: Iterable[str],
) -> list[dict[str, Any]]:
    observed: dict[tuple[str, int | None, str], int] = defaultdict(int)
    for example in examples:
        observed[(example["method"], example["seed"], example["split"])] += 1
    coverage: list[dict[str, Any]] = []
    for method in methods:
        current_seeds = _seeds_for_method(method, seeds)
        for seed in current_seeds:
            for split in splits:
                path = _prediction_path(results_root, method, seed, split)
                expected = len(data_index.get(split, {}))
                actual = observed.get((method, seed, split), 0)
                coverage.append({
                    "method": method,
                    "seed": seed,
                    "split": split,
                    "prediction_file": str(path.relative_to(results_root)) if path.is_relative_to(results_root) else str(path),
                    "prediction_file_exists": path.is_file(),
                    "expected_examples": expected,
                    "observed_examples": actual,
                    "coverage_rate": actual / expected if expected else None,
                    "complete": actual == expected,
                })
    return coverage


def _metric_sort_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    method_order = METHOD_ORDER.get(row["method"], len(METHOD_ORDER))
    seed = row["seed"]
    seed_order = -1 if seed is None else int(seed)
    split_order = SPLIT_ORDER.get(row["split"], len(SPLIT_ORDER))
    task_order = -1 if row["task_type"] == "overall" else TASK_TYPES.index(row["task_type"])
    return method_order, seed_order, split_order, task_order


def _warmstart_candidate_selection(
    results_root: Path,
    validation_index: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    candidates = (
        ("sft_seed17", 16),
        ("pilot_sft32_seed17", 32),
    )
    rows: list[dict[str, Any]] = []
    for run_name, fallback_steps in candidates:
        config_path = results_root / run_name / "run_config.json"
        config = _read_json(config_path) if config_path.is_file() else {}
        run_summary_path = results_root / run_name / "run_summary.json"
        run_summary = _read_json(run_summary_path) if run_summary_path.is_file() else {}
        settings = config.get("settings", {})
        steps = int(settings.get("steps", fallback_steps)) if isinstance(settings, dict) else fallback_steps
        prediction_path = results_root / "eval" / run_name / "validation" / "predictions.jsonl"
        scores: list[dict[str, Any]] = []
        if prediction_path.is_file():
            seen: set[str] = set()
            for prediction_row in _read_jsonl(prediction_path):
                row_id = prediction_row.get("id")
                if not isinstance(row_id, str) or row_id in seen:
                    raise ValueError(f"missing or duplicate candidate prediction id in {prediction_path}")
                seen.add(row_id)
                canonical = validation_index.get(row_id)
                if canonical is None:
                    raise ValueError(f"candidate prediction {row_id!r} has no canonical validation row")
                prediction = prediction_row.get("prediction")
                if not isinstance(prediction, str):
                    raise ValueError(f"candidate prediction {row_id!r} lacks string prediction")
                scores.append(score_response(prediction, canonical))
        format_valid = _mean(score["format_valid"] for score in scores)
        joint_correct = _mean(score["joint_correct"] for score in scores)
        eligible = bool(
            format_valid is not None and joint_correct is not None
            and format_valid >= 0.95 and joint_correct < 0.95
        )
        rows.append({
            "candidate": f"SFT-{steps}",
            "run": run_name,
            "seed": 17,
            "steps": steps,
            "n": len(scores),
            "prediction_file": str(prediction_path.relative_to(results_root)) if prediction_path.is_relative_to(results_root) else str(prediction_path),
            "format_valid": format_valid,
            "answer_correct": _mean(score["answer_correct"] for score in scores),
            "joint_correct": joint_correct,
            "format_pass_95": format_valid >= 0.95 if format_valid is not None else None,
            "joint_below_95": joint_correct < 0.95 if joint_correct is not None else None,
            "eligible": eligible,
            "selected": False,
            "wall_time_seconds": _numeric(run_summary.get("wall_time_seconds")),
            "generated_tokens": _numeric(run_summary.get("generated_tokens")),
            "updated_tokens": _numeric(run_summary.get("updated_tokens")),
            "peak_cuda_allocated_gib": (
                _numeric(run_summary.get("peak_cuda_allocated_bytes")) / (1024 ** 3)
                if _numeric(run_summary.get("peak_cuda_allocated_bytes")) is not None else None
            ),
            "peak_cuda_reserved_gib": (
                _numeric(run_summary.get("peak_cuda_reserved_bytes")) / (1024 ** 3)
                if _numeric(run_summary.get("peak_cuda_reserved_bytes")) is not None else None
            ),
        })
    eligible_rows = [row for row in rows if row["eligible"]]
    selected = min(eligible_rows, key=lambda row: row["steps"]) if eligible_rows else None
    if selected is not None:
        selected["selected"] = True
    return rows, selected["run"] if selected is not None else None


def _matched_method_differences(
    metrics_by_seed: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparisons = (
        ("dpo_minus_sft", "dpo", "sft"),
        ("grpo_full_minus_format", "grpo", "grpo_format"),
    )
    metrics = (
        "format_valid", "answer_correct", "evidence_exact", "joint_correct",
        *DIAGNOSTIC_METRICS,
    )
    lookup = {
        (row["method"], row["seed"], row["split"], row["task_type"]): row
        for row in metrics_by_seed
    }
    rows: list[dict[str, Any]] = []
    for comparison, method, reference in comparisons:
        for seed in sorted({row["seed"] for row in metrics_by_seed if row["method"] in (method, reference)}):
            for split in EVAL_SPLITS:
                task_types = ("overall", *TASK_TYPES)
                for task_type in task_types:
                    lhs = lookup.get((method, seed, split, task_type))
                    rhs = lookup.get((reference, seed, split, task_type))
                    if lhs is None or rhs is None:
                        continue
                    for metric in metrics:
                        left_value, right_value = _numeric(lhs.get(metric)), _numeric(rhs.get(metric))
                        if left_value is None or right_value is None:
                            continue
                        rows.append({
                            "comparison": comparison,
                            "method": method,
                            "reference": reference,
                            "seed": seed,
                            "split": split,
                            "task_type": task_type,
                            "metric": metric,
                            "difference": left_value - right_value,
                        })
    aggregate = _aggregate_seed_rows(
        rows,
        ("comparison", "split", "task_type", "metric"),
        ("difference",),
    )
    return rows, aggregate


def _aggregate_seed_rows(
    rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
    value_fields: Iterable[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in key_fields)].append(row)
    aggregate: list[dict[str, Any]] = []
    for key, values in grouped.items():
        seed_values = {row.get("seed") for row in values}
        item: dict[str, Any] = dict(zip(key_fields, key))
        item["n_seeds"] = len(seed_values)
        if "n" in values[0]:
            item["examples_per_seed"] = _mean(row.get("n") for row in values)
        for field in value_fields:
            numbers = [number for row in values if (number := _numeric(row.get(field))) is not None]
            item[f"{field}_mean"] = statistics.fmean(numbers) if numbers else None
            item[f"{field}_min"] = min(numbers) if numbers else None
            item[f"{field}_max"] = max(numbers) if numbers else None
        aggregate.append(item)
    return aggregate


def _outcome(score: dict[str, Any]) -> str:
    if not score.get("format_valid", False):
        return "format_invalid"
    if not score.get("answer_correct", False):
        return "answer_error"
    if not score.get("evidence_exact", False):
        return "evidence_error"
    return "joint_correct"


def _failure_rows(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int | None, str], list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[(example["method"], example["seed"], example["split"])].append(example)
    result = []
    for (method, seed, split), values in grouped.items():
        counts = {outcome: 0 for outcome in OUTCOMES}
        for value in values:
            counts[_outcome(value["score"])] += 1
        result.append({
            "method": method,
            "seed": seed,
            "split": split,
            "n": len(values),
            **{f"{key}_rate": count / len(values) for key, count in counts.items()},
        })
    return sorted(
        result,
        key=lambda row: (
            METHOD_ORDER.get(row["method"], 99),
            -1 if row["seed"] is None else row["seed"],
            SPLIT_ORDER.get(row["split"], 99),
        ),
    )


def _counterfactual_summary(
    examples: list[dict[str, Any]],
    data_index: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    expected: dict[str, list[str]] = defaultdict(list)
    for row_id, row in data_index.get("test_counterfactual", {}).items():
        pair_id = row.get("pair_id")
        if pair_id is not None:
            expected[pair_id].append(row_id)
    grouped: dict[tuple[str, int | None, str], list[dict[str, Any]]] = defaultdict(list)
    present_runs: set[tuple[str, int | None]] = set()
    for example in examples:
        if example["split"] != "test_counterfactual":
            continue
        pair_id = example.get("pair_id")
        if pair_id is not None:
            key = (example["method"], example["seed"], pair_id)
            grouped[key].append(example)
            present_runs.add((example["method"], example["seed"]))

    details: list[dict[str, Any]] = []
    by_run: list[dict[str, Any]] = []
    for method, seed in sorted(present_runs, key=lambda key: (METHOD_ORDER.get(key[0], 99), -1 if key[1] is None else key[1])):
        pair_successes: list[float] = []
        complete_count = 0
        for pair_id, expected_ids in sorted(expected.items()):
            pair_rows = sorted(grouped.get((method, seed, pair_id), []), key=lambda row: row["id"])
            complete = len(pair_rows) == 2 and set(row["id"] for row in pair_rows) == set(expected_ids)
            success = (
                float(all(row["score"]["joint_correct"] for row in pair_rows))
                if complete
                else None
            )
            if complete:
                pair_successes.append(success)
                complete_count += 1
            details.append({
                "method": method,
                "seed": seed,
                "pair_id": pair_id,
                "sample_ids": ";".join(expected_ids),
                "n_predictions": len(pair_rows),
                "complete": complete,
                "first_joint_correct": pair_rows[0]["score"]["joint_correct"] if complete else None,
                "second_joint_correct": pair_rows[1]["score"]["joint_correct"] if complete else None,
                "pair_joint_success": success,
            })
        by_run.append({
            "method": method,
            "seed": seed,
            "expected_pairs": len(expected),
            "complete_pairs": complete_count,
            "incomplete_pairs": len(expected) - complete_count,
            "paired_double_joint_success": _mean(pair_successes),
        })
    aggregate = _aggregate_seed_rows(
        by_run,
        ("method",),
        ("paired_double_joint_success", "complete_pairs", "incomplete_pairs"),
    )
    return details, by_run, aggregate


def _strict_scoring_answer(text: str) -> str | None:
    """Extract the answer under score_response's JSON keys/type policy.

    Citation validity is deliberately not required here: this diagnostic isolates
    answer behavior and uses the same parsed-object, exact-key, string-answer
    conditions as score_response(answer_correct).
    """
    parsed = _json_object(text)
    if parsed is None or set(parsed) != {"answer", "evidence"}:
        return None
    answer = parsed.get("answer")
    return answer if isinstance(answer, str) else None


def _counterfactual_answer_change_summary(
    examples: list[dict[str, Any]],
    data_index: dict[str, dict[str, dict[str, Any]]],
    methods: Iterable[str],
    seeds: Iterable[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    gold_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in data_index.get("test_counterfactual", {}).values():
        pair_id = row.get("pair_id")
        if pair_id is not None:
            gold_groups[pair_id].append(row)
    expected: dict[str, list[dict[str, Any]]] = {}
    for pair_id, rows in gold_groups.items():
        if len(rows) != 2:
            raise ValueError(f"counterfactual pair {pair_id!r} has {len(rows)} canonical rows; expected two")
        expected[pair_id] = sorted(rows, key=lambda row: row["id"])

    prediction_index = {
        (example["method"], example["seed"], example["id"]): example
        for example in examples
        if example["split"] == "test_counterfactual"
    }
    pair_details: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for method in methods:
        for seed in _seeds_for_method(method, seeds):
            changed_count = sum(rows[0]["answer"] != rows[1]["answer"] for rows in expected.values())
            unchanged_count = len(expected) - changed_count
            changed_behavior_successes = 0
            correct_change_successes = 0
            changed_parse_available = 0
            unchanged_consistency_successes = 0
            unchanged_parse_available = 0
            complete_count = 0

            for pair_id, canonical_rows in sorted(expected.items()):
                predictions = [
                    prediction_index.get((method, seed, canonical_row["id"]))
                    for canonical_row in canonical_rows
                ]
                complete = all(prediction is not None for prediction in predictions)
                if complete:
                    complete_count += 1
                strict_answers = [
                    _strict_scoring_answer(prediction["prediction"]) if prediction is not None else None
                    for prediction in predictions
                ]
                both_strict_answers_available = all(answer is not None for answer in strict_answers)
                gold_changed = canonical_rows[0]["answer"] != canonical_rows[1]["answer"]
                prediction_changed: bool | None = None
                correct_answer_change: bool | None = None
                unchanged_prediction_consistent: bool | None = None

                if gold_changed:
                    if both_strict_answers_available:
                        changed_parse_available += 1
                        prediction_changed = strict_answers[0] != strict_answers[1]
                        correct_answer_change = (
                            strict_answers[0] == canonical_rows[0]["answer"]
                            and strict_answers[1] == canonical_rows[1]["answer"]
                        )
                        changed_behavior_successes += int(prediction_changed)
                        correct_change_successes += int(correct_answer_change)
                else:
                    if both_strict_answers_available:
                        unchanged_parse_available += 1
                        unchanged_prediction_consistent = strict_answers[0] == strict_answers[1]
                        unchanged_consistency_successes += int(unchanged_prediction_consistent)
                    else:
                        # Parse failures stay in the unchanged-pair denominator as failures.
                        unchanged_prediction_consistent = False

                pair_details.append({
                    "method": method,
                    "seed": seed,
                    "pair_id": pair_id,
                    "sample_ids": ";".join(row["id"] for row in canonical_rows),
                    "gold_answer_changed": gold_changed,
                    "gold_first_answer": canonical_rows[0]["answer"],
                    "gold_second_answer": canonical_rows[1]["answer"],
                    "n_predictions": sum(prediction is not None for prediction in predictions),
                    "complete": complete,
                    "first_strict_prediction_answer": strict_answers[0],
                    "second_strict_prediction_answer": strict_answers[1],
                    "both_strict_answers_available": both_strict_answers_available,
                    "prediction_answer_changed": prediction_changed,
                    "correct_answer_change": correct_answer_change,
                    "unchanged_prediction_consistent": unchanged_prediction_consistent,
                })

            changed_behavior_rate = (
                changed_behavior_successes / changed_count if changed_count else None
            )
            correct_answer_change_rate = (
                correct_change_successes / changed_count if changed_count else None
            )
            unchanged_consistency_rate = (
                unchanged_consistency_successes / unchanged_count if unchanged_count else None
            )
            run_summaries.append({
                "method": method,
                "seed": seed,
                "expected_pairs": len(expected),
                "gold_changed_pairs": changed_count,
                "gold_unchanged_pairs": unchanged_count,
                "complete_pairs": complete_count,
                "changed_strict_answer_available_pairs": changed_parse_available,
                "changed_strict_answer_unavailable_pairs": changed_count - changed_parse_available,
                "changed_behavior_successes": changed_behavior_successes,
                "changed_behavior_denominator": changed_count,
                "changed_behavior_rate": changed_behavior_rate,
                "correct_answer_change_successes": correct_change_successes,
                "correct_answer_change_denominator": changed_count,
                "correct_answer_change_rate": correct_answer_change_rate,
                "unchanged_strict_answer_available_pairs": unchanged_parse_available,
                "unchanged_strict_answer_unavailable_pairs": unchanged_count - unchanged_parse_available,
                "unchanged_consistency_successes": unchanged_consistency_successes,
                "unchanged_consistency_denominator": unchanged_count,
                "unchanged_answer_consistency_rate": unchanged_consistency_rate,
            })

    run_summaries.sort(
        key=lambda row: (METHOD_ORDER.get(row["method"], 99), -1 if row["seed"] is None else int(row["seed"]))
    )
    pair_details.sort(
        key=lambda row: (METHOD_ORDER.get(row["method"], 99), -1 if row["seed"] is None else int(row["seed"]), row["pair_id"])
    )
    aggregate = _aggregate_seed_rows(
        run_summaries,
        ("method",),
        (
            "expected_pairs", "gold_changed_pairs", "gold_unchanged_pairs", "complete_pairs",
            "changed_strict_answer_available_pairs", "changed_strict_answer_unavailable_pairs",
            "changed_behavior_successes", "changed_behavior_denominator", "changed_behavior_rate",
            "correct_answer_change_successes", "correct_answer_change_denominator",
            "correct_answer_change_rate", "unchanged_strict_answer_available_pairs",
            "unchanged_strict_answer_unavailable_pairs", "unchanged_consistency_successes",
            "unchanged_consistency_denominator", "unchanged_answer_consistency_rate",
        ),
    )
    return pair_details, run_summaries, aggregate


def _read_training_resources(
    results_root: Path, methods: Iterable[str], seeds: Iterable[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    resources: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    missing_summaries: list[str] = []
    for method in methods:
        if method == "baseline":
            continue
        for seed in _seeds_for_method(method, seeds):
            run_dir = results_root / f"{method}_seed{seed}"
            summary_path = run_dir / "run_summary.json"
            if not summary_path.is_file():
                missing_summaries.append(str(summary_path.relative_to(results_root)))
                continue
            summary = _read_json(summary_path)
            config_path = run_dir / "run_config.json"
            config = _read_json(config_path) if config_path.is_file() else {}
            settings = config.get("settings", {})
            beta = _numeric(settings.get("beta")) if isinstance(settings, dict) else None
            log_path = run_dir / "step_log.jsonl"
            log_rows = _read_jsonl(log_path) if log_path.is_file() else []
            for record in log_rows:
                point = {"method": method, "seed": seed, **record}
                if method == "dpo" and beta is not None and beta > 0:
                    chosen_reward = _numeric(record.get("chosen_reward"))
                    rejected_reward = _numeric(record.get("rejected_reward"))
                    if chosen_reward is not None and rejected_reward is not None:
                        chosen_delta = chosen_reward / beta
                        rejected_delta = rejected_reward / beta
                        point.update({
                            "beta": beta,
                            "chosen_policy_ref_logp_delta": chosen_delta,
                            "rejected_policy_ref_logp_delta": rejected_delta,
                            "implicit_margin_recomputed": chosen_delta - rejected_delta,
                            "implicit_margin_check_error": (
                                chosen_delta - rejected_delta - _numeric(record.get("preference_margin"))
                                if _numeric(record.get("preference_margin")) is not None else None
                            ),
                        })
                curves.append(point)

            rollout_path = run_dir / "rollouts.jsonl"
            rollout_rows = _read_jsonl(rollout_path) if rollout_path.is_file() else []
            effective = tied = 0
            observed_rollouts = 0
            for group in rollout_rows:
                candidates = group.get("candidates", [])
                rewards = [value for candidate in candidates if (value := _numeric(candidate.get("reward"))) is not None]
                if group.get("decision") == "preference_pair":
                    effective += 1
                    observed_rollouts += 1
                elif group.get("decision") == "skip_tied_rewards":
                    tied += 1
                    observed_rollouts += 1
                elif rewards:
                    observed_rollouts += 1
                    if len(set(rewards)) > 1:
                        effective += 1
                    else:
                        tied += 1

            allocated_bytes = _numeric(summary.get("peak_cuda_allocated_bytes"))
            reserved_bytes = _numeric(summary.get("peak_cuda_reserved_bytes"))
            run_config_method = config.get("method")
            stage_wall = _numeric(summary.get("wall_time_seconds"))
            stage_generated = _numeric(summary.get("generated_tokens"))
            stage_updated = _numeric(summary.get("updated_tokens"))
            stage_updates = _numeric(summary.get("training_updates", summary.get("updates")))
            stage_allocated_gib = allocated_bytes / (1024 ** 3) if allocated_bytes is not None else None
            stage_reserved_gib = reserved_bytes / (1024 ** 3) if reserved_bytes is not None else None

            adapter_input = config.get("adapter_input")
            if not adapter_input and isinstance(settings, dict):
                adapter_input = settings.get("adapter")
            warmstart_run = None
            warmstart_summary: dict[str, Any] | None = None
            warmstart_available = not bool(adapter_input)
            if adapter_input:
                adapter_path = Path(str(adapter_input))
                warmstart_run = adapter_path.parent.name
                warmstart_summary_path = results_root / warmstart_run / "run_summary.json"
                if warmstart_run and warmstart_summary_path.is_file():
                    warmstart_summary = _read_json(warmstart_summary_path)
                    warmstart_available = True
                else:
                    warmstart_available = False
                    missing_summaries.append(
                        f"warmstart:{warmstart_run or adapter_input}/run_summary.json"
                    )
            warmstart_allocated_bytes = (
                _numeric(warmstart_summary.get("peak_cuda_allocated_bytes"))
                if warmstart_summary is not None else None
            )
            warmstart_reserved_bytes = (
                _numeric(warmstart_summary.get("peak_cuda_reserved_bytes"))
                if warmstart_summary is not None else None
            )
            warmstart_wall = (
                _numeric(warmstart_summary.get("wall_time_seconds"))
                if warmstart_summary is not None else 0.0 if warmstart_available else None
            )
            warmstart_generated = (
                _numeric(warmstart_summary.get("generated_tokens"))
                if warmstart_summary is not None else 0.0 if warmstart_available else None
            )
            warmstart_updated = (
                _numeric(warmstart_summary.get("updated_tokens"))
                if warmstart_summary is not None else 0.0 if warmstart_available else None
            )
            warmstart_updates = (
                _numeric(warmstart_summary.get("training_updates", warmstart_summary.get("updates")))
                if warmstart_summary is not None else 0.0 if warmstart_available else None
            )
            warmstart_allocated_gib = (
                warmstart_allocated_bytes / (1024 ** 3)
                if warmstart_allocated_bytes is not None else 0.0 if warmstart_available else None
            )
            warmstart_reserved_gib = (
                warmstart_reserved_bytes / (1024 ** 3)
                if warmstart_reserved_bytes is not None else 0.0 if warmstart_available else None
            )

            def end_to_end_sum(stage_value: float | None, warmstart_value: float | None) -> float | None:
                if stage_value is None or warmstart_value is None:
                    return None
                return stage_value + warmstart_value

            def end_to_end_peak(stage_value: float | None, warmstart_value: float | None) -> float | None:
                if stage_value is None or warmstart_value is None:
                    return None
                return max(stage_value, warmstart_value)

            resource = {
                "method": method,
                "seed": seed,
                "run_config_method": run_config_method,
                "wall_time_seconds": stage_wall,
                "generated_tokens": stage_generated,
                "updated_tokens": stage_updated,
                "peak_cuda_allocated_gib": stage_allocated_gib,
                "peak_cuda_reserved_gib": stage_reserved_gib,
                "training_updates": stage_updates,
                "warmstart_run": warmstart_run,
                "warmstart_available": warmstart_available,
                "warmstart_wall_time_seconds": warmstart_wall,
                "warmstart_generated_tokens": warmstart_generated,
                "warmstart_updated_tokens": warmstart_updated,
                "warmstart_training_updates": warmstart_updates,
                "end_to_end_wall_time_seconds": end_to_end_sum(stage_wall, warmstart_wall),
                "end_to_end_generated_tokens": end_to_end_sum(stage_generated, warmstart_generated),
                "end_to_end_updated_tokens": end_to_end_sum(stage_updated, warmstart_updated),
                "end_to_end_training_updates": end_to_end_sum(stage_updates, warmstart_updates),
                "end_to_end_peak_cuda_allocated_gib": end_to_end_peak(stage_allocated_gib, warmstart_allocated_gib if warmstart_available else None),
                "end_to_end_peak_cuda_reserved_gib": end_to_end_peak(stage_reserved_gib, warmstart_reserved_gib if warmstart_available else None),
                "rollout_prompts": _numeric(summary.get("rollout_prompts")),
                "effective_groups": effective if observed_rollouts else None,
                "tied_groups": tied if observed_rollouts else None,
                "effective_group_rate": effective / observed_rollouts if observed_rollouts else None,
                "rollout_groups_logged": observed_rollouts,
                "summary_skipped_ties": _numeric(summary.get("skipped_ties")),
                "preference_pairs": _numeric(config.get("dataset_rows")) if method == "dpo" else None,
                "step_log_records": len(log_rows),
                "failed_gradient_steps": sum(record.get("gradient_finite") is False for record in log_rows),
            }
            resources.append(resource)

    resources.sort(key=lambda row: (METHOD_ORDER.get(row["method"], 99), int(row["seed"])))
    return resources, curves, missing_summaries


def _resource_seed_aggregate(resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _aggregate_seed_rows(resources, ("method",), RESOURCE_METRICS)


def _dpo_logp_summaries(
    curves: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for point in curves:
        if point.get("method") == "dpo":
            by_seed[int(point["seed"])].append(point)
    summaries: list[dict[str, Any]] = []
    for seed, points in sorted(by_seed.items()):
        ordered = sorted(points, key=lambda point: int(point["step"]))
        first, last = ordered[0], ordered[-1]
        summaries.append({
            "method": "dpo",
            "seed": seed,
            "steps": len(ordered),
            "beta": _numeric(first.get("beta")),
            "start_step": first["step"],
            "end_step": last["step"],
            "start_chosen_policy_ref_logp_delta": _numeric(first.get("chosen_policy_ref_logp_delta")),
            "end_chosen_policy_ref_logp_delta": _numeric(last.get("chosen_policy_ref_logp_delta")),
            "start_rejected_policy_ref_logp_delta": _numeric(first.get("rejected_policy_ref_logp_delta")),
            "end_rejected_policy_ref_logp_delta": _numeric(last.get("rejected_policy_ref_logp_delta")),
            "start_implicit_margin": _numeric(first.get("preference_margin")),
            "end_implicit_margin": _numeric(last.get("preference_margin")),
            "mean_implicit_margin": _mean(point.get("preference_margin") for point in ordered),
            "max_abs_margin_check_error": max(
                (abs(float(point["implicit_margin_check_error"]))
                 for point in ordered if _numeric(point.get("implicit_margin_check_error")) is not None),
                default=None,
            ),
        })
    aggregate = _aggregate_seed_rows(
        summaries,
        ("method",),
        (
            "steps", "beta", "start_chosen_policy_ref_logp_delta",
            "end_chosen_policy_ref_logp_delta", "start_rejected_policy_ref_logp_delta",
            "end_rejected_policy_ref_logp_delta", "start_implicit_margin",
            "end_implicit_margin", "mean_implicit_margin", "max_abs_margin_check_error",
        ),
    )
    return summaries, aggregate


def _draw_dpo_logp_figure(figures_dir: Path, curves: list[dict[str, Any]]) -> list[str]:
    points = [
        point for point in curves
        if point.get("method") == "dpo"
        and _numeric(point.get("chosen_policy_ref_logp_delta")) is not None
        and _numeric(point.get("rejected_policy_ref_logp_delta")) is not None
        and _numeric(point.get("preference_margin")) is not None
    ]
    if not points:
        return []
    _, plt = _matplotlib_setup()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    color_by_seed = {17: "#4477AA", 29: "#CC6677"}
    panels = (
        ("chosen_policy_ref_logp_delta", "Chosen 相对参考模型 log-prob 差"),
        ("rejected_policy_ref_logp_delta", "Rejected 相对参考模型 log-prob 差"),
        ("preference_margin", "DPO 隐式 preference margin"),
    )
    for axis, (field, title) in zip(axes, panels):
        for seed in sorted({int(point["seed"]) for point in points}):
            seed_points = sorted(
                (point for point in points if int(point["seed"]) == seed),
                key=lambda point: int(point["step"]),
            )
            axis.plot(
                [int(point["step"]) for point in seed_points],
                [float(point[field]) for point in seed_points],
                label=f"seed {seed}", color=color_by_seed.get(seed, None), linewidth=1.4,
            )
        axis.set_title(title)
        axis.set_xlabel("更新步")
        axis.grid(alpha=0.25)
        if field == "preference_margin":
            axis.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("相对 log-prob 差（天然对数）")
    axes[1].set_ylabel("相对 log-prob 差（天然对数）")
    axes[2].set_ylabel("log-prob 比率差")
    axes[0].legend(frameon=False)
    fig.suptitle("DPO 训练日志中的 chosen/rejected 相对 log-prob 与隐式偏好间隔")
    fig.tight_layout()
    base = figures_dir / "dpo_logp_margin"
    _save_figure(fig, base)
    plt.close(fig)
    return [str(base.with_suffix(".png")), str(base.with_suffix(".pdf"))]


def _error_category(example: dict[str, Any]) -> str | None:
    answer = lenient_answer(example["prediction"])
    if example["answer"] == "未知" and answer not in (None, "未知"):
        return "unanswerable_hallucination"
    score = example["score"]
    if not score["format_valid"]:
        return "format_invalid"
    if not score["answer_correct"]:
        return "answer_incorrect"
    if not score["evidence_exact"]:
        return "evidence_inexact"
    return None


def _select_error_cases(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        if not example["split"].startswith("test_"):
            continue
        category = _error_category(example)
        if category is not None:
            candidates[(example["method"], category)].append(example)
    result: list[dict[str, Any]] = []
    for (method, category), rows in sorted(
        candidates.items(),
        key=lambda item: (
            METHOD_ORDER.get(item[0][0], 99),
            ERROR_CATEGORY_ORDER.get(item[0][1], 99),
        ),
    ):
        chosen = min(
            rows,
            key=lambda row: (
                -1 if row["seed"] is None else row["seed"],
                SPLIT_ORDER.get(row["split"], 99),
                row["id"],
            ),
        )
        result.append({
            "method": method,
            "seed": chosen["seed"],
            "split": chosen["split"],
            "category": category,
            "category_label": ERROR_CATEGORY_LABELS[category],
            "id": chosen["id"],
            "task_type": chosen["task_type"],
            "pair_id": chosen["pair_id"],
            "prompt": chosen["prompt"],
            "gold_answer": chosen["answer"],
            "gold_evidence": chosen["evidence"],
            "prediction": chosen["prediction"],
            "score": chosen["score"],
            "selection_rule": "每方法、每错误类别至多1例；优先种子17，再29，随后test_id、test_ood、test_counterfactual及升序id。",
        })
    return result


def _failure_seed_aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _aggregate_seed_rows(
        rows,
        ("method", "split"),
        tuple(f"{outcome}_rate" for outcome in OUTCOMES),
    )


def _ordered_methods(rows: list[dict[str, Any]]) -> list[str]:
    present = {row["method"] for row in rows}
    return [method for method in METHODS if method in present]


def _fmt(value: Any, digits: int = 3) -> str:
    number = _numeric(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _fmt_seed_range(row: dict[str, Any], metric: str) -> str:
    mean = _numeric(row.get(f"{metric}_mean"))
    minimum = _numeric(row.get(f"{metric}_min"))
    maximum = _numeric(row.get(f"{metric}_max"))
    if mean is None:
        return "—"
    if int(row.get("n_seeds", 0)) < 2 or minimum is None or maximum is None:
        return f"{mean:.3f} (n={row.get('n_seeds', 0)})"
    return f"{mean:.3f} [{minimum:.3f}, {maximum:.3f}]"


def _method_display_label(method: str, n_seeds: int | None) -> str:
    label = METHOD_LABELS[method]
    return f"{label}（n=1）" if n_seeds == 1 else label


CF_ANSWER_CHANGE_START = "<!-- BEGIN CF ANSWER-CHANGE DIAGNOSTICS -->"
CF_ANSWER_CHANGE_END = "<!-- END CF ANSWER-CHANGE DIAGNOSTICS -->"


def _fmt_cf_count_per_seed(row: dict[str, Any], field: str) -> str:
    mean = _numeric(row.get(f"{field}_mean"))
    minimum = _numeric(row.get(f"{field}_min"))
    maximum = _numeric(row.get(f"{field}_max"))
    if mean is None:
        return "—"
    if int(row.get("n_seeds", 0)) < 2:
        return f"{mean:g} (n=1)"
    if minimum == maximum:
        return f"{mean:g} / seed (n={row['n_seeds']})"
    return f"{mean:g} [{minimum:g}, {maximum:g}] / seed (n={row['n_seeds']})"


def _fmt_cf_rate_with_denominator(
    row: dict[str, Any],
    rate_field: str,
    successes_field: str,
    denominator_field: str,
) -> str:
    denominator = _numeric(row.get(f"{denominator_field}_mean"))
    successes = _numeric(row.get(f"{successes_field}_mean"))
    if denominator is None or denominator == 0:
        return "— (0 pairs)"
    rate_text = _fmt_seed_range(row, rate_field)
    unit = "pairs per seed" if int(row.get("n_seeds", 0)) >= 2 else "pairs"
    return f"{rate_text} ({successes:g}/{denominator:g} {unit})"


def _counterfactual_answer_change_report_lines(aggregate: list[dict[str, Any]]) -> list[str]:
    lookup = {row["method"]: row for row in aggregate}
    lines = [
        CF_ANSWER_CHANGE_START,
        "",
        "### 反事实答案变化行为诊断",
        "",
        "先按 canonical pair_id 统计 gold 答案改变与不改变的 pair 数。严格答案抽取使用单个无外围文本的 JSON 对象、恰好包含 answer/evidence 两键且 answer 为字符串，与 score_response 的 answer_correct 解析前提一致；该答案行为指标不要求引用字段正确。各比例以该 gold 类别的全部 pair 为分母，格式错误、无法解析和缺失预测均计为失败。围栏输出不参与严格抽取；已有宽松答案诊断独立保留。",
        "答案发生变化率只衡量两侧均可严格抽取且预测不同，不要求预测正确；正确变化率要求两侧答案分别等于各自 gold，因此不等同于联合正确或双题 joint 成功。gold 不变的一致率衡量两侧可严格抽取且预测相同，不要求两侧答案正确。",
        "",
        "| 方法 | Gold 改变 pairs | 严格预测变化率 | 正确答案变化率 | Gold 不变 pairs | 预测一致率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = lookup.get(method)
        if row is None:
            continue
        lines.append(
            f"| {METHOD_LABELS[method]} | {_fmt_cf_count_per_seed(row, 'gold_changed_pairs')} | "
            f"{_fmt_cf_rate_with_denominator(row, 'changed_behavior_rate', 'changed_behavior_successes', 'changed_behavior_denominator')} | "
            f"{_fmt_cf_rate_with_denominator(row, 'correct_answer_change_rate', 'correct_answer_change_successes', 'correct_answer_change_denominator')} | "
            f"{_fmt_cf_count_per_seed(row, 'gold_unchanged_pairs')} | "
            f"{_fmt_cf_rate_with_denominator(row, 'unchanged_answer_consistency_rate', 'unchanged_consistency_successes', 'unchanged_consistency_denominator')} |"
        )
    if aggregate and all(_numeric(row.get("gold_unchanged_pairs_mean")) == 0 for row in aggregate):
        total_pairs = _numeric(aggregate[0].get("expected_pairs_mean")) or 0
        lines.extend([
            "",
            f"该测试划分的 {total_pairs:g} 个 pair 全部属于 gold 答案改变类；gold 不变类分母为 0，因此预测一致率未定义，表中以 `— (0 pairs)` 表示。",
        ])
    lines.extend([
        "",
        "逐 seed 分母、成功数、严格可解析数与 pair 明细见 `results/analysis/counterfactual_answer_change_by_seed.csv` 和 `results/analysis/counterfactual_answer_change_pairs.csv`；跨 seed 均值及 min–max 见 `results/analysis/counterfactual_answer_change_seed_mean_range.csv`。",
        CF_ANSWER_CHANGE_END,
    ])
    return lines


def _write_csv_outputs(
    output_dir: Path,
    metrics_by_seed: list[dict[str, Any]],
    metrics_aggregate: list[dict[str, Any]],
    failure_by_seed: list[dict[str, Any]],
    failure_aggregate: list[dict[str, Any]],
    resources: list[dict[str, Any]],
    resources_aggregate: list[dict[str, Any]],
    pair_details: list[dict[str, Any]],
    pair_by_seed: list[dict[str, Any]],
    pair_aggregate: list[dict[str, Any]],
) -> None:
    metric_columns = [
        "method", "seed", "split", "task_type", "n",
        *METRICS, *DIAGNOSTIC_METRICS, "abstain_examples", "truncation_observations", "truncation_rate",
        "mean_generated_tokens", "generated_tokens_total",
    ]
    metric_aggregate_columns = [
        "method", "split", "task_type", "n_seeds", "examples_per_seed",
        *[f"{metric}_{suffix}" for metric in (*METRICS, *DIAGNOSTIC_METRICS, "truncation_rate", "mean_generated_tokens") for suffix in ("mean", "min", "max")],
    ]
    failure_columns = ["method", "seed", "split", "n", *[f"{outcome}_rate" for outcome in OUTCOMES]]
    failure_aggregate_columns = [
        "method", "split", "n_seeds",
        *[f"{outcome}_rate_{suffix}" for outcome in OUTCOMES for suffix in ("mean", "min", "max")],
    ]
    resource_columns = [
        "method", "seed", "run_config_method", "warmstart_run", "warmstart_available", *RESOURCE_METRICS,
        "rollout_groups_logged", "summary_skipped_ties", "step_log_records", "failed_gradient_steps",
    ]
    resource_aggregate_columns = [
        "method", "n_seeds",
        *[f"{metric}_{suffix}" for metric in RESOURCE_METRICS for suffix in ("mean", "min", "max")],
    ]
    pair_detail_columns = [
        "method", "seed", "pair_id", "sample_ids", "n_predictions", "complete",
        "first_joint_correct", "second_joint_correct", "pair_joint_success",
    ]
    pair_run_columns = [
        "method", "seed", "expected_pairs", "complete_pairs", "incomplete_pairs",
        "paired_double_joint_success",
    ]
    pair_aggregate_columns = [
        "method", "n_seeds",
        *[f"{metric}_{suffix}" for metric in ("paired_double_joint_success", "complete_pairs", "incomplete_pairs") for suffix in ("mean", "min", "max")],
    ]
    _write_csv(output_dir / "metrics_by_seed.csv", metrics_by_seed, metric_columns)
    _write_csv(output_dir / "metrics_seed_mean_range.csv", metrics_aggregate, metric_aggregate_columns)
    _write_csv(output_dir / "failure_decomposition_by_seed.csv", failure_by_seed, failure_columns)
    _write_csv(output_dir / "failure_decomposition_seed_mean_range.csv", failure_aggregate, failure_aggregate_columns)
    _write_csv(output_dir / "run_resources_by_seed.csv", resources, resource_columns)
    _write_csv(output_dir / "run_resources_seed_mean_range.csv", resources_aggregate, resource_aggregate_columns)
    _write_csv(output_dir / "counterfactual_pairs_by_seed.csv", pair_details, pair_detail_columns)
    _write_csv(output_dir / "counterfactual_by_seed.csv", pair_by_seed, pair_run_columns)
    _write_csv(output_dir / "counterfactual_seed_mean_range.csv", pair_aggregate, pair_aggregate_columns)


def _write_counterfactual_answer_change_csvs(
    output_dir: Path,
    pair_details: list[dict[str, Any]],
    by_seed: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
) -> None:
    detail_columns = [
        "method", "seed", "pair_id", "sample_ids", "gold_answer_changed",
        "gold_first_answer", "gold_second_answer", "n_predictions", "complete",
        "first_strict_prediction_answer", "second_strict_prediction_answer",
        "both_strict_answers_available", "prediction_answer_changed",
        "correct_answer_change", "unchanged_prediction_consistent",
    ]
    by_seed_fields = [
        "expected_pairs", "gold_changed_pairs", "gold_unchanged_pairs", "complete_pairs",
        "changed_strict_answer_available_pairs", "changed_strict_answer_unavailable_pairs",
        "changed_behavior_successes", "changed_behavior_denominator", "changed_behavior_rate",
        "correct_answer_change_successes", "correct_answer_change_denominator",
        "correct_answer_change_rate", "unchanged_strict_answer_available_pairs",
        "unchanged_strict_answer_unavailable_pairs", "unchanged_consistency_successes",
        "unchanged_consistency_denominator", "unchanged_answer_consistency_rate",
    ]
    by_seed_columns = ["method", "seed", *by_seed_fields]
    aggregate_columns = [
        "method", "n_seeds",
        *[f"{field}_{suffix}" for field in by_seed_fields for suffix in ("mean", "min", "max")],
    ]
    _write_csv(output_dir / "counterfactual_answer_change_pairs.csv", pair_details, detail_columns)
    _write_csv(output_dir / "counterfactual_answer_change_by_seed.csv", by_seed, by_seed_columns)
    _write_csv(output_dir / "counterfactual_answer_change_seed_mean_range.csv", aggregate, aggregate_columns)


def _write_counterfactual_answer_change_report(report_path: Path, aggregate: list[dict[str, Any]]) -> None:
    section = "\n".join(_counterfactual_answer_change_report_lines(aggregate))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    current = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    if CF_ANSWER_CHANGE_START in current and CF_ANSWER_CHANGE_END in current:
        start = current.index(CF_ANSWER_CHANGE_START)
        end = current.index(CF_ANSWER_CHANGE_END, start) + len(CF_ANSWER_CHANGE_END)
        updated = current[:start] + section + current[end:]
    elif "\n## 训练资源" in current:
        updated = current.replace("\n## 训练资源", "\n" + section + "\n\n## 训练资源", 1)
    else:
        updated = current.rstrip() + "\n\n" + section + "\n"
    report_path.write_text(updated, encoding="utf-8")


def update_counterfactual_answer_change_diagnostics(
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    report_path: str | Path = DEFAULT_REPORT,
    methods: Iterable[str] = METHODS,
    seeds: Iterable[int] = SEEDS,
) -> dict[str, Any]:
    """Add counterfactual answer-change diagnostics without rewriting other result tables."""
    results_root = _resolve_path(results_root)
    data_dir = _resolve_path(data_dir)
    output_dir = _resolve_path(output_dir)
    report_path = _resolve_path(report_path)
    methods, seeds = tuple(methods), tuple(int(seed) for seed in seeds)
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "analysis_manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(
            "existing summary.json, analysis_manifest.json, and results report are required; "
            "run the complete analysis once before adding this diagnostic"
        )
    data_index = _load_data_index(data_dir, ("test_counterfactual",))
    examples, missing = _load_evaluations(
        results_root, data_index, methods, seeds, ("test_counterfactual",)
    )
    pair_details, by_seed, aggregate = _counterfactual_answer_change_summary(
        examples, data_index, methods, seeds
    )
    _write_counterfactual_answer_change_csvs(output_dir, pair_details, by_seed, aggregate)

    definition = {
        "gold_pair_counts": "counted from canonical test_counterfactual rows grouped by pair_id",
        "strict_answer_extraction": "one JSON object, exact keys answer/evidence, answer is string; same parse/key/type preconditions as score_response.answer_correct",
        "changed_behavior_rate": "on all gold-answer-changed pairs, both strict answers parse and predicted answers differ",
        "correct_answer_change_rate": "on all gold-answer-changed pairs, both predicted answers equal their respective gold answers",
        "unchanged_answer_consistency_rate": "on all gold-answer-unchanged pairs, both strict answers parse and predictions match",
        "denominator_policy": "all canonical pairs in the relevant gold category; missing or unparseable outputs count as failures",
    }
    diagnostic_manifest = {
        "pair_details": len(pair_details),
        "run_rows": len(by_seed),
        "missing_prediction_files": missing,
        "gold_changed_pairs_per_split": by_seed[0]["gold_changed_pairs"] if by_seed else 0,
        "gold_unchanged_pairs_per_split": by_seed[0]["gold_unchanged_pairs"] if by_seed else 0,
        "definitions": definition,
    }
    summary = _read_json(summary_path)
    manifest = _read_json(manifest_path)
    summary["counterfactual_answer_change_pairs_by_seed"] = pair_details
    summary["counterfactual_answer_change_by_seed"] = by_seed
    summary["counterfactual_answer_change_seed_mean_range"] = aggregate
    summary_manifest = summary.setdefault("manifest", {})
    summary_manifest["analysis_version"] = "results-summary-v4"
    summary_manifest["counterfactual_answer_change_diagnostics"] = diagnostic_manifest
    summary_manifest["counterfactual_answer_change_definitions"] = definition
    manifest["analysis_version"] = "results-summary-v4"
    manifest["counterfactual_answer_change_diagnostics"] = diagnostic_manifest
    manifest["counterfactual_answer_change_definitions"] = definition
    _write_json(summary_path, summary)
    _write_json(manifest_path, manifest)
    _write_counterfactual_answer_change_report(report_path, aggregate)
    return {
        "pair_details": len(pair_details),
        "run_rows": len(by_seed),
        "missing_prediction_files": missing,
        "gold_changed_pairs_per_split": diagnostic_manifest["gold_changed_pairs_per_split"],
        "gold_unchanged_pairs_per_split": diagnostic_manifest["gold_unchanged_pairs_per_split"],
        "methods": [
            {
                "method": row["method"],
                "n_seeds": row["n_seeds"],
                "changed_behavior_rate_mean": row["changed_behavior_rate_mean"],
                "correct_answer_change_rate_mean": row["correct_answer_change_rate_mean"],
                "unchanged_answer_consistency_rate_mean": row["unchanged_answer_consistency_rate_mean"],
            }
            for row in aggregate
        ],
    }


def _write_summary_report(
    report_path: Path,
    metrics_aggregate: list[dict[str, Any]],
    resources_aggregate: list[dict[str, Any]],
    pair_aggregate: list[dict[str, Any]],
    counterfactual_answer_change_aggregate: list[dict[str, Any]],
    warmstart_candidates: list[dict[str, Any]],
    warmstart_selected: str | None,
    method_difference_aggregate: list[dict[str, Any]],
    dpo_logp_summaries: list[dict[str, Any]],
    dpo_logp_summary_aggregate: list[dict[str, Any]],
    baseline_validation_fence_counts: tuple[int, int] | None,
    missing: list[str],
    incomplete_coverage: list[dict[str, Any]],
    missing_training_summaries: list[str],
    error_case_count: int,
) -> None:
    lines = [
        "# 实验结果",
        "",
        "## 评价口径",
        "",
        "逐样本主指标根据保存的模型输出与 canonical 数据标签重新计算。联合正确要求答案正确且证据编号集合完全匹配。表内先按种子计算，再对可用种子等权汇总；两个种子时显示 `均值 [最小值, 最大值]`，单种子时显示 `数值 (n=1)`，不将其表述为双种子均值。基线只运行一次，SFT-256监督对照只运行 seed 17。验证集仅用于训练过程的选择，测试集汇总用于最终报告，不作测试集调参。",
        "",
        "## 测试集总体表现",
        "",
        "下表为严格 JSON 格式正确率、答案正确率、证据精确率和联合正确率；分数范围为 0–1。",
        "",
        "| 测试划分 | 方法 | 格式正确 | 答案正确 | 证据精确 | 联合正确 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lookup = {
        (row["method"], row["split"], row["task_type"]): row
        for row in metrics_aggregate
    }
    for split in TEST_SPLITS:
        for method in METHODS:
            row = lookup.get((method, split, "overall"))
            if row is None:
                continue
            lines.append(
                f"| {SPLIT_LABELS[split]} | {METHOD_LABELS[method]} | "
                f"{_fmt_seed_range(row, 'format_valid')} | {_fmt_seed_range(row, 'answer_correct')} | "
                f"{_fmt_seed_range(row, 'evidence_exact')} | {_fmt_seed_range(row, 'joint_correct')} |"
            )

    lines.extend([
        "",
        "## 诊断性指标（非主指标）",
        "",
        "宽松答案正确率使用 `lenient_answer` 从原始输出抽取答案后与 gold 比较；去围栏联合正确率只移除一对首尾 Markdown JSON 代码围栏（开头的反引号与 json 标签，以及末尾的反引号），再按严格 `score_response` 重算联合正确。对所有方法使用同一规则。这些诊断不改变严格 JSON 主指标，不参与训练奖励或测试集模型选择。",
        "",
        "| 评估划分 | 方法 | 宽松答案正确率 | 去围栏联合正确率 |",
        "|---|---|---:|---:|",
    ])
    for split in EVAL_SPLITS:
        for method in METHODS:
            row = lookup.get((method, split, "overall"))
            if row is None:
                continue
            lines.append(
                f"| {SPLIT_LABELS[split]} | {METHOD_LABELS[method]} | "
                f"{_fmt_seed_range(row, 'lenient_answer_correct')} | "
                f"{_fmt_seed_range(row, 'normalized_joint_correct')} |"
            )
    baseline_validation = lookup.get(("baseline", "validation", "overall"))
    if baseline_validation is not None and baseline_validation_fence_counts is not None:
        fenced_count, total_count = baseline_validation_fence_counts
        lines.append("")
        lines.append(
            f"基线验证集严格格式正确率为 {_fmt_seed_range(baseline_validation, 'format_valid')}，"
            f"严格答案正确率为 {_fmt_seed_range(baseline_validation, 'answer_correct')}；"
            f"{fenced_count}/{total_count} 条输出带首尾 Markdown 代码围栏。"
            f"`lenient_answer` 正确率为 {_fmt_seed_range(baseline_validation, 'lenient_answer_correct')}，"
            f"仅去除外围围栏后的联合正确率为 {_fmt_seed_range(baseline_validation, 'normalized_joint_correct')}。"
            "因此严格答案分数为零不表示模型没有生成可抽取的正确答案；格式诊断与答案内容诊断应分别解读。"
        )

    lines.extend([
        "",
        "## SFT warmstart 验证候选",
        "",
        "仅依据 validation 计算。预先规则为严格格式正确率至少 0.95 且联合正确率低于 0.95；符合条件时选择训练步数最少的候选。此选择没有使用任何测试划分。",
        "",
        "| 候选 | Seed | 步数 | 样本数 | 格式正确 | 联合正确 | 训练时长 s | 更新 tokens | 峰值 allocated GiB | 符合门槛 | 选择 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in warmstart_candidates:
        lines.append(
            f"| {row['candidate']} | {row['seed']} | {row['steps']} | {row['n']} | "
            f"{_fmt(row['format_valid'])} | {_fmt(row['joint_correct'])} | "
            f"{_fmt(row['wall_time_seconds'])} | {_fmt(row['updated_tokens'], 0)} | "
            f"{_fmt(row['peak_cuda_allocated_gib'])} | "
            f"{'是' if row['eligible'] else '否'} | {'选中' if row['selected'] else ''} |"
        )
    if warmstart_selected is None:
        lines.append("")
        lines.append("没有候选同时满足门槛；当前主 SFT 运行配置不由此分析脚本修改。")
    else:
        selected_row = next(row for row in warmstart_candidates if row["run"] == warmstart_selected)
        lines.append("")
        lines.append(
            f"按该规则选择 {selected_row['candidate']}（{warmstart_selected}）；主训练与结果目录保持原样。"
        )

    lines.extend(["", "## 同分布测试的任务分层", "", "联合正确率；双种子显示均值及 min–max，单种子标注 n=1。", "", "| 方法 | 直接查找 | 两步关系/加法 | 不可回答 | 否定排除 |", "|---|---:|---:|---:|---:|"])
    for method in METHODS:
        cells = []
        found = False
        for task in TASK_TYPES:
            row = lookup.get((method, "test_id", task))
            cells.append(_fmt_seed_range(row, "joint_correct") if row else "—")
            found = found or row is not None
        if found:
            lines.append(f"| {METHOD_LABELS[method]} | " + " | ".join(cells) + " |")

    comparison_lookup = {
        (row["comparison"], row["split"], row["task_type"], row["metric"]): row
        for row in method_difference_aggregate
    }
    lines.extend([
        "",
        "## DPO 与 GRPO 对照",
        "",
        "下表给出按相同 seed 配对的严格联合正确率差值，正值表示左侧方法高于对照方法；方括号为种子间 min–max。DPO 相对 SFT 的差值用于核对后训练是否退化，full-reward GRPO 与 format-only GRPO 的差值用于区分奖励目标的影响。差值本身不说明因果机制。",
        "",
        "| 测试划分 | DPO − SFT | Full GRPO − Format GRPO |",
        "|---|---:|---:|",
    ])
    for split in TEST_SPLITS:
        dpo_row = comparison_lookup.get(("dpo_minus_sft", split, "overall", "joint_correct"))
        grpo_row = comparison_lookup.get(("grpo_full_minus_format", split, "overall", "joint_correct"))
        if dpo_row is not None or grpo_row is not None:
            lines.append(
                f"| {SPLIT_LABELS[split]} | "
                f"{_fmt_seed_range(dpo_row, 'difference') if dpo_row else '—'} | "
                f"{_fmt_seed_range(grpo_row, 'difference') if grpo_row else '—'} |"
            )
    dpo_test_differences = [
        comparison_lookup.get(("dpo_minus_sft", split, "overall", "joint_correct"))
        for split in TEST_SPLITS
    ]
    grpo_test_differences = [
        comparison_lookup.get(("grpo_full_minus_format", split, "overall", "joint_correct"))
        for split in TEST_SPLITS
    ]
    if all(row is not None and row["difference_max"] < 0 for row in dpo_test_differences):
        lines.append("")
        lines.append("两个种子下 DPO 的严格联合正确率均低于同种子 SFT，且三项测试差值的种子范围都为负；这是观察到的后训练退化，结论限于本实验配置与合成评测集。")
    if all(row is not None and row["difference_min"] > 0 for row in grpo_test_differences):
        lines.append("在两个种子的配对差值中，Full GRPO 的严格联合正确率均高于 format-only GRPO；该结果说明本实验设置下完整奖励优于格式奖励，不能外推为其他数据或模型上的普遍结论。")

    lines.extend([
        "",
        "## DPO 偏好 log-prob 训练轨迹",
        "",
        "训练日志保存的 chosen_reward 与 rejected_reward 分别等于 β 乘以 chosen/rejected 的 policy 相对 reference 总序列 log-prob 差；图和 CSV 用 β 还原这两个差值，并直接保留日志的 preference_margin。它不是 policy 的绝对 log-prob。该轨迹是训练动态的支持线索，不能单独证明测试集退化的原因。",
        "",
        "| Seed | 步数 | β | Chosen Δlog-prob 起点→终点 | Rejected Δlog-prob 起点→终点 | Implicit margin 起点→终点 |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in dpo_logp_summaries:
        lines.append(
            f"| {row['seed']} | {row['steps']} | {_fmt(row['beta'])} | "
            f"{_fmt(row['start_chosen_policy_ref_logp_delta'])} → {_fmt(row['end_chosen_policy_ref_logp_delta'])} | "
            f"{_fmt(row['start_rejected_policy_ref_logp_delta'])} → {_fmt(row['end_rejected_policy_ref_logp_delta'])} | "
            f"{_fmt(row['start_implicit_margin'])} → {_fmt(row['end_implicit_margin'])} |"
        )
    for row in dpo_logp_summary_aggregate:
        lines.append(
            f"| 两种子均值 [min, max] | {_fmt_seed_range(row, 'steps')} | {_fmt(row.get('beta_mean'))} | "
            f"{_fmt_seed_range(row, 'start_chosen_policy_ref_logp_delta')} → {_fmt_seed_range(row, 'end_chosen_policy_ref_logp_delta')} | "
            f"{_fmt_seed_range(row, 'start_rejected_policy_ref_logp_delta')} → {_fmt_seed_range(row, 'end_rejected_policy_ref_logp_delta')} | "
            f"{_fmt_seed_range(row, 'start_implicit_margin')} → {_fmt_seed_range(row, 'end_implicit_margin')} |"
        )
    if len(dpo_logp_summaries) >= 2 and all(
        row["end_implicit_margin"] > row["start_implicit_margin"]
        and row["end_rejected_policy_ref_logp_delta"] < row["start_rejected_policy_ref_logp_delta"]
        for row in dpo_logp_summaries
    ):
        lines.append("")
        lines.append("两个 DPO 种子的隐式 margin 都从接近零升至较大正值；rejected 相对参考模型的 log-prob 差也比 chosen 更大幅度下降。这个趋势与训练中加强 chosen/rejected 区分相符，可作为测试退化的机制线索，但仅凭日志不能确认它导致了泛化下降。")
    lines.append("")
    lines.append("逐步图：`figures/dpo_logp_margin.png`；逐步源数据与每种子摘要见 `results/analysis/dpo_logp_margin_by_step.csv` 和 `results/analysis/dpo_logp_margin_summary_by_seed.csv`。")

    lines.extend(["", "## 反事实成对成功率", "", "每对两条样本均达到 joint_correct 才记为 1；任一题失败记为 0。只对完整预测对计算成功率，完成对数见下表和 `results/analysis/counterfactual_pairs_by_seed.csv` 的逐对记录。", "", "| 方法 | 双题联合成功率 | 完成种子数 |", "|---|---:|---:|"])
    pair_lookup = {row["method"]: row for row in pair_aggregate}
    for method in METHODS:
        row = pair_lookup.get(method)
        if row:
            lines.append(
                f"| {METHOD_LABELS[method]} | {_fmt_seed_range(row, 'paired_double_joint_success')} | {row.get('n_seeds', 0)} |"
            )

    lines.extend(_counterfactual_answer_change_report_lines(counterfactual_answer_change_aggregate))

    lines.extend([
        "",
        "## 训练资源",
        "",
        "时长来自各 run_summary，生成与更新 token 数为对应训练阶段日志累计值。表中区分后训练阶段本身与加上其 adapter_input 所指 SFT warmstart 的总成本；SFT 阶段和 SFT-256 对照的两列数值相同。端到端峰值显存取 warmstart 与后训练阶段各自峰值的较大者，不相加。显存按 summary 原始 byte 数除以 1024³ 换算为 GiB；运行设备容量为 6 GiB（6144 MiB）。运行时长可能受 Unity 等共享进程影响，因此只报告观测值，不视为独占硬件基准。",
        "",
        "| 方法 | 阶段时长 s | 含 warmstart 时长 s | 阶段生成 tokens | 含 warmstart 生成 tokens | 阶段更新 tokens | 含 warmstart 更新 tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    resource_lookup = {row["method"]: row for row in resources_aggregate}
    for method in TRAIN_METHODS:
        row = resource_lookup.get(method)
        if row:
            lines.append(
                f"| {METHOD_LABELS[method]} | {_fmt_seed_range(row, 'wall_time_seconds')} | "
                f"{_fmt_seed_range(row, 'end_to_end_wall_time_seconds')} | "
                f"{_fmt_seed_range(row, 'generated_tokens')} | {_fmt_seed_range(row, 'end_to_end_generated_tokens')} | "
                f"{_fmt_seed_range(row, 'updated_tokens')} | {_fmt_seed_range(row, 'end_to_end_updated_tokens')} |"
            )
    lines.extend([
        "",
        "| 方法 | 阶段峰值 allocated GiB | 含 warmstart 峰值 allocated GiB | 阶段峰值 reserved GiB | 含 warmstart 峰值 reserved GiB | 有效 rollout 组 | 平局组 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for method in TRAIN_METHODS:
        row = resource_lookup.get(method)
        if row:
            lines.append(
                f"| {METHOD_LABELS[method]} | {_fmt_seed_range(row, 'peak_cuda_allocated_gib')} | "
                f"{_fmt_seed_range(row, 'end_to_end_peak_cuda_allocated_gib')} | "
                f"{_fmt_seed_range(row, 'peak_cuda_reserved_gib')} | "
                f"{_fmt_seed_range(row, 'end_to_end_peak_cuda_reserved_gib')} | "
                f"{_fmt_seed_range(row, 'effective_groups')} | {_fmt_seed_range(row, 'tied_groups')} |"
            )

    lines.extend([
        "",
        "## 错误案例与图表",
        "",
        f"已按预先固定规则选取 {error_case_count} 个代表性错误，完整提示、参考答案、模型输出和评分保存在 `results/analysis/error_cases.jsonl`。每种方法、每类错误至多选 1 例；优先种子 17，再 29，再按 `test_id`、`test_ood`、`test_counterfactual` 与样本 ID 排序。该规则只用于展示，不用于模型选择。",
        "",
        "主图源文件为 `figures/fig1_test_joint_accuracy.png`（三测试划分联合正确率）、`figures/fig2_task_error_analysis.png`（任务分层与错误分解）、`figures/fig3_training_resources_curves.png`（阶段资源、loss 与 rollout 奖励曲线）和 `figures/dpo_logp_margin.png`（DPO 相对参考 log-prob 与隐式 margin）；对应 PDF、CSV/JSON 源数据和复用分析代码保存在工程目录。根目录实验报告以相对路径 `figures/` 引用图片。",
        "",
    ])
    if missing or incomplete_coverage or missing_training_summaries:
        lines.extend([
            "## 结果覆盖",
            "",
            f"本次汇总缺少 {len(missing)} 个预期预测文件；另有 {len(incomplete_coverage)} 个方法/种子/划分的预测数与 canonical 数据规模不符，并缺少 {len(missing_training_summaries)} 个训练运行汇总。详情见 `results/analysis/evaluation_coverage.csv` 与 `results/analysis/analysis_manifest.json`；不对缺失样本或运行资源外推。",
            "",
        ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _save_figure(fig: Any, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")


def _matplotlib_setup() -> tuple[Any, Any]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError as exc:
        raise RuntimeError("result figures require matplotlib; install the project's declared plotting dependency") from exc
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for font in ("Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
        if font in installed:
            plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    return matplotlib, plt


def _draw_figures(
    figures_dir: Path,
    metrics_aggregate: list[dict[str, Any]],
    failure_aggregate: list[dict[str, Any]],
    resources_aggregate: list[dict[str, Any]],
    curves: list[dict[str, Any]],
) -> list[str]:
    _, plt = _matplotlib_setup()
    import numpy as np

    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    colors = {
        "baseline": "#777777",
        "sft": "#4477AA",
        "sft_long": "#66AADD",
        "dpo": "#228833",
        "online_dpo": "#CC6677",
        "grpo": "#AA3377",
        "grpo_format": "#EE7733",
    }
    all_methods = [method for method in METHODS if any(row["method"] == method for row in metrics_aggregate)]
    aggregate_lookup = {
        (row["method"], row["split"], row["task_type"]): row
        for row in metrics_aggregate
    }

    # Figure 1: overall joint correctness across the three held-out test sets.
    fig, axes = plt.subplots(1, len(TEST_SPLITS), figsize=(16, 5.1), sharey=True)
    for axis, split in zip(axes, TEST_SPLITS):
        means, low, high = [], [], []
        panel_methods = []
        for method in all_methods:
            row = aggregate_lookup.get((method, split, "overall"))
            if row is None or _numeric(row.get("joint_correct_mean")) is None:
                continue
            mean = float(row["joint_correct_mean"])
            minimum = float(row["joint_correct_min"])
            maximum = float(row["joint_correct_max"])
            means.append(mean)
            low.append(max(0.0, mean - minimum))
            high.append(max(0.0, maximum - mean))
            panel_methods.append(method)
        x = np.arange(len(panel_methods))
        axis.bar(
            x,
            means,
            yerr=[low, high],
            capsize=3,
            color=[colors[method] for method in panel_methods],
            edgecolor="white",
            linewidth=0.6,
        )
        axis.set_xticks(
            x,
            [
                _method_display_label(
                    method,
                    aggregate_lookup.get((method, split, "overall"), {}).get("n_seeds"),
                )
                for method in panel_methods
            ],
            rotation=30,
            ha="right",
        )
        axis.set_title(SPLIT_LABELS[split])
        axis.set_ylim(0, 1.0)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        if split == "test_counterfactual":
            axis.text(
                0.02, 0.97, "样本级；双题配对率见表",
                transform=axis.transAxes, va="top", fontsize=8,
            )
    axes[0].set_ylabel("联合正确率")
    fig.suptitle("三类测试集的样本级联合正确率", y=0.99)
    fig.text(
        0.5, 0.94,
        "双种子均值；单次运行标 n=1；误差线为种子 min–max",
        ha="center", va="top", fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    base = figures_dir / "fig1_test_joint_accuracy"
    _save_figure(fig, base)
    plt.close(fig)
    outputs.extend((str(base.with_suffix(".png")), str(base.with_suffix(".pdf"))))

    # Figure 2: task-stratified accuracy and mutually exclusive error outcomes.
    fig, (ax_heat, ax_stack) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.05, 1.35]})
    heat_methods = all_methods
    matrix = np.full((len(heat_methods), len(TASK_TYPES)), np.nan)
    for row_index, method in enumerate(heat_methods):
        for col_index, task in enumerate(TASK_TYPES):
            row = aggregate_lookup.get((method, "test_id", task))
            if row is not None:
                matrix[row_index, col_index] = row["joint_correct_mean"]
    image = ax_heat.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax_heat.set_xticks(np.arange(len(TASK_TYPES)), [TASK_LABELS[task] for task in TASK_TYPES], rotation=25, ha="right")
    ax_heat.set_yticks(
        np.arange(len(heat_methods)),
        [
            _method_display_label(
                method,
                aggregate_lookup.get((method, "test_id", "overall"), {}).get("n_seeds"),
            )
            for method in heat_methods
        ],
    )
    ax_heat.set_title("同分布测试：任务分层联合正确率")
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = matrix[r, c]
            ax_heat.text(
                c, r, "—" if np.isnan(value) else f"{value:.2f}",
                ha="center", va="center", fontsize=9,
                color="white" if not np.isnan(value) and value > 0.55 else "black",
            )
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.045, pad=0.03)
    colorbar.set_label("联合正确率")

    failure_lookup = {(row["method"], row["split"]): row for row in failure_aggregate}
    y = np.arange(len(heat_methods))
    left = np.zeros(len(heat_methods))
    stack_colors = {
        "joint_correct": "#228833",
        "evidence_error": "#4477AA",
        "answer_error": "#EE7733",
        "format_invalid": "#CC6677",
    }
    for outcome in OUTCOMES:
        values = []
        for method in heat_methods:
            row = failure_lookup.get((method, "test_id"), {})
            values.append(float(row.get(f"{outcome}_rate_mean") or 0.0))
        ax_stack.barh(
            y, values, left=left, label=OUTCOME_LABELS[outcome],
            color=stack_colors[outcome], edgecolor="white", linewidth=0.5,
        )
        left += np.asarray(values)
    ax_stack.set_yticks(
        y,
        [
            _method_display_label(
                method,
                aggregate_lookup.get((method, "test_id", "overall"), {}).get("n_seeds"),
            )
            for method in heat_methods
        ],
    )
    ax_stack.invert_yaxis()
    ax_stack.set_xlim(0, 1)
    ax_stack.set_xlabel("样本比例")
    ax_stack.set_title("同分布测试：互斥结果分解")
    ax_stack.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    fig.suptitle("任务分层表现与主要错误类型", y=1.02)
    fig.tight_layout()
    base = figures_dir / "fig2_task_error_analysis"
    _save_figure(fig, base)
    plt.close(fig)
    outputs.extend((str(base.with_suffix(".png")), str(base.with_suffix(".pdf"))))

    # Figure 3: training costs and per-step curves; no model is loaded here.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    resource_lookup = {row["method"]: row for row in resources_aggregate}
    training_methods = [method for method in TRAIN_METHODS if method in resource_lookup]
    for axis, stage_metric, total_metric, label, title in (
        (axes[0, 0], "wall_time_seconds", "end_to_end_wall_time_seconds", "训练时长（秒）", "阶段耗时与含 warmstart 总耗时"),
        (axes[0, 1], "peak_cuda_allocated_gib", "end_to_end_peak_cuda_allocated_gib", "峰值显存（GiB）", "阶段峰值与含 warmstart 峰值 allocated"),
    ):
        x = np.arange(len(training_methods))
        width = 0.36
        for index, method in enumerate(training_methods):
            row = resource_lookup[method]
            for metric, offset, legend_label, alpha, hatch in (
                (stage_metric, -width / 2, "训练阶段", 1.0, None),
                (total_metric, width / 2, "含 warmstart", 0.55, "///"),
            ):
                mean = _numeric(row.get(f"{metric}_mean"))
                minimum = _numeric(row.get(f"{metric}_min"))
                maximum = _numeric(row.get(f"{metric}_max"))
                if mean is None or minimum is None or maximum is None:
                    continue
                axis.bar(
                    index + offset, mean,
                    yerr=[[max(0.0, mean - minimum)], [max(0.0, maximum - mean)]],
                    width=width * 0.9, capsize=3, color=colors[method], alpha=alpha,
                    hatch=hatch, label=legend_label if index == 0 else None,
                )
        axis.set_xticks(
            x,
            [
                _method_display_label(method, resource_lookup[method].get("n_seeds"))
                for method in training_methods
            ],
            rotation=25,
            ha="right",
        )
        axis.set_ylabel(label)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, fontsize=8)

    curve_methods = [method for method in TRAIN_METHODS if any(row.get("loss") is not None and row["method"] == method for row in curves)]
    for method in curve_methods:
        buckets: dict[int, list[float]] = defaultdict(list)
        for point in curves:
            if point["method"] == method and _numeric(point.get("loss")) is not None:
                buckets[int(point["step"])].append(float(point["loss"]))
        xs = sorted(buckets)
        means = [statistics.fmean(buckets[x]) for x in xs]
        low = [min(buckets[x]) for x in xs]
        high = [max(buckets[x]) for x in xs]
        n_seeds = len({point["seed"] for point in curves if point["method"] == method})
        axes[1, 0].plot(
            xs, means, label=_method_display_label(method, n_seeds),
            color=colors[method], linewidth=1.5,
        )
        if any(a != b for a, b in zip(low, high)):
            axes[1, 0].fill_between(xs, low, high, color=colors[method], alpha=0.12)
    axes[1, 0].set_xlabel("更新步")
    axes[1, 0].set_ylabel("训练损失")
    axes[1, 0].set_title("训练损失曲线（跨种子逐步均值）")
    axes[1, 0].grid(alpha=0.25)
    if curve_methods:
        axes[1, 0].legend(frameon=False, fontsize=8)
    else:
        axes[1, 0].text(0.5, 0.5, "无可用 step_log", ha="center", va="center", transform=axes[1, 0].transAxes)

    reward_methods = ("online_dpo", "grpo", "grpo_format")
    for method in reward_methods:
        buckets = defaultdict(list)
        for point in curves:
            if point["method"] == method and _numeric(point.get("reward_mean")) is not None:
                buckets[int(point["step"])].append(float(point["reward_mean"]))
        if not buckets:
            continue
        xs = sorted(buckets)
        means = [statistics.fmean(buckets[x]) for x in xs]
        low = [min(buckets[x]) for x in xs]
        high = [max(buckets[x]) for x in xs]
        n_seeds = len({point["seed"] for point in curves if point["method"] == method})
        axes[1, 1].plot(
            xs, means, label=_method_display_label(method, n_seeds),
            color=colors[method], linewidth=1.5,
        )
        if any(a != b for a, b in zip(low, high)):
            axes[1, 1].fill_between(xs, low, high, color=colors[method], alpha=0.12)
    axes[1, 1].set_xlabel("成功更新步")
    axes[1, 1].set_ylabel("rollout 平均奖励")
    axes[1, 1].set_title("在线方法的 rollout 奖励曲线")
    axes[1, 1].grid(alpha=0.25)
    if axes[1, 1].lines:
        axes[1, 1].legend(frameon=False, fontsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, "无可用 rollout 奖励日志", ha="center", va="center", transform=axes[1, 1].transAxes)
    fig.suptitle("训练资源与优化过程（资源误差线为种子 min–max）", y=1.02)
    fig.tight_layout()
    base = figures_dir / "fig3_training_resources_curves"
    _save_figure(fig, base)
    plt.close(fig)
    outputs.extend((str(base.with_suffix(".png")), str(base.with_suffix(".pdf"))))
    outputs.extend(_draw_dpo_logp_figure(figures_dir, curves))
    return outputs


def analyze_results(
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    report_path: str | Path | None = DEFAULT_REPORT,
    methods: Iterable[str] = METHODS,
    seeds: Iterable[int] = SEEDS,
    splits: Iterable[str] = EVAL_SPLITS,
    make_plots: bool = True,
    write_report: bool = True,
) -> dict[str, Any]:
    """Read completed artifacts, compute summaries, write tables/plots/report."""
    results_root = _resolve_path(results_root)
    data_dir = _resolve_path(data_dir)
    output_dir = _resolve_path(output_dir)
    figures_dir = _resolve_path(figures_dir)
    methods = tuple(methods)
    seeds = tuple(int(seed) for seed in seeds)
    splits = tuple(splits)
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    data_index = _load_data_index(data_dir, splits)
    examples, missing = _load_evaluations(results_root, data_index, methods, seeds, splits)
    coverage = _evaluation_coverage(results_root, data_index, examples, methods, seeds, splits)
    incomplete_coverage = [row for row in coverage if not row["complete"]]
    metrics_by_seed = _metrics_by_seed(examples)
    metrics_aggregate = _aggregate_seed_rows(
        metrics_by_seed,
        ("method", "split", "task_type"),
        (*METRICS, *DIAGNOSTIC_METRICS, "truncation_rate", "mean_generated_tokens"),
    )
    metrics_aggregate.sort(
        key=lambda row: (
            METHOD_ORDER.get(row["method"], 99),
            SPLIT_ORDER.get(row["split"], 99),
            -1 if row["task_type"] == "overall" else TASK_TYPES.index(row["task_type"]),
        )
    )
    failure_by_seed = _failure_rows(examples)
    failure_aggregate = _failure_seed_aggregate(failure_by_seed)
    resources, curves, missing_training_summaries = _read_training_resources(results_root, methods, seeds)
    resources_aggregate = _resource_seed_aggregate(resources)
    warmstart_candidates, warmstart_selected = _warmstart_candidate_selection(
        results_root, data_index.get("validation", {})
    ) if "validation" in data_index else ([], None)
    baseline_validation_rows = [
        example for example in examples
        if example["method"] == "baseline" and example["split"] == "validation"
    ]
    baseline_validation_fence_counts = None
    if baseline_validation_rows:
        baseline_validation_fence_counts = (
            sum(
                len(example["prediction"].strip()) >= 6
                and example["prediction"].strip().startswith("```")
                and example["prediction"].strip().endswith("```")
                for example in baseline_validation_rows
            ),
            len(baseline_validation_rows),
        )
    method_differences_by_seed, method_difference_aggregate = _matched_method_differences(metrics_by_seed)
    dpo_logp_summaries, dpo_logp_summary_aggregate = _dpo_logp_summaries(curves)
    dpo_logp_by_step = sorted(
        (point for point in curves if point.get("method") == "dpo"),
        key=lambda point: (int(point["seed"]), int(point["step"])),
    )
    pair_details, pair_by_seed, pair_aggregate = _counterfactual_summary(examples, data_index)
    cf_answer_change_details, cf_answer_change_by_seed, cf_answer_change_aggregate = (
        _counterfactual_answer_change_summary(examples, data_index, methods, seeds)
    )
    error_cases = _select_error_cases(examples)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_outputs(
        output_dir,
        metrics_by_seed,
        metrics_aggregate,
        failure_by_seed,
        failure_aggregate,
        resources,
        resources_aggregate,
        pair_details,
        pair_by_seed,
        pair_aggregate,
    )
    _write_csv(
        output_dir / "evaluation_coverage.csv",
        coverage,
        [
            "method", "seed", "split", "prediction_file", "prediction_file_exists",
            "expected_examples", "observed_examples", "coverage_rate", "complete",
        ],
    )
    _write_jsonl(output_dir / "error_cases.jsonl", error_cases)
    _write_csv(
        output_dir / "warmstart_validation_candidates.csv",
        warmstart_candidates,
        [
            "candidate", "run", "seed", "steps", "n", "prediction_file",
            "format_valid", "answer_correct", "joint_correct", "format_pass_95",
            "joint_below_95", "eligible", "selected", "wall_time_seconds",
            "generated_tokens", "updated_tokens", "peak_cuda_allocated_gib", "peak_cuda_reserved_gib",
        ],
    )
    _write_csv(
        output_dir / "method_differences_by_seed.csv",
        method_differences_by_seed,
        ["comparison", "method", "reference", "seed", "split", "task_type", "metric", "difference"],
    )
    _write_csv(
        output_dir / "method_differences_seed_mean_range.csv",
        method_difference_aggregate,
        [
            "comparison", "split", "task_type", "metric", "n_seeds",
            "difference_mean", "difference_min", "difference_max",
        ],
    )
    _write_csv(
        output_dir / "dpo_logp_margin_by_step.csv",
        dpo_logp_by_step,
        [
            "method", "seed", "step", "beta", "loss", "chosen_reward", "rejected_reward",
            "chosen_policy_ref_logp_delta", "rejected_policy_ref_logp_delta",
            "preference_margin", "implicit_margin_recomputed", "implicit_margin_check_error",
        ],
    )
    dpo_summary_columns = [
        "method", "seed", "steps", "beta", "start_step", "end_step",
        "start_chosen_policy_ref_logp_delta", "end_chosen_policy_ref_logp_delta",
        "start_rejected_policy_ref_logp_delta", "end_rejected_policy_ref_logp_delta",
        "start_implicit_margin", "end_implicit_margin", "mean_implicit_margin",
        "max_abs_margin_check_error",
    ]
    _write_csv(output_dir / "dpo_logp_margin_summary_by_seed.csv", dpo_logp_summaries, dpo_summary_columns)
    _write_csv(
        output_dir / "dpo_logp_margin_summary_seed_mean_range.csv",
        dpo_logp_summary_aggregate,
        ["method", "n_seeds", *[
            f"{field}_{suffix}"
            for field in dpo_summary_columns[2:]
            if field not in ("start_step", "end_step")
            for suffix in ("mean", "min", "max")
        ]],
    )
    _write_counterfactual_answer_change_csvs(
        output_dir,
        cf_answer_change_details,
        cf_answer_change_by_seed,
        cf_answer_change_aggregate,
    )
    if make_plots:
        figure_paths = _draw_figures(
            figures_dir, metrics_aggregate, failure_aggregate, resources_aggregate, curves
        )
    else:
        figure_paths = []
    if write_report and report_path is not None:
        report_path = _resolve_path(report_path)
        _write_summary_report(
            report_path,
            metrics_aggregate,
            resources_aggregate,
            pair_aggregate,
            cf_answer_change_aggregate,
            warmstart_candidates,
            warmstart_selected,
            method_difference_aggregate,
            dpo_logp_summaries,
            dpo_logp_summary_aggregate,
            baseline_validation_fence_counts,
            missing,
            incomplete_coverage,
            missing_training_summaries,
            len(error_cases),
        )

    manifest = {
        "analysis_version": "results-summary-v4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "expected_methods": list(methods),
        "expected_seeds": list(seeds),
        "method_evaluation_seeds": {
            method: list(_seeds_for_method(method, seeds)) for method in methods
        },
        "single_seed_methods": [
            method for method in methods if len(_seeds_for_method(method, seeds)) == 1
        ],
        "expected_splits": list(splits),
        "prediction_examples": len(examples),
        "missing_prediction_files": missing,
        "incomplete_evaluation_runs": incomplete_coverage,
        "runs_with_training_summaries": len(resources),
        "missing_training_summaries": missing_training_summaries,
        "counterfactual_pair_details": len(pair_details),
        "counterfactual_answer_change_diagnostics": {
            "pair_details": len(cf_answer_change_details),
            "run_rows": len(cf_answer_change_by_seed),
            "gold_changed_pairs_per_split": cf_answer_change_by_seed[0]["gold_changed_pairs"] if cf_answer_change_by_seed else 0,
            "gold_unchanged_pairs_per_split": cf_answer_change_by_seed[0]["gold_unchanged_pairs"] if cf_answer_change_by_seed else 0,
        },
        "counterfactual_answer_change_definitions": {
            "strict_answer_extraction": "one JSON object, exact keys answer/evidence, answer is string",
            "changed_behavior_rate": "both strict answers parse and predicted answers differ over all gold-changed pairs",
            "correct_answer_change_rate": "both predictions equal their respective gold answers over all gold-changed pairs",
            "unchanged_answer_consistency_rate": "both strict answers parse and predictions match over all gold-unchanged pairs",
            "denominator_policy": "all canonical pairs of the relevant gold class; missing/unparseable outputs count as failures",
        },
        "selected_error_cases": len(error_cases),
        "warmstart_candidate_selection": {
            "rule": "validation format_valid >= 0.95 and joint_correct < 0.95; choose minimum steps",
            "selected_run": warmstart_selected,
            "candidates": warmstart_candidates,
        },
        "method_differences_by_seed": len(method_differences_by_seed),
        "dpo_logp_steps": len(dpo_logp_by_step),
        "dpo_logp_margin_check_max_abs_error": max(
            (abs(float(point["implicit_margin_check_error"])) for point in dpo_logp_by_step
             if _numeric(point.get("implicit_margin_check_error")) is not None),
            default=None,
        ),
        "baseline_validation_outer_fenced": {
            "n": baseline_validation_fence_counts[0],
            "total": baseline_validation_fence_counts[1],
        } if baseline_validation_fence_counts is not None else None,
        "test_sets_used_for_model_selection": False,
        "training_resources_note": "wall time may include contention from shared Unity processes; no exclusive benchmark claim",
        "figure_paths": figure_paths,
        "selection_rule": "每方法、每错误类别至多1例；优先种子17，再29，随后test_id、test_ood、test_counterfactual及升序id。",
        "metric_source": "strict main metrics use score_response on original prediction text and canonical rows; lenient_answer_correct and normalized_joint_correct are diagnostic only",
        "diagnostic_metric_definitions": {
            "lenient_answer_correct": "lenient_answer(original_prediction) == canonical_answer",
            "normalized_joint_correct": "score_response after removing only one outer ```json ... ``` fence",
        },
    }
    _write_json(output_dir / "analysis_manifest.json", manifest)
    summary = {
        "manifest": manifest,
        "evaluation_coverage": coverage,
        "metrics_by_seed": metrics_by_seed,
        "metrics_seed_mean_range": metrics_aggregate,
        "failure_decomposition_by_seed": failure_by_seed,
        "failure_decomposition_seed_mean_range": failure_aggregate,
        "run_resources_by_seed": resources,
        "missing_training_summaries": missing_training_summaries,
        "run_resources_seed_mean_range": resources_aggregate,
        "counterfactual_pairs_by_seed": pair_details,
        "counterfactual_by_seed": pair_by_seed,
        "counterfactual_seed_mean_range": pair_aggregate,
        "counterfactual_answer_change_pairs_by_seed": cf_answer_change_details,
        "counterfactual_answer_change_by_seed": cf_answer_change_by_seed,
        "counterfactual_answer_change_seed_mean_range": cf_answer_change_aggregate,
        "warmstart_validation_candidates": warmstart_candidates,
        "warmstart_selected_run": warmstart_selected,
        "method_differences_by_seed": method_differences_by_seed,
        "method_differences_seed_mean_range": method_difference_aggregate,
        "dpo_logp_margin_by_step": dpo_logp_by_step,
        "dpo_logp_margin_summary_by_seed": dpo_logp_summaries,
        "dpo_logp_margin_summary_seed_mean_range": dpo_logp_summary_aggregate,
        "error_cases": error_cases,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--no-plots", action="store_true", help="write tables only")
    parser.add_argument("--no-report", action="store_true", help="do not write research/experiment_results.md")
    parser.add_argument(
        "--counterfactual-diagnostics-only",
        action="store_true",
        help="add answer-change diagnostics from counterfactual predictions without rewriting other results",
    )
    args = parser.parse_args()
    if args.counterfactual_diagnostics_only:
        result = update_counterfactual_answer_change_diagnostics(
            results_root=args.results_root,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            report_path=args.report,
            methods=args.methods,
            seeds=args.seeds,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    summary = analyze_results(
        results_root=args.results_root,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        report_path=args.report,
        methods=args.methods,
        seeds=args.seeds,
        make_plots=not args.no_plots,
        write_report=not args.no_report,
    )
    manifest = summary["manifest"]
    print(json.dumps({
        "prediction_examples": manifest["prediction_examples"],
        "runs_with_training_summaries": manifest["runs_with_training_summaries"],
        "selected_error_cases": manifest["selected_error_cases"],
        "missing_prediction_files": len(manifest["missing_prediction_files"]),
        "missing_training_summaries": len(manifest["missing_training_summaries"]),
        "figure_paths": manifest["figure_paths"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
