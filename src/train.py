"""SFT, offline DPO, OnlineDPO, GRPO, and per-example evaluation CLI."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from src.modeling import (
    MODEL_ID,
    MODEL_REVISION,
    activate_train_adapter,
    adapter_parameter_counts,
    batch_sequence_logps,
    collate_features,
    dpo_loss,
    frozen_adapter,
    generate_candidates,
    generated_feature,
    grpo_loss,
    group_advantages,
    load_policy_reference_model,
    load_sft_model,
    load_tokenizer,
    model_logits,
    preference_features,
    prompt_token_ids,
    sequence_logps,
    set_adapter_for_inference,
    sft_feature,
    token_logps_from_logits,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "experiment.json"
METHODS = ("sft", "dpo", "online_dpo", "grpo", "evaluate")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"input JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row at {path}:{line_no} must be an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"input JSONL is empty: {path}")
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(stream: Any, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    stream.flush()


def assert_training_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path_parts = [Path(part).stem.lower() for part in path.parts]
    if any(
        part in {"test", "tests", "valid", "validation"}
        or part.startswith(("test_", "validation_", "valid_"))
        for part in path_parts
    ):
        raise ValueError(f"training input path appears to be a held-out split: {path}")
    for index, row in enumerate(rows):
        split = str(row.get("split", "")).lower()
        if not split.startswith("train"):
            raise ValueError(
                f"refusing to train on row {row.get('id', index)} with split={split!r}; "
                "training inputs must identify a train* split"
            )


def assert_evaluation_rows(rows: list[dict[str, Any]], path: Path) -> None:
    for index, row in enumerate(rows):
        split = str(row.get("split", "")).lower()
        if split.startswith("train"):
            raise ValueError(f"refusing to evaluate row {row.get('id', index)} from training split")
        if not ("test" in split or "valid" in split or "validation" in split):
            raise ValueError(f"evaluation row {row.get('id', index)} has unexpected split={split!r}")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def choose_rows(rows: list[dict[str, Any]], rng: random.Random, count: int) -> list[dict[str, Any]]:
    return [rng.choice(rows) for _ in range(count)]


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> random.Random:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return rng


def disable_dropout(model: torch.nn.Module) -> None:
    """Keep likelihood forwards deterministic while checkpointed updates train."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        # Qwen keeps attention dropout as a scalar in its config/module.
        if hasattr(module, "attention_dropout"):
            try:
                module.attention_dropout = 0.0
            except (AttributeError, TypeError):
                pass
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "attention_dropout"):
            config.attention_dropout = 0.0


def optimizer_for(model: torch.nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters are active")
    return torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)


def clip_gradients(model: torch.nn.Module) -> tuple[float | None, bool]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
    finite = bool(torch.isfinite(norm).item())
    return (float(norm.detach()) if finite else None), finite


def cuda_memory(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {"cuda_allocated_bytes": 0, "cuda_reserved_bytes": 0}
    return {
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def peak_cuda_memory(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {"peak_cuda_allocated_bytes": 0, "peak_cuda_reserved_bytes": 0}
    return {
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def log_step(stream: Any, step: int, started: float, device: torch.device, **metrics: Any) -> None:
    values: dict[str, Any] = {
        "step": step,
        "wall_time_seconds": round(time.perf_counter() - started, 3),
        **cuda_memory(device),
        **metrics,
    }
    append_jsonl(stream, values)


def save_adapter(model: torch.nn.Module, output: Path, adapter_name: str | None) -> Path:
    adapter_dir = output / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    if adapter_name is None:
        model.save_pretrained(str(adapter_dir))
    else:
        model.set_adapter(adapter_name)
        model.save_pretrained(str(adapter_dir), selected_adapters=[adapter_name])
    selected_dir = adapter_dir / adapter_name if adapter_name and adapter_name != "default" else adapter_dir
    actual_dir = selected_dir if (selected_dir / "adapter_config.json").is_file() else adapter_dir
    return actual_dir


def init_run(
    method: str,
    args: argparse.Namespace,
    settings: dict[str, Any],
    rows: list[dict[str, Any]],
    output: Path,
    model: torch.nn.Module,
    device: torch.device,
    input_path: Path,
    run_started: float | None = None,
) -> tuple[float, Path]:
    output.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = run_started if run_started is not None else time.perf_counter()
    counts = adapter_parameter_counts(model)
    run_config = {
        "method": method,
        "seed": settings["seed"],
        "input": str(input_path),
        "output": str(output),
        "model_id": args.model_path or MODEL_ID,
        "model_revision": None if args.model_path else MODEL_REVISION,
        "adapter_input": settings.get("adapter"),
        "device": str(device),
        "dtype": "bfloat16",
        "dataset_rows": len(rows),
        "parameters": counts,
        "settings": settings,
    }
    write_json(output / "run_config.json", run_config)
    return started, output / "step_log.jsonl"


def finish_run(
    output: Path,
    started: float,
    device: torch.device,
    generated_tokens: int,
    updated_tokens: int,
    updates: int,
    extra: dict[str, Any] | None = None,
) -> None:
    summary = {
        "updates": updates,
        "training_updates": updates,
        "wall_time_seconds": round(time.perf_counter() - started, 3),
        "generated_tokens": int(generated_tokens),
        "updated_tokens": int(updated_tokens),
        **peak_cuda_memory(device),
    }
    if extra:
        summary.update(extra)
    write_json(output / "run_summary.json", summary)


def build_optimizer_step_batches(
    rows: list[dict[str, Any]],
    rng: random.Random,
    batch_size: int,
    grad_accum: int,
) -> list[list[dict[str, Any]]]:
    return [choose_rows(rows, rng, batch_size) for _ in range(grad_accum)]


def train_sft(args: argparse.Namespace, settings: dict[str, Any]) -> None:
    run_started = time.perf_counter()
    input_path, output = project_path(settings["input"]), project_path(settings["output"])
    rows = read_jsonl(input_path)
    assert_training_rows(rows, input_path)
    rng = seed_everything(settings["seed"])
    tokenizer = load_tokenizer(args.model_path)
    features = [
        sft_feature(tokenizer, row, settings["max_prompt_length"], settings["max_completion_length"])
        for row in rows
    ]
    features_by_object = {id(row): feature for row, feature in zip(rows, features)}
    device = choose_device(settings["device"])
    model = load_sft_model(args.model_path, str(device), settings["lora_rank"], settings["lora_alpha"])
    disable_dropout(model)
    optimizer = optimizer_for(model, settings["learning_rate"])
    started, log_path = init_run("sft", args, settings, rows, output, model, device, input_path, run_started)
    updated_tokens = 0
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, settings["steps"] + 1):
            optimizer.zero_grad(set_to_none=True)
            loss_values: list[float] = []
            step_tokens = 0
            model.train()
            for micro_rows in build_optimizer_step_batches(rows, rng, settings["batch_size"], settings["grad_accum"]):
                micro = [features_by_object[id(row)] for row in micro_rows]
                batch = collate_features(micro, tokenizer.pad_token_id)
                logits = model_logits(model, batch, device)
                labels = batch["labels"][:, 1:].to(device).contiguous().view(-1)
                shifted = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
                loss = F.cross_entropy(shifted.float(), labels, ignore_index=-100)
                (loss / settings["grad_accum"]).backward()
                loss_values.append(float(loss.detach()))
                step_tokens += int((batch["labels"][:, 1:] != -100).sum().item())
            grad_norm, gradient_finite = clip_gradients(model)
            if not gradient_finite:
                log_step(
                    log, step, started, device, loss=sum(loss_values) / len(loss_values),
                    generated_tokens=0, update_tokens=step_tokens, gradient_norm=None,
                    gradient_finite=False, updated=False,
                )
                raise FloatingPointError("non-finite SFT gradient norm; optimizer step was skipped")
            optimizer.step()
            updated_tokens += step_tokens
            log_step(
                log, step, started, device, loss=sum(loss_values) / len(loss_values),
                generated_tokens=0, update_tokens=step_tokens,
                gradient_norm=grad_norm, gradient_finite=True,
            )
    adapter_path = save_adapter(model, output, None)
    finish_run(output, started, device, 0, updated_tokens, settings["steps"], {"adapter_path": str(adapter_path)})


def load_preference_rows(input_path: Path, seed: int) -> list[dict[str, Any]]:
    rows = read_jsonl(input_path)
    assert_training_rows(rows, input_path)
    if all("chosen" in row and "rejected" in row for row in rows):
        return rows
    from src.data import make_preference

    rng = random.Random(seed)
    result = [make_preference(row, rng) for row in rows]
    assert_training_rows(result, input_path)
    return result


def preference_microbatch(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    settings: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    chosen: list[dict[str, list[int]]] = []
    rejected: list[dict[str, list[int]]] = []
    for row in rows:
        left, right = preference_features(
            tokenizer, row, settings["max_prompt_length"], settings["max_completion_length"]
        )
        chosen.append(left)
        rejected.append(right)
    features: list[dict[str, list[int]]] = []
    for left, right in zip(chosen, rejected):
        features.extend((left, right))
    batch = collate_features(features, tokenizer.pad_token_id)
    model.eval()
    with torch.no_grad(), frozen_adapter(model, "reference", "policy"):
        reference_logps, _ = batch_sequence_logps(model, batch, device)
    activate_train_adapter(model, "policy")
    model.train()
    policy_logps, _ = batch_sequence_logps(model, batch, device)
    policy_logps = policy_logps.reshape(-1, 2)
    reference_logps = reference_logps.reshape(-1, 2)
    loss, metrics = dpo_loss(
        policy_logps[:, 0], policy_logps[:, 1], reference_logps[:, 0], reference_logps[:, 1], settings["beta"]
    )
    token_count = int(batch["response_mask"][:, 1:].sum().item())
    return loss, metrics, token_count


def train_dpo(args: argparse.Namespace, settings: dict[str, Any]) -> None:
    run_started = time.perf_counter()
    input_path, output = project_path(settings["input"]), project_path(settings["output"])
    rows = load_preference_rows(input_path, settings["seed"])
    rng = seed_everything(settings["seed"])
    tokenizer = load_tokenizer(args.model_path)
    device = choose_device(settings["device"])
    model = load_policy_reference_model(args.model_path, str(project_path(settings["adapter"])), str(device))
    disable_dropout(model)
    optimizer = optimizer_for(model, settings["learning_rate"])
    started, log_path = init_run("dpo", args, settings, rows, output, model, device, input_path, run_started)
    updated_tokens = 0
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, settings["steps"] + 1):
            optimizer.zero_grad(set_to_none=True)
            losses: list[float] = []
            step_tokens = 0
            all_stats: dict[str, list[float]] = {}
            for micro_rows in build_optimizer_step_batches(rows, rng, settings["batch_size"], settings["grad_accum"]):
                loss, stats, count = preference_microbatch(model, tokenizer, micro_rows, device, settings)
                (loss / settings["grad_accum"]).backward()
                losses.append(float(loss.detach()))
                step_tokens += count
                for name, value in stats.items():
                    all_stats.setdefault(name, []).extend(value.reshape(-1).detach().float().cpu().tolist())
            grad_norm, gradient_finite = clip_gradients(model)
            if not gradient_finite:
                log_step(
                    log, step, started, device, loss=sum(losses) / len(losses),
                    generated_tokens=0, update_tokens=step_tokens, gradient_norm=None,
                    gradient_finite=False, updated=False,
                )
                raise FloatingPointError("non-finite DPO gradient norm; optimizer step was skipped")
            optimizer.step()
            updated_tokens += step_tokens
            flattened = {name: sum(values) / len(values) for name, values in all_stats.items() if values}
            log_step(
                log, step, started, device, loss=sum(losses) / len(losses),
                generated_tokens=0, update_tokens=step_tokens, gradient_norm=grad_norm,
                gradient_finite=True, **flattened,
            )
    adapter_path = save_adapter(model, output, "policy")
    finish_run(output, started, device, 0, updated_tokens, settings["steps"], {"adapter_path": str(adapter_path)})


def _reward(text: str, row: dict[str, Any], mode: str) -> tuple[float, dict[str, Any]]:
    from src.rewards import format_only_reward, score_response

    details = score_response(text, row)
    value = float(details["reward"]) if mode == "full" else float(format_only_reward(text))
    if not math.isfinite(value):
        raise ValueError("reward function returned a non-finite value")
    return value, details


def _rollout_group(
    model: torch.nn.Module,
    tokenizer: Any,
    row: dict[str, Any],
    device: torch.device,
    rng: random.Random,
    settings: dict[str, Any],
    method: str,
    rollouts: Any,
    rollout_id: int,
) -> tuple[dict[str, Any], int]:
    candidates, prompt_ids = generate_candidates(
        model,
        tokenizer,
        str(row["prompt"]),
        device,
        count=settings["group_size"],
        max_prompt_length=settings["max_prompt_length"],
        max_new_tokens=settings["max_new_tokens"],
        adapter_name="policy",
        row_id=row.get("id"),
    )
    reward_values: list[float] = []
    for candidate in candidates:
        value, score = _reward(candidate["text"], row, settings["reward_mode"])
        candidate["reward"] = value
        candidate["score"] = score
        reward_values.append(value)
    record = {
        "rollout_id": rollout_id,
        "method": method,
        "reward_mode": settings["reward_mode"],
        "id": row.get("id"),
        "split": row.get("split"),
        "task_type": row.get("task_type"),
        "prompt": row["prompt"],
        "candidates": [
            {
                "index": index,
                "text": candidate["text"],
                "reward": candidate["reward"],
                "score": candidate["score"],
                "generated_tokens": candidate["generated_tokens"],
                "truncated": candidate["truncated"],
            }
            for index, candidate in enumerate(candidates)
        ],
    }
    if method == "online_dpo":
        maximum, minimum = max(reward_values), min(reward_values)
        if maximum == minimum:
            record["decision"] = "skip_tied_rewards"
            record["chosen_index"] = None
            record["rejected_index"] = None
        else:
            best = rng.choice([index for index, value in enumerate(reward_values) if value == maximum])
            worst = rng.choice([index for index, value in enumerate(reward_values) if value == minimum])
            record["decision"] = "preference_pair"
            record["chosen_index"] = best
            record["rejected_index"] = worst
    else:
        record["advantages"] = group_advantages(
            torch.tensor(reward_values, dtype=torch.float32), settings["group_size"], settings["advantage_eps"]
        ).tolist()
    if method == "online_dpo":
        append_jsonl(rollouts, record)
    record["_prompt_ids"] = prompt_ids
    record["_candidates"] = candidates
    return record, sum(int(candidate["generated_tokens"]) for candidate in candidates)


def _online_preference_features(group: dict[str, Any]) -> list[dict[str, list[int]]]:
    chosen = group["_candidates"][group["chosen_index"]]
    rejected = group["_candidates"][group["rejected_index"]]
    prompt_ids = group["_prompt_ids"]
    return [
        generated_feature(prompt_ids, chosen["completion_ids"]),
        generated_feature(prompt_ids, rejected["completion_ids"]),
    ]


def train_online_dpo(args: argparse.Namespace, settings: dict[str, Any]) -> None:
    run_started = time.perf_counter()
    input_path, output = project_path(settings["input"]), project_path(settings["output"])
    rows = read_jsonl(input_path)
    assert_training_rows(rows, input_path)
    rng = seed_everything(settings["seed"])
    tokenizer = load_tokenizer(args.model_path)
    device = choose_device(settings["device"])
    model = load_policy_reference_model(args.model_path, str(project_path(settings["adapter"])), str(device))
    disable_dropout(model)
    optimizer = optimizer_for(model, settings["learning_rate"])
    started, log_path = init_run("online_dpo", args, settings, rows, output, model, device, input_path, run_started)
    generated_tokens = updated_tokens = rollout_id = 0
    rollout_prompts = skipped_ties = 0
    updates = attempts = 0
    rollout_budget = settings.get("max_rollout_prompts")
    with log_path.open("w", encoding="utf-8") as log, (output / "rollouts.jsonl").open("w", encoding="utf-8") as rollouts:
        while updates < settings["steps"] and (rollout_budget is None or rollout_prompts < rollout_budget):
            attempts += 1
            if rollout_budget is None and attempts > settings["steps"] * 20:
                raise RuntimeError("too many tie-only OnlineDPO rollouts; rewards do not provide preference pairs")
            optimizer.zero_grad(set_to_none=True)
            micro_pairs: list[list[dict[str, list[int]]]] = []
            micro_rewards: list[float] = []
            step_generated_tokens = 0
            prompt_count = settings["batch_size"] * settings["grad_accum"]
            if rollout_budget is not None:
                prompt_count = min(prompt_count, rollout_budget - rollout_prompts)
            prompt_rows = choose_rows(rows, rng, prompt_count)
            for start in range(0, prompt_count, settings["batch_size"]):
                pairs: list[list[dict[str, list[int]]]] = []
                for row in prompt_rows[start:start + settings["batch_size"]]:
                    group, count = _rollout_group(
                        model, tokenizer, row, device, rng, settings, "online_dpo", rollouts, rollout_id
                    )
                    rollout_id += 1
                    rollout_prompts += 1
                    generated_tokens += count
                    step_generated_tokens += count
                    if group["decision"] == "preference_pair":
                        pairs.append(_online_preference_features(group))
                        micro_rewards.extend(item["reward"] for item in group["_candidates"])
                    else:
                        skipped_ties += 1
                if pairs:
                    micro_pairs.append([feature for pair in pairs for feature in pair])
            if not micro_pairs:
                log_step(
                    log, attempts, started, device, updated=False, skipped_tie_only=True,
                    generated_tokens=step_generated_tokens, update_tokens=0,
                )
                continue
            losses: list[float] = []
            all_stats: dict[str, list[float]] = {}
            step_update_tokens = 0
            total_pairs = sum(len(features) // 2 for features in micro_pairs)
            for features in micro_pairs:
                batch = collate_features(features, tokenizer.pad_token_id)
                model.eval()
                with torch.no_grad(), frozen_adapter(model, "reference", "policy"):
                    reference_logps, _ = batch_sequence_logps(model, batch, device)
                activate_train_adapter(model, "policy")
                model.train()
                policy_logps, _ = batch_sequence_logps(model, batch, device)
                policy_logps = policy_logps.reshape(-1, 2)
                reference_logps = reference_logps.reshape(-1, 2)
                loss, stats = dpo_loss(
                    policy_logps[:, 0], policy_logps[:, 1],
                    reference_logps[:, 0], reference_logps[:, 1], settings["beta"],
                )
                (loss * ((len(features) // 2) / total_pairs)).backward()
                losses.append(float(loss.detach()))
                batch_tokens = int(batch["response_mask"][:, 1:].sum().item())
                step_update_tokens += batch_tokens
                for name, value in stats.items():
                    all_stats.setdefault(name, []).extend(value.reshape(-1).detach().float().cpu().tolist())
            grad_norm, gradient_finite = clip_gradients(model)
            if not gradient_finite:
                log_step(
                    log, attempts, started, device, updated=False, loss=sum(losses) / len(losses),
                    generated_tokens=step_generated_tokens, update_tokens=step_update_tokens,
                    gradient_norm=None, gradient_finite=False,
                )
                raise FloatingPointError("non-finite OnlineDPO gradient norm; optimizer step was skipped")
            optimizer.step()
            updates += 1
            updated_tokens += step_update_tokens
            flattened = {name: sum(values) / len(values) for name, values in all_stats.items() if values}
            log_step(
                log, updates, started, device, attempt=attempts, updated=True,
                loss=sum(losses) / len(losses), reward_mean=sum(micro_rewards) / len(micro_rewards),
                generated_tokens=step_generated_tokens, update_tokens=step_update_tokens,
                gradient_norm=grad_norm, gradient_finite=True, **flattened
            )
    adapter_path = save_adapter(model, output, "policy")
    finish_run(
        output, started, device, generated_tokens, updated_tokens, updates,
        {
            "adapter_path": str(adapter_path),
            "rollout_prompts": rollout_prompts,
            "skipped_ties": skipped_ties,
            "max_rollout_prompts": rollout_budget,
            "rollout_budget_exhausted": rollout_budget is not None and rollout_prompts >= rollout_budget,
        },
    )


def _make_grpo_rollout_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    rng: random.Random,
    settings: dict[str, Any],
    device: torch.device,
    rollouts: Any,
    first_rollout_id: int,
) -> tuple[list[dict[str, Any]], int]:
    groups: list[dict[str, Any]] = []
    generated_tokens = 0
    for offset, row in enumerate(rows):
        group, count = _rollout_group(
            model, tokenizer, row, device, rng, settings, "grpo", rollouts, first_rollout_id + offset
        )
        generated_tokens += count
        candidates = group["_candidates"]
        features = [generated_feature(group["_prompt_ids"], item["completion_ids"]) for item in candidates]
        old_tokens: list[torch.Tensor] = []
        reference_tokens: list[torch.Tensor] = []
        # One completion per forward keeps the live logits small on 6 GiB GPUs.
        model.eval()
        for feature in features:
            batch = collate_features([feature], tokenizer.pad_token_id)
            with torch.no_grad():
                old_sum, old_token = batch_sequence_logps(model, batch, device)
                with frozen_adapter(model, "reference", "policy"):
                    reference_sum, reference_token = batch_sequence_logps(model, batch, device)
            old_tokens.append(old_token.detach().cpu())
            reference_tokens.append(reference_token.detach().cpu())
        advantages = group_advantages(
            torch.tensor([item["reward"] for item in candidates], dtype=torch.float32),
            settings["group_size"],
            settings["advantage_eps"],
        )
        group["_features"] = features
        group["_old_tokens"] = old_tokens
        group["_reference_tokens"] = reference_tokens
        group["_advantages"] = advantages
        for index, (item, old, reference, advantage) in enumerate(
            zip(candidates, old_tokens, reference_tokens, advantages.tolist())
        ):
            item["old_logps"] = float(old.sum().item())
            item["reference_logps"] = float(reference.sum().item())
            item["advantage"] = float(advantage)
            group["candidates"][index].update(
                old_logps=item["old_logps"],
                reference_logps=item["reference_logps"],
                advantage=item["advantage"],
            )
        append_jsonl(rollouts, {key: value for key, value in group.items() if not key.startswith("_")})
        groups.append(group)
    return groups, generated_tokens


def _update_grpo_groups(
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    groups: list[dict[str, Any]],
    device: torch.device,
    settings: dict[str, Any],
) -> tuple[dict[str, float], int]:
    optimizer.zero_grad(set_to_none=True)
    model.train()
    loss_values: list[float] = []
    stats: dict[str, list[float]] = {"policy_loss": [], "kl": [], "clip_fraction": [], "mean_ratio": []}
    token_count = 0
    denominator = max(1, sum(len(group["_features"]) for group in groups))
    for group in groups:
        for index, feature in enumerate(group["_features"]):
            batch = collate_features([feature], tokenizer.pad_token_id)
            logits = model_logits(model, batch, device)
            input_ids = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device)
            response_mask = batch["response_mask"][:, 1:].to(device) & attention[:, 1:].bool()
            current_tokens = token_logps_from_logits(
                logits, input_ids, batch["response_mask"].to(device), attention
            )
            valid_current = current_tokens
            valid_old = group["_old_tokens"][index].to(device)
            valid_reference = group["_reference_tokens"][index].to(device)
            # The shifted arrays contain zero prompt positions; response_mask
            # selects just the response targets, including EOS.
            current_response = valid_current
            old_response = valid_old
            reference_response = valid_reference
            advantage = group["_advantages"][index:index + 1].to(device)
            loss, metrics = grpo_loss(
                current_response, old_response, reference_response, response_mask, advantage,
                settings["clip_epsilon"], settings["beta"],
            )
            (loss / denominator).backward()
            loss_values.append(float(loss.detach()))
            token_count += int(response_mask.sum().item())
            for name in stats:
                stats[name].append(float(metrics[name].detach()))
    grad_norm, gradient_finite = clip_gradients(model)
    if gradient_finite:
        optimizer.step()
    return {
        "loss": sum(loss_values) / len(loss_values),
        "gradient_norm": grad_norm,
        "gradient_finite": gradient_finite,
        **{key: sum(values) / len(values) for key, values in stats.items()},
    }, token_count


def train_grpo(args: argparse.Namespace, settings: dict[str, Any]) -> None:
    run_started = time.perf_counter()
    input_path, output = project_path(settings["input"]), project_path(settings["output"])
    rows = read_jsonl(input_path)
    assert_training_rows(rows, input_path)
    rng = seed_everything(settings["seed"])
    tokenizer = load_tokenizer(args.model_path)
    device = choose_device(settings["device"])
    model = load_policy_reference_model(args.model_path, str(project_path(settings["adapter"])), str(device))
    disable_dropout(model)
    optimizer = optimizer_for(model, settings["learning_rate"])
    started, log_path = init_run("grpo", args, settings, rows, output, model, device, input_path, run_started)
    generated_tokens = updated_tokens = rollout_id = 0
    retained_groups: list[dict[str, Any]] | None = None
    with log_path.open("w", encoding="utf-8") as log, (output / "rollouts.jsonl").open("w", encoding="utf-8") as rollouts:
        for step in range(1, settings["steps"] + 1):
            epoch = (step - 1) % settings["update_epochs"]
            step_generated_tokens = 0
            if epoch == 0 or retained_groups is None:
                prompt_rows = [row for batch in build_optimizer_step_batches(rows, rng, settings["batch_size"], settings["grad_accum"]) for row in batch]
                retained_groups, count = _make_grpo_rollout_batch(
                    model, tokenizer, prompt_rows, rng, settings, device, rollouts, rollout_id
                )
                rollout_id += len(prompt_rows)
                generated_tokens += count
                step_generated_tokens = count
            metrics, step_tokens = _update_grpo_groups(
                model, tokenizer, optimizer, retained_groups, device, settings
            )
            group_rewards = [item["reward"] for group in retained_groups for item in group["_candidates"]]
            log_step(
                log, step, started, device, update_epoch=epoch + 1,
                rollout_batch=(step - 1) // settings["update_epochs"] + 1,
                updated=metrics["gradient_finite"],
                reward_mean=sum(group_rewards) / len(group_rewards),
                generated_tokens=step_generated_tokens, update_tokens=step_tokens, **metrics,
            )
            if not metrics["gradient_finite"]:
                raise FloatingPointError("non-finite GRPO gradient norm; optimizer step was skipped")
            updated_tokens += step_tokens
    adapter_path = save_adapter(model, output, "policy")
    finish_run(
        output, started, device, generated_tokens, updated_tokens, settings["steps"],
        {"adapter_path": str(adapter_path), "rollout_prompts": rollout_id},
    )


def greedy_responses(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    max_prompt_length: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    model.eval()
    prompt_ids = [
        prompt_token_ids(tokenizer, str(row["prompt"]), max_prompt_length, row.get("id"))
        for row in rows
    ]
    width = max(len(ids) for ids in prompt_ids)
    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for ids in prompt_ids:
        padding = width - len(ids)
        input_rows.append([tokenizer.pad_token_id] * padding + ids)
        mask_rows.append([0] * padding + [1] * len(ids))
    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=device)
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )[:, width:].tolist()
    eos_id = tokenizer.eos_token_id
    results: list[dict[str, Any]] = []
    for new_ids in generated:
        if eos_id is not None and eos_id in new_ids:
            end = new_ids.index(eos_id)
            completion = new_ids[:end + 1]
            text_ids = new_ids[:end]
            truncated = False
        else:
            completion = new_ids
            text_ids = new_ids
            truncated = len(completion) >= max_new_tokens
        results.append({
            "text": tokenizer.decode(text_ids, skip_special_tokens=True).strip(),
            "generated_tokens": len(completion),
            "truncated": bool(truncated),
        })
    return results


def greedy_response(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_prompt_length: int,
    max_new_tokens: int,
    row_id: Any = None,
) -> dict[str, Any]:
    return greedy_responses(
        model, tokenizer, [{"id": row_id, "prompt": prompt}], device, max_prompt_length, max_new_tokens
    )[0]


def evaluate(args: argparse.Namespace, settings: dict[str, Any]) -> None:
    from src.rewards import score_response

    run_started = time.perf_counter()
    input_path, output = project_path(settings["input"]), project_path(settings["output"])
    rows = read_jsonl(input_path)
    assert_evaluation_rows(rows, input_path)
    tokenizer = load_tokenizer(args.model_path)
    device = choose_device(settings["device"])
    model = load_base_for_eval(args.model_path, str(device), settings.get("adapter"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"
    counts = {key: 0.0 for key in ("format_valid", "answer_correct", "evidence_exact", "evidence_f1", "joint_correct")}
    total_tokens = 0
    tokenizer.padding_side = "left"
    with predictions_path.open("w", encoding="utf-8") as predictions:
        for start in range(0, len(rows), settings["batch_size"]):
            batch_rows = rows[start:start + settings["batch_size"]]
            responses = greedy_responses(
                model, tokenizer, batch_rows, device,
                settings["max_prompt_length"], settings["max_new_tokens"],
            )
            for row, response in zip(batch_rows, responses):
                score = score_response(response["text"], row)
                total_tokens += response["generated_tokens"]
                for key in counts:
                    counts[key] += float(score[key])
                append_jsonl(predictions, {
                    "id": row.get("id"), "split": row.get("split"), "task_type": row.get("task_type"),
                    "prompt": row["prompt"], "prediction": response["text"],
                    "score": score, "generated_tokens": response["generated_tokens"],
                    "truncated": response["truncated"],
                })
    active = getattr(model, "active_adapter", None)
    if isinstance(active, list):
        active = active[0] if active else None
    peft_config = getattr(model, "peft_config", {})
    adapter_config = peft_config.get(active) if active is not None else None
    adapter_summary = None
    if adapter_config is not None:
        adapter_summary = {
            "active_adapter": active,
            "rank": getattr(adapter_config, "r", None),
            "alpha": getattr(adapter_config, "lora_alpha", None),
            "target_modules": sorted(adapter_config.target_modules) if isinstance(adapter_config.target_modules, set) else adapter_config.target_modules,
            "dropout": getattr(adapter_config, "lora_dropout", None),
        }
    summary = {
        "method": "evaluate",
        "input": str(input_path),
        "examples": len(rows),
        "batch_size": settings["batch_size"],
        "generated_tokens": total_tokens,
        "mean_metrics": {key: value / len(rows) for key, value in counts.items()},
        "prediction_file": str(predictions_path),
        "wall_time_seconds": round(time.perf_counter() - run_started, 3),
        "device": str(device),
        "model_id": args.model_path or MODEL_ID,
        "model_revision": None if args.model_path else MODEL_REVISION,
        "adapter_path": None if not settings.get("adapter") or settings["adapter"].lower() == "none" else str(project_path(settings["adapter"])),
        "adapter_config": adapter_summary,
        "parameters": adapter_parameter_counts(model),
        **peak_cuda_memory(device),
    }
    write_json(output / "evaluation_summary.json", summary)


def load_base_for_eval(model_name_or_path: str | None, device: str, adapter_path: str | None) -> torch.nn.Module:
    from src.modeling import load_base_model
    if adapter_path and adapter_path.lower() != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(load_base_model(model_name_or_path, device), str(project_path(adapter_path)))
    else:
        model = load_base_model(model_name_or_path, device)
    set_adapter_for_inference(model)
    return model


def read_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("experiment config must be a JSON object")
    return value


def parser_for(config: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    commands = parser.add_subparsers(dest="command", required=True)
    for method in METHODS:
        sub = commands.add_parser(method)
        sub.add_argument("--config", default=argparse.SUPPRESS)
        sub.add_argument("--seed", type=int, default=None)
        sub.add_argument("--steps", type=int, default=None)
        sub.add_argument("--batch-size", type=int, default=None)
        sub.add_argument("--grad-accum", type=int, default=None)
        sub.add_argument("--input", default=None)
        sub.add_argument("--output", default=None)
        sub.add_argument("--adapter", default=None)
        sub.add_argument("--model-path", default=None)
        sub.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None)
        sub.add_argument("--learning-rate", type=float, default=None)
        sub.add_argument("--max-prompt-length", type=int, default=None)
        sub.add_argument("--max-completion-length", type=int, default=None)
        sub.add_argument("--max-new-tokens", type=int, default=None)
        if method in ("online_dpo", "grpo"):
            sub.add_argument("--reward-mode", choices=("full", "format"), default=None)
        if method == "online_dpo":
            sub.add_argument("--max-rollout-prompts", type=int, default=None)
        if method == "grpo":
            sub.add_argument("--update-epochs", type=int, default=None)
    return parser


def resolve_settings(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "seed": config["seed"],
        **config["model"],
        **config["training"],
        **config["commands"][args.command],
        **config.get("objectives", {}).get(args.command, {}),
    }
    overrides = {
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "input": args.input,
        "output": args.output,
        "adapter": args.adapter,
        "learning_rate": args.learning_rate,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
    }
    if hasattr(args, "reward_mode"):
        overrides["reward_mode"] = args.reward_mode
    if hasattr(args, "max_rollout_prompts"):
        overrides["max_rollout_prompts"] = args.max_rollout_prompts
    if hasattr(args, "update_epochs"):
        overrides["update_epochs"] = args.update_epochs
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value
    positive_keys = ["batch_size", "max_prompt_length", "max_completion_length", "max_new_tokens"]
    if args.command != "evaluate":
        positive_keys.extend(("steps", "grad_accum"))
    for key in positive_keys:
        if int(settings[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if settings.get("update_epochs", 1) < 1:
        raise ValueError("update_epochs must be at least 1")
    if settings.get("max_rollout_prompts") is not None and settings["max_rollout_prompts"] < 1:
        raise ValueError("max_rollout_prompts must be positive when provided")
    return settings


def main(argv: Iterable[str] | None = None) -> None:
    args = parser_for({}).parse_args(argv)
    config = read_config(project_path(args.config))
    settings = resolve_settings(args, config)
    if args.command == "sft":
        train_sft(args, settings)
    elif args.command == "dpo":
        train_dpo(args, settings)
    elif args.command == "online_dpo":
        train_online_dpo(args, settings)
    elif args.command == "grpo":
        train_grpo(args, settings)
    else:
        evaluate(args, settings)


if __name__ == "__main__":
    main()
