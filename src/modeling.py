"""Shared low-memory model, tokenization, and objective helpers."""

from __future__ import annotations

import contextlib
import json
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def target_text(row: dict[str, Any]) -> str:
    """Return the canonical supervised response for a task row."""
    target = row.get("target")
    if target is None:
        if "answer" not in row or "evidence" not in row:
            raise ValueError(f"row {row.get('id', '<unknown>')} lacks target/answer/evidence")
        target = {"answer": row["answer"], "evidence": row["evidence"]}
    if isinstance(target, str):
        try:
            target = json.loads(target)
        except json.JSONDecodeError as exc:
            raise ValueError(f"row {row.get('id', '<unknown>')} has invalid JSON target") from exc
    if not isinstance(target, dict) or set(target) != {"answer", "evidence"}:
        raise ValueError(f"row {row.get('id', '<unknown>')} target must contain answer and evidence")
    if not isinstance(target["answer"], str) or not isinstance(target["evidence"], list):
        raise ValueError(f"row {row.get('id', '<unknown>')} target has invalid answer/evidence types")
    if any(type(index) is not int or index < 1 for index in target["evidence"]):
        raise ValueError(f"row {row.get('id', '<unknown>')} target evidence must contain positive integers")
    target = {"answer": target["answer"], "evidence": target["evidence"]}
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def render_prompt(tokenizer: Any, prompt: str) -> str:
    """Render one user prompt with the model's actual chat template."""
    if not hasattr(tokenizer, "apply_chat_template") or not tokenizer.chat_template:
        raise ValueError("the selected tokenizer does not define a chat template")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def prompt_token_ids(tokenizer: Any, prompt: str, max_prompt_length: int, row_id: Any = None) -> list[int]:
    rendered = render_prompt(tokenizer, prompt)
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("rendered prompt has no tokens")
    if len(ids) > max_prompt_length:
        label = f" for row {row_id}" if row_id is not None else ""
        raise ValueError(
            f"prompt{label} has {len(ids)} tokens, exceeding max_prompt_length={max_prompt_length}; "
            "raise the limit instead of truncating task instructions"
        )
    return list(ids)


def _completion_ids(tokenizer: Any, text: Any, max_completion_length: int, row_id: Any = None) -> list[int]:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    ids = list(tokenizer(_as_text(text), add_special_tokens=False)["input_ids"])
    if len(ids) + 1 > max_completion_length:
        label = f" for row {row_id}" if row_id is not None else ""
        raise ValueError(
            f"completion{label} has {len(ids) + 1} tokens including EOS, exceeding "
            f"max_completion_length={max_completion_length}; raise the limit instead of truncating the target"
        )
    ids.append(int(eos_id))
    return ids


def _feature(prompt_ids: Sequence[int], completion_ids: Sequence[int]) -> dict[str, list[int]]:
    input_ids = list(prompt_ids) + list(completion_ids)
    labels = [-100] * len(prompt_ids) + list(completion_ids)
    response_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def sft_feature(
    tokenizer: Any,
    row: dict[str, Any],
    max_prompt_length: int = 320,
    max_completion_length: int = 64,
) -> dict[str, list[int]]:
    prompt_ids = prompt_token_ids(tokenizer, str(row["prompt"]), max_prompt_length, row.get("id"))
    completion_ids = _completion_ids(tokenizer, target_text(row), max_completion_length, row.get("id"))
    return _feature(prompt_ids, completion_ids)


def preference_features(
    tokenizer: Any,
    row: dict[str, Any],
    max_prompt_length: int = 320,
    max_completion_length: int = 64,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    if "chosen" not in row or "rejected" not in row:
        raise ValueError(f"preference row {row.get('id', '<unknown>')} lacks chosen/rejected")
    prompt_ids = prompt_token_ids(tokenizer, str(row["prompt"]), max_prompt_length, row.get("id"))
    chosen = _feature(prompt_ids, _completion_ids(tokenizer, row["chosen"], max_completion_length, row.get("id")))
    rejected = _feature(prompt_ids, _completion_ids(tokenizer, row["rejected"], max_completion_length, row.get("id")))
    return chosen, rejected


def generated_feature(prompt_ids: Sequence[int], completion_ids: Sequence[int]) -> dict[str, list[int]]:
    """Build a likelihood feature from sampled token ids; EOS remains scored."""
    return _feature(prompt_ids, completion_ids)


def collate_features(features: Sequence[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    if not features:
        raise ValueError("cannot collate an empty feature batch")
    width = max(len(item["input_ids"]) for item in features)
    ids, labels, response_masks, attention = [], [], [], []
    for item in features:
        n = len(item["input_ids"])
        padding = width - n
        ids.append(item["input_ids"] + [int(pad_token_id)] * padding)
        labels.append(item["labels"] + [-100] * padding)
        response_masks.append(item["response_mask"] + [0] * padding)
        attention.append([1] * n + [0] * padding)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "response_mask": torch.tensor(response_masks, dtype=torch.bool),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


def token_logps_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    sequence_chunk_size: int = 32,
) -> torch.Tensor:
    """Return shifted token log-probabilities, zeroing prompt and padding.

    EOS is included when its response-mask bit is set. Right padding and every
    token after EOS are excluded. Chunking limits the temporary float32
    log-softmax tensor for a large vocabulary.
    """
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("expected logits [batch, length, vocab] and input_ids [batch, length]")
    if logits.shape[:2] != input_ids.shape or response_mask.shape != input_ids.shape:
        raise ValueError("logits, input_ids, and response_mask sequence dimensions must match")
    if input_ids.shape[1] < 2:
        return logits.new_zeros((input_ids.shape[0], 0), dtype=torch.float32)
    target = input_ids[:, 1:]
    valid = response_mask[:, 1:].bool()
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask shape must match input_ids")
        valid = valid & attention_mask[:, 1:].bool()
    pieces: list[torch.Tensor] = []
    for start in range(0, target.shape[1], max(1, sequence_chunk_size)):
        end = min(start + max(1, sequence_chunk_size), target.shape[1])
        chunk_logits = logits[:, start:end, :].float()
        chunk_target = target[:, start:end]
        chunk_valid = valid[:, start:end]
        selected = F.log_softmax(chunk_logits, dim=-1).gather(-1, chunk_target.unsqueeze(-1)).squeeze(-1)
        pieces.append(selected * chunk_valid.to(selected.dtype))
    return torch.cat(pieces, dim=1)


def sequence_logps(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_logps = token_logps_from_logits(logits, input_ids, response_mask, attention_mask)
    return token_logps.sum(dim=-1), token_logps


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sequence-sum DPO logistic loss with a fixed reference policy."""
    margin = (policy_chosen_logps - policy_rejected_logps) - (
        reference_chosen_logps - reference_rejected_logps
    )
    scaled = float(beta) * margin
    loss = -F.logsigmoid(scaled).mean()
    return loss, {"preference_margin": margin.detach(), "chosen_reward": (beta * (policy_chosen_logps - reference_chosen_logps)).detach(), "rejected_reward": (beta * (policy_rejected_logps - reference_rejected_logps)).detach()}


def group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-4) -> torch.Tensor:
    """Center and population-standardize rewards independently per prompt."""
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    if rewards.numel() % group_size:
        raise ValueError("reward count must be divisible by group_size")
    grouped = rewards.reshape(-1, group_size).float()
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, unbiased=False, keepdim=True)
    advantages = centered / (std + float(eps))
    return advantages.reshape(-1)


def grpo_loss(
    current_token_logps: torch.Tensor,
    old_token_logps: torch.Tensor,
    reference_token_logps: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
    beta: float = 0.02,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """GRPO clipped surrogate plus sampled reverse-KL estimate.

    Token losses are averaged over valid response tokens per completion, then
    averaged over completions. Old/reference values are treated as constants.
    """
    if current_token_logps.shape != old_token_logps.shape or current_token_logps.shape != reference_token_logps.shape:
        raise ValueError("current, old, and reference token logps must have identical shapes")
    if response_mask.shape != current_token_logps.shape:
        raise ValueError("response_mask must match token logps")
    if advantages.ndim != 1 or advantages.shape[0] != current_token_logps.shape[0]:
        raise ValueError("one scalar advantage is required per completion")
    mask = response_mask.to(dtype=current_token_logps.dtype)
    lengths = mask.sum(dim=-1).clamp_min(1.0)
    old = old_token_logps.detach()
    reference = reference_token_logps.detach()
    ratio = torch.exp(current_token_logps - old)
    adv = advantages.to(current_token_logps.dtype).unsqueeze(-1)
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(ratio * adv, clipped * adv)
    per_response_policy = -(surrogate * mask).sum(dim=-1) / lengths
    delta = reference - current_token_logps
    kl_tokens = torch.exp(delta) - delta - 1.0
    per_response_kl = (kl_tokens * mask).sum(dim=-1) / lengths
    policy_loss = per_response_policy.mean()
    kl_mean = per_response_kl.mean()
    loss = policy_loss + float(beta) * kl_mean
    clip_fraction = (((ratio - clipped).abs() > 1e-8).to(mask.dtype) * mask).sum() / mask.sum().clamp_min(1.0)
    return loss, {
        "policy_loss": policy_loss.detach(),
        "kl": kl_mean.detach(),
        "clip_fraction": clip_fraction.detach(),
        "mean_ratio": ((ratio * mask).sum() / mask.sum().clamp_min(1.0)).detach(),
    }


def load_tokenizer(model_name_or_path: str | None = None) -> Any:
    from transformers import AutoTokenizer

    source = model_name_or_path or MODEL_ID
    kwargs = {} if model_name_or_path else {"revision": MODEL_REVISION}
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define either pad_token_id or eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(model_name_or_path: str | None = None, device: str | None = None) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    source = model_name_or_path or MODEL_ID
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    if not model_name_or_path:
        kwargs["revision"] = MODEL_REVISION
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    model.config.use_cache = False
    if device:
        model.to(device)
    return model


def load_sft_model(model_name_or_path: str | None, device: str, lora_rank: int = 8, lora_alpha: int = 16) -> torch.nn.Module:
    from peft import LoraConfig, TaskType, get_peft_model

    model = load_base_model(model_name_or_path, device)
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, config)
    prepare_for_updates(model)
    activate_train_adapter(model, "default")
    return model


def load_policy_reference_model(
    model_name_or_path: str | None,
    adapter_path: str,
    device: str,
) -> torch.nn.Module:
    """Load policy and a frozen, identical-to-SFT reference over one base."""
    from peft import PeftModel

    base = load_base_model(model_name_or_path, device)
    model = PeftModel.from_pretrained(base, adapter_path, adapter_name="policy", is_trainable=True)
    model.load_adapter(adapter_path, adapter_name="reference", is_trainable=False)
    prepare_for_updates(model)
    activate_train_adapter(model, "policy")
    return model


def prepare_for_updates(model: torch.nn.Module) -> None:
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def _adapter_param_matches(name: str, adapter_name: str) -> bool:
    return "lora_" in name and f".{adapter_name}." in f".{name}."


def activate_train_adapter(model: torch.nn.Module, adapter_name: str) -> None:
    if hasattr(model, "set_adapter"):
        model.set_adapter(adapter_name)
    matched = 0
    for name, parameter in model.named_parameters():
        trainable = _adapter_param_matches(name, adapter_name)
        parameter.requires_grad_(trainable)
        matched += int(trainable)
    if matched == 0:
        raise RuntimeError(f"no trainable LoRA parameters found for adapter {adapter_name!r}")


def set_adapter_for_inference(model: torch.nn.Module, adapter_name: str | None = None) -> None:
    if adapter_name and hasattr(model, "set_adapter"):
        model.set_adapter(adapter_name)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()


@contextlib.contextmanager
def frozen_adapter(model: torch.nn.Module, adapter_name: str, restore_adapter: str = "policy") -> Iterator[None]:
    original = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    model.set_adapter(adapter_name)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        model.set_adapter(restore_adapter)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original[name])


def adapter_parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}


def model_logits(model: torch.nn.Module, batch: dict[str, torch.Tensor], device: torch.device | str) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    attention = batch["attention_mask"].to(device)
    return model(input_ids=input_ids, attention_mask=attention, use_cache=False, return_dict=True).logits


def batch_sequence_logps(model: torch.nn.Module, batch: dict[str, torch.Tensor], device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model_logits(model, batch, device)
    ids = batch["input_ids"].to(device)
    response_mask = batch["response_mask"].to(device)
    attention = batch["attention_mask"].to(device)
    return sequence_logps(logits, ids, response_mask, attention)


def generate_candidates(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device | str,
    count: int = 4,
    max_prompt_length: int = 320,
    max_new_tokens: int = 64,
    adapter_name: str | None = None,
    row_id: Any = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Sample independent candidates from the policy using full-support top-p=1."""
    if adapter_name and hasattr(model, "set_adapter"):
        model.set_adapter(adapter_name)
    model.eval()
    prompt_ids = prompt_token_ids(tokenizer, prompt, max_prompt_length, row_id)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            num_beams=1,
            num_return_sequences=count,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    candidates: list[dict[str, Any]] = []
    for sequence in generated:
        new_ids = sequence[len(prompt_ids):].tolist()
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and eos_id in new_ids:
            end = new_ids.index(eos_id)
            completion_ids = [int(x) for x in new_ids[: end + 1]]
            truncated = False
            text_ids = new_ids[:end]
        else:
            completion_ids = [int(x) for x in new_ids]
            truncated = len(completion_ids) >= max_new_tokens
            text_ids = completion_ids
        text = tokenizer.decode(text_ids, skip_special_tokens=True).strip()
        candidates.append({
            "text": text,
            "completion_ids": completion_ids,
            "generated_tokens": len(completion_ids),
            "truncated": bool(truncated),
        })
    return candidates, prompt_ids
