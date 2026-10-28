"""Serial, resumable execution of the locked post-training experiment suite.

Run from the repository root with the project's Python environment:
    .venv/Scripts/python.exe -m src.run_suite

Completed outputs are validated and skipped. Existing partial or incompatible
outputs stop the suite so a failed run is never mistaken for a completed one.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EVAL_ROOT = RESULTS / "eval"
RUN_LOG_DIR = RESULTS / "suite_seed17_seed29"
RUN_LOG = RUN_LOG_DIR / "run_suite.log"
SPLITS = ("validation", "test_id", "test_ood", "test_counterfactual")
TEST_SPLITS = ("test_id", "test_ood", "test_counterfactual")
METHODS = ("dpo", "online_dpo", "grpo", "grpo_format")
MAX_CUDA_BYTES = 6 * 1024**3


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read valid JSON metadata at {relative(path)}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object at {relative(path)}")
    return value


def jsonl_count(path: Path) -> int:
    if not path.is_file():
        raise RuntimeError(f"missing JSONL output: {relative(path)}")
    with path.open("r", encoding="utf-8-sig") as stream:
        return sum(1 for line in stream if line.strip())


def normalized(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return os.path.normcase(str(resolved.resolve()))


class Suite:
    def __init__(self) -> None:
        RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.done_models = 0

    def record(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with RUN_LOG.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()

    def command(self, args: list[str], label: str) -> None:
        display = subprocess.list2cmdline(args) if os.name == "nt" else " ".join(args)
        self.record(f"START {label}: {display}")
        start = time.monotonic()
        with RUN_LOG.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                args,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        elapsed = time.monotonic() - start
        if code != 0:
            self.record(f"FAIL {label}: exit={code}, wall={elapsed:.1f}s")
            raise RuntimeError(f"{label} failed with exit code {code}; see {relative(RUN_LOG)}")
        self.record(f"EXIT {label}: 0, wall={elapsed:.1f}s")

    def training_complete(
        self,
        output: Path,
        method: str,
        settings: dict[str, Any],
    ) -> str | None:
        if not output.exists():
            return None
        if not output.is_dir():
            raise RuntimeError(f"training output exists but is not a directory: {relative(output)}")
        config_path = output / "run_config.json"
        summary_path = output / "run_summary.json"
        log_path = output / "step_log.jsonl"
        if not (config_path.is_file() and summary_path.is_file() and log_path.is_file()):
            raise RuntimeError(
                f"incomplete training output exists at {relative(output)}; inspect it and choose an explicit recovery"
            )
        config, summary = read_json(config_path), read_json(summary_path)
        saved = config.get("settings", {})
        for key in ("method",):
            if config.get(key) != method:
                raise RuntimeError(f"{relative(output)} belongs to method={config.get(key)!r}, expected {method!r}")
        for key in ("seed", "steps", "batch_size", "grad_accum", "learning_rate"):
            if key in settings and saved.get(key) != settings[key]:
                raise RuntimeError(f"{relative(output)} has incompatible {key}={saved.get(key)!r}, expected {settings[key]!r}")
        for key in ("reward_mode", "max_rollout_prompts", "update_epochs", "group_size", "beta", "clip_epsilon"):
            if key in settings and saved.get(key) != settings[key]:
                raise RuntimeError(f"{relative(output)} has incompatible {key}={saved.get(key)!r}, expected {settings[key]!r}")
        for key in ("input", "output"):
            if key in settings and normalized(config.get(key, "")) != normalized(settings[key]):
                raise RuntimeError(f"{relative(output)} has incompatible {key}={config.get(key)!r}, expected {settings[key]!r}")
        if "adapter" in settings and normalized(config.get("adapter_input", "")) != normalized(settings["adapter"]):
            raise RuntimeError(f"{relative(output)} was trained from a different SFT adapter")
        adapter_value = summary.get("adapter_path")
        if not adapter_value:
            raise RuntimeError(f"{relative(output)} lacks the saved adapter_path")
        adapter = Path(adapter_value)
        if not adapter.is_absolute():
            adapter = ROOT / adapter
        if not (adapter / "adapter_config.json").is_file() or not (adapter / "adapter_model.safetensors").is_file():
            raise RuntimeError(f"saved adapter files are incomplete: {adapter}")
        try:
            step_rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid step log {relative(log_path)}: {exc}") from exc
        updates = int(summary.get("updates", -1))
        successful = [row for row in step_rows if row.get("updated", True)]
        if len(successful) != updates:
            raise RuntimeError(f"{relative(output)} records {updates} updates but {len(successful)} successful step rows")
        for row in successful:
            if row.get("gradient_finite") is not True:
                raise RuntimeError(f"non-finite or unrecorded gradient in {relative(log_path)} step {row.get('step')}")
            for key in ("loss", "gradient_norm"):
                if key in row and row[key] is not None and not math.isfinite(float(row[key])):
                    raise RuntimeError(f"non-finite {key} in {relative(log_path)} step {row.get('step')}")
        if method == "grpo" and successful:
            ratio = float(successful[0].get("mean_ratio", math.nan))
            if not math.isfinite(ratio) or abs(ratio - 1.0) > 0.01:
                raise RuntimeError(f"initial GRPO old-policy ratio is not near 1 in {relative(log_path)}: {ratio}")
        peak = max(
            int(summary.get("peak_cuda_allocated_bytes", 0)),
            int(summary.get("peak_cuda_reserved_bytes", 0)),
        )
        if peak > MAX_CUDA_BYTES:
            raise RuntimeError(f"reported CUDA peak exceeds 6 GiB at {relative(output)}")
        if method != "online_dpo" and updates != settings["steps"]:
            raise RuntimeError(f"{relative(output)} completed {updates}, expected {settings['steps']} updates")
        if method == "online_dpo" and updates < settings["steps"] and not summary.get("rollout_budget_exhausted"):
            raise RuntimeError(f"{relative(output)} ended before steps without exhausting the prompt budget")
        if method in {"online_dpo", "grpo"}:
            rollout_path = output / "rollouts.jsonl"
            if not rollout_path.is_file():
                raise RuntimeError(f"missing rollout audit log: {relative(rollout_path)}")
            try:
                rollout_rows = [
                    json.loads(line)
                    for line in rollout_path.read_text(encoding="utf-8-sig").splitlines()
                    if line.strip()
                ]
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid rollout audit log {relative(rollout_path)}: {exc}") from exc
            rollout_prompts = int(summary.get("rollout_prompts", -1))
            if rollout_prompts != len(rollout_rows):
                raise RuntimeError(
                    f"{relative(rollout_path)} has {len(rollout_rows)} groups, summary records {rollout_prompts}"
                )
            group_size = int(settings.get("group_size", 4))
            if any(len(row.get("candidates", [])) != group_size for row in rollout_rows):
                raise RuntimeError(f"{relative(rollout_path)} contains a group with the wrong candidate count")
            if any(row.get("split") != "train_pref" for row in rollout_rows):
                raise RuntimeError(f"{relative(rollout_path)} contains a non-training rollout")
            if any(row.get("reward_mode") != settings.get("reward_mode", "full") for row in rollout_rows):
                raise RuntimeError(f"{relative(rollout_path)} contains an unexpected reward mode")
            completion_tokens = sum(
                int(candidate.get("generated_tokens", -1))
                for row in rollout_rows
                for candidate in row["candidates"]
            )
            if completion_tokens != int(summary.get("generated_tokens", -1)):
                raise RuntimeError(
                    f"{relative(rollout_path)} tokens sum to {completion_tokens}, "
                    f"summary records {summary.get('generated_tokens')}"
                )
            if method == "online_dpo":
                expected_groups = settings.get("max_rollout_prompts")
            else:
                steps_per_rollout = int(settings["batch_size"]) * int(settings["grad_accum"])
                epochs = int(settings["update_epochs"])
                if int(settings["steps"]) % epochs:
                    raise RuntimeError("GRPO updates must divide evenly across rollout reuse epochs")
                expected_groups = int(settings["steps"]) // epochs * steps_per_rollout
            if expected_groups is not None and rollout_prompts != int(expected_groups):
                raise RuntimeError(
                    f"{relative(output)} used {rollout_prompts} rollout prompts, expected {expected_groups}"
                )
        return str(adapter)

    def training_run(self, args: list[str], output: Path, method: str, settings: dict[str, Any]) -> str:
        adapter = self.training_complete(output, method, settings)
        if adapter:
            self.record(f"SKIP complete training {relative(output)}; adapter={adapter}")
            return adapter
        if output.exists():
            raise RuntimeError(f"refusing to overwrite nonempty output {relative(output)}")
        self.command([sys.executable, "-m", "src.train", *args], f"train {relative(output)}")
        adapter = self.training_complete(output, method, settings)
        if not adapter:
            raise RuntimeError(f"training exited successfully but did not create a complete output: {relative(output)}")
        return adapter

    def evaluation_complete(
        self,
        output: Path,
        input_path: Path,
        expected_examples: int,
        adapter_path: str | None,
    ) -> bool:
        if not output.exists():
            return False
        if not output.is_dir():
            raise RuntimeError(f"evaluation output exists but is not a directory: {relative(output)}")
        summary_path = output / "evaluation_summary.json"
        predictions_path = output / "predictions.jsonl"
        if not summary_path.is_file() or not predictions_path.is_file():
            raise RuntimeError(f"incomplete evaluation output exists at {relative(output)}; refusing overwrite")
        summary = read_json(summary_path)
        if int(summary.get("examples", -1)) != expected_examples or jsonl_count(predictions_path) != expected_examples:
            raise RuntimeError(f"evaluation row count mismatch in {relative(output)}")
        if normalized(summary.get("input", "")) != normalized(input_path):
            raise RuntimeError(f"evaluation input mismatch in {relative(output)}")
        stored_adapter = summary.get("adapter_path")
        if adapter_path is None:
            if stored_adapter is not None:
                raise RuntimeError(f"expected base model evaluation, found adapter at {stored_adapter}")
        elif normalized(stored_adapter or "") != normalized(adapter_path):
            raise RuntimeError(f"evaluation adapter mismatch in {relative(output)}")
        peak = max(
            int(summary.get("peak_cuda_allocated_bytes", 0)),
            int(summary.get("peak_cuda_reserved_bytes", 0)),
        )
        if peak > MAX_CUDA_BYTES:
            raise RuntimeError(f"reported evaluation CUDA peak exceeds 6 GiB at {relative(output)}")
        return True

    def evaluate_one(self, model_name: str, adapter_path: str | None, split: str) -> dict[str, Any]:
        input_path = ROOT / "data" / f"{split}.jsonl"
        count = sum(1 for line in input_path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
        output = EVAL_ROOT / model_name / split
        if self.evaluation_complete(output, input_path, count, adapter_path):
            self.record(f"SKIP complete evaluation {relative(output)}")
            return read_json(output / "evaluation_summary.json")
        if output.exists():
            raise RuntimeError(f"refusing to overwrite incomplete evaluation at {relative(output)}")
        args = [
            "evaluate",
            "--input", relative(input_path),
            "--output", relative(output),
            "--batch-size", "8",
        ]
        if adapter_path is None:
            args.extend(("--adapter", "none"))
        else:
            args.extend(("--adapter", str(Path(adapter_path).resolve())))
        self.command([sys.executable, "-m", "src.train", *args], f"evaluate {model_name}/{split}")
        if not self.evaluation_complete(output, input_path, count, adapter_path):
            raise RuntimeError(f"evaluation exited successfully but output is incomplete: {relative(output)}")
        return read_json(output / "evaluation_summary.json")

    def evaluate_model(self, model_name: str, adapter_path: str | None) -> None:
        summaries = {split: self.evaluate_one(model_name, adapter_path, split) for split in SPLITS}
        per_split = {
            split: {
                key: round(float(value), 6)
                for key, value in summary["mean_metrics"].items()
            }
            for split, summary in summaries.items()
        }
        self.done_models += 1
        self.record(f"MODEL COMPLETE {model_name}: {json.dumps(per_split, ensure_ascii=False, sort_keys=True)}")

    def train_seed_sft(self, seed: int) -> str:
        output = RESULTS / f"sft_seed{seed}"
        settings = {
            "seed": seed, "steps": 16, "batch_size": 1, "grad_accum": 4,
            "learning_rate": 0.0002, "input": "data/train_sft.jsonl", "output": relative(output),
        }
        args = [
            "sft", "--seed", str(seed), "--steps", "16", "--batch-size", "1",
            "--grad-accum", "4", "--learning-rate", "0.0002", "--output", relative(output),
        ]
        return self.training_run(args, output, "sft", settings)

    def train_branch(self, method: str, seed: int, sft_adapter: str) -> str:
        output = RESULTS / f"{method}_seed{seed}"
        method_arg = "grpo" if method == "grpo_format" else method
        settings_by_method: dict[str, dict[str, Any]] = {
            "dpo": {
                "seed": seed, "steps": 64, "batch_size": 1, "grad_accum": 4,
                "learning_rate": 0.00005, "input": "data/preferences_train.jsonl",
            },
            "online_dpo": {
                "seed": seed, "steps": 64, "batch_size": 1, "grad_accum": 2,
                "learning_rate": 0.00005, "reward_mode": "full", "max_rollout_prompts": 128,
                "group_size": 4, "input": "data/train_pref.jsonl",
            },
            "grpo": {
                "seed": seed, "steps": 128, "batch_size": 1, "grad_accum": 2,
                "learning_rate": 0.00001, "reward_mode": "full", "group_size": 4,
                "update_epochs": 2, "beta": 0.02, "clip_epsilon": 0.2, "input": "data/train_pref.jsonl",
            },
            "grpo_format": {
                "seed": seed, "steps": 128, "batch_size": 1, "grad_accum": 2,
                "learning_rate": 0.00001, "reward_mode": "format", "group_size": 4,
                "update_epochs": 2, "beta": 0.02, "clip_epsilon": 0.2, "input": "data/train_pref.jsonl",
            },
        }
        settings = settings_by_method[method]
        settings["adapter"] = sft_adapter
        settings["output"] = relative(output)
        args = [
            method_arg, "--seed", str(seed), "--batch-size", "1", "--adapter", sft_adapter,
            "--output", relative(output),
        ]
        if method == "dpo":
            args.extend(("--steps", "64", "--grad-accum", "4", "--learning-rate", "0.00005"))
        elif method == "online_dpo":
            args.extend((
                "--steps", "64", "--grad-accum", "2", "--max-rollout-prompts", "128",
                "--learning-rate", "0.00005", "--reward-mode", "full",
            ))
        else:
            reward_mode = "format" if method == "grpo_format" else "full"
            args.extend((
                "--steps", "128", "--grad-accum", "2", "--update-epochs", "2",
                "--learning-rate", "0.00001", "--reward-mode", reward_mode,
            ))
        return self.training_run(args, output, method_arg, settings)

    def run(self) -> None:
        self.record("SUITE LOCKED: seed17/29, SFT16, DPO64/accum4, OnlineDPO budget128; GRPO has 128 optimizer updates total with each rollout batch reused for two policy updates. Validation and test splits are reporting-only.")

        # Existing validation artifacts are verified and skipped; no test result is used for decisions.
        self.evaluate_model("base", None)
        long_adapter = str(RESULTS / "sft_long_seed17" / "adapter")
        if not (Path(long_adapter) / "adapter_config.json").is_file():
            raise RuntimeError("formal long SFT adapter is missing")
        self.evaluate_model("sft_long_seed17", long_adapter)

        sft17_adapter = str(RESULTS / "sft_seed17" / "adapter")
        if not (Path(sft17_adapter) / "adapter_config.json").is_file():
            raise RuntimeError("selected seed17 SFT16 adapter is missing")
        self.evaluate_model("sft_seed17", sft17_adapter)

        sft29_adapter = self.train_seed_sft(29)
        self.evaluate_model("sft_seed29", sft29_adapter)

        for seed, adapter in ((17, sft17_adapter), (29, sft29_adapter)):
            for method in METHODS:
                trained_adapter = self.train_branch(method, seed, adapter)
                self.evaluate_model(f"{method}_seed{seed}", trained_adapter)

        elapsed = time.monotonic() - self.started
        self.record(f"SUITE COMPLETE models={self.done_models}, wall={elapsed:.1f}s")


def main() -> int:
    if Path.cwd().resolve() != ROOT.resolve():
        print(f"Run from repository root: {ROOT}", file=sys.stderr)
        return 2
    try:
        Suite().run()
    except Exception as exc:
        print(f"RUN SUITE STOPPED: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
