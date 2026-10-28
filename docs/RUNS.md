# Locked run protocol

## Fixed setup

- Base: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`; BF16; LoRA rank 8, alpha 16, `q_proj`/`v_proj`.
- Train prompt/completion limits: 256/64 tokens; generation limit: 64; training batch size: 1.
- SFT: seed 17 reuses the selected 16-update checkpoint from `results/pilot_sft16_seed17`, now `results/sft_seed17`; seed 29 trains 16 updates with gradient accumulation 4 and learning rate `2e-4`.
- Offline DPO: 64 updates, gradient accumulation 4, beta `0.1`, `data/preferences_train.jsonl`.
- OnlineDPO: 64 maximum updates, gradient accumulation 2, full reward, four candidates, prompt budget 128, beta `0.1`. Budget exhaustion is a normal stop; summaries retain the actual update and prompt counts.
- GRPO: 128 optimizer updates total, gradient accumulation 2, four candidates, and two policy updates per rollout batch. This yields 128 prompt groups and 512 sampled completions per run; population-standardized group advantages, clip epsilon `0.2`, beta `0.02`.
- `grpo_format` uses the same GRPO setup and the format-only reward.
- Each method branch starts from its own seed's SFT adapter as both policy initialization and fixed reference. Branches do not initialize from one another.
- Evaluation uses greedy decoding, batch size 8, and validation plus `test_id`, `test_ood`, and `test_counterfactual`. Test splits are reporting-only and do not change this protocol. Base, selected seed17 SFT, and long seed17 SFT validation outputs already exist and are reused.

## Start or resume the suite

Run from the repository root with the prepared environment:

```powershell
.\.venv\Scripts\python.exe -m src.run_suite
```

The runner executes commands serially on the selected GPU, writes a command transcript to `results/suite_seed17_seed29/run_suite.log`, and skips an output only after checking its run configuration and input/output paths, completion summary, adapter files, log rows, finite gradients, and memory peak. For OnlineDPO and GRPO it also verifies rollout-group counts, candidate counts, training split and reward mode, and generated-token totals. An existing partial or incompatible output stops the suite without overwriting it. Successful but incomplete outputs are treated as errors. The runner's exit status is nonzero on the first failure.

## Training commands issued by the runner

The official seed17 SFT checkpoint is reused and is not retrained. The runner trains seed29 SFT with:

```powershell
.\.venv\Scripts\python.exe -m src.train sft --seed 29 --steps 16 --batch-size 1 --grad-accum 4 --learning-rate 0.0002 --output results/sft_seed29
```

For each seed (`17`, then `29`), the four branches are launched from that seed's SFT adapter:

```powershell
.\.venv\Scripts\python.exe -m src.train dpo --seed <seed> --steps 64 --batch-size 1 --grad-accum 4 --adapter results/sft_seed<seed>/adapter --output results/dpo_seed<seed>
.\.venv\Scripts\python.exe -m src.train online_dpo --seed <seed> --steps 64 --batch-size 1 --grad-accum 2 --max-rollout-prompts 128 --reward-mode full --adapter results/sft_seed<seed>/adapter --output results/online_dpo_seed<seed>
.\.venv\Scripts\python.exe -m src.train grpo --seed <seed> --steps 128 --batch-size 1 --grad-accum 2 --update-epochs 2 --reward-mode full --adapter results/sft_seed<seed>/adapter --output results/grpo_seed<seed>
.\.venv\Scripts\python.exe -m src.train grpo --seed <seed> --steps 128 --batch-size 1 --grad-accum 2 --update-epochs 2 --reward-mode format --adapter results/sft_seed<seed>/adapter --output results/grpo_format_seed<seed>
```

Each model is evaluated on all four splits with this command shape; `<model>` is `base`, `sft_long_seed17`, `sft_seed17`, `sft_seed29`, or a branch name with its seed suffix:

```powershell
.\.venv\Scripts\python.exe -m src.train evaluate --adapter <adapter-path-or-none> --input data/<split>.jsonl --output results/eval/<model>/<split> --batch-size 8
```

`src.run_suite` records the fully expanded commands and verifies each output before advancing. Each training directory retains `run_config.json`, `step_log.jsonl`, `run_summary.json`, its adapter, and rollout audit JSONL for online methods. Each evaluation directory retains its summary and per-example `predictions.jsonl`.

## Runtime environment note

During the serial GPU suite, Unity editor GPU processes remained active alongside the training and evaluation process. They were left untouched. The reported per-process CUDA allocated/reserved peaks come from PyTorch run summaries; system-wide GPU utilization and wall-clock durations reflect this shared desktop session and are not exclusive-GPU performance benchmarks. Runtime comparisons should account for this concurrent graphics workload.

Across the 12 persisted training summaries, the maximum PyTorch allocated peak was 1.4756 GiB (1,584,440,320 bytes) for `online_dpo_seed29`; the maximum reserved peak was 2.2344 GiB (2,399,141,888 bytes), reported by `sft_seed17`, `sft_long_seed17`, and retained `pilot_sft32_seed17`. Considering the eight preference-optimization branches alone, the maximum allocated and reserved peaks were 1.4756 GiB (`online_dpo_seed29`) and 2.2148 GiB (`online_dpo_seed17`), respectively. These are process-level peaks, not whole-GPU occupancy.
