# 训练与评估实现

本目录提供同一 Qwen2.5-0.5B-Instruct 底座上的 LoRA SFT、离线 DPO、OnlineDPO、GRPO 和逐样本评估。基础权重固定为 `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775`。每个训练分支使用 BF16 底座和 rank 8、alpha 16、`q_proj`/`v_proj` LoRA；微批大小默认为 1，梯度累积默认为 1，上下文上限 256，completion 上限 64。更新阶段使用 `use_cache=False` 和 `use_reentrant=False` 梯度检查点，采样阶段用 `use_cache=True`。训练 forward 处于 train mode 以启用 Transformers 的梯度检查点；模型 attention dropout 与 LoRA dropout 均设为 0，所以策略 log-prob、old log-prob 与 reference log-prob 不受 dropout 噪声影响。每个更新在 optimizer step 前执行 max-norm 1.0 梯度裁剪；step 日志记录梯度范数是否有限，非有限梯度会阻止该次参数更新。

输入先通过 tokenizer 自带的 chat template 构造 user prompt 和 assistant generation prefix。SFT 序列由该前缀、规范 JSON target 和 EOS 组成；损失只计算 completion token（包括 EOS），prompt 和 padding 标签设为 `-100`。若 prompt 或目标超过配置上限，训练立即报错并报告样本 ID；不会静默截断指令或标签。

离线 DPO 读取 `chosen`/`rejected` 成对偏好，policy 与 reference adapter 分别从同一 SFT adapter 初始化，固定 reference 与可训练 policy 共用单个底座。双方只在 completion token 上计算 log-prob，并沿每条序列求和。目标为标准 logistic DPO，默认 `beta=0.1`。

OnlineDPO 对训练 prompt 采样 4 个 completion，解码分布明确设为 temperature 1、top-p 1、top-k 0、repetition penalty 1，不施加额外截断采样或重复惩罚。`full` reward 使用验证器的格式、答案及联合证据奖励；`format` 消融只使用规范 JSON 格式奖励。最大和最小 reward 对组成 chosen/rejected，四个 reward 相同时记录并跳过该组。训练不向模型输入 gold answer 或 evidence。

GRPO 对每个训练 prompt 采样 4 个 completion，以组内 reward 均值和 population standard deviation 标准化优势，标准差加 `1e-4`。rollout 时冻结并记录 old-policy 与 SFT-reference 的 token log-prob；更新目标由 `epsilon=0.2` clipped surrogate 与 `beta=0.02` 的采样 reverse-KL 估计组成。每条 response 先按有效 completion token 求平均，再对 responses 求平均。默认对每个 rollout 组进行两次顺序 optimizer update，第二次复用旧 rollout 与 old log-prob，因此 clipping ratio 可以偏离 1。EOS 计入有效 token；EOS 后 padding 排除；到达最大生成长度未生成 EOS 的 completion 标记为 truncated。

评估采用 greedy decoding 和固定 64 个新 token 上限；按左侧 padding 与 attention mask 批量推理，再逐样本写出预测、评分和截断标记，最后汇总格式有效率、答案正确率、证据 exact match、证据 F1 和联合正确率。评估默认批量为 8，可用 `--batch-size` 调整。汇总保存模型/adapter 配置、评估墙钟时间与 CUDA 峰值显存。训练命令拒绝任何不属于 `train*` 的行；评估命令拒绝训练行。请使用 `validation.jsonl` 或三个 test 文件做评估，不将其作为训练输入。

## 命令

在作业根目录执行；下列 Windows 命令使用项目虚拟环境。先训练一个最小 SFT 更新，再从该 adapter 分别启动各偏好/RL 分支：

```powershell
.\.venv\Scripts\python.exe -m src.train sft --steps 1 --batch-size 1 --grad-accum 1
.\.venv\Scripts\python.exe -m src.train dpo --steps 1 --batch-size 1 --grad-accum 1
.\.venv\Scripts\python.exe -m src.train online_dpo --steps 1 --batch-size 1 --grad-accum 1 --max-rollout-prompts 1
.\.venv\Scripts\python.exe -m src.train grpo --steps 2 --batch-size 1 --grad-accum 1
.\.venv\Scripts\python.exe -m src.train evaluate --input data/test_id.jsonl --output results/evaluate_test_id --batch-size 8
```

GRPO 的 `--steps 2` 与默认两次 update epoch 对应一个 rollout 组的两次 optimizer update。默认 OnlineDPO 设置为至多采样 128 个 prompt、64 次 optimizer update、每次 2 个 prompt；`--max-rollout-prompts` 限制总 prompt 数，预算用完后会保存部分结果。正式配置中 GRPO 使用 128 次 optimizer update、每次 2 个新 prompt，并以两轮更新复用 rollout；因此与 OnlineDPO 都最多生成 128 组、每组 4 个 completion。完整训练步数、学习率、输入路径、适配器路径和方法参数集中在 `configs/experiment.json`；命令行参数可覆盖常用值。若已将模型下载到本地目录，可用 `--model-path <本地模型目录>` 替换固定 HF id。评估基础模型时设置 `--adapter none`；默认评估 SFT adapter。

每个训练输出目录保存 `run_config.json`、逐 optimizer update 的 `step_log.jsonl`、`run_summary.json` 和 LoRA adapter。SFT adapter 位于 `adapter/`；偏好/RL 分支 adapter 位于 `adapter/policy/`，实际可加载路径也记录在 `run_summary.json`。OnlineDPO/GRPO 另保存逐 prompt 的 `rollouts.jsonl`，包含候选文本、分数、reward、截断状态、偏好选择或 old/reference log-prob 与优势。汇总中记录总墙钟时间、生成 token、更新 token 和 CUDA 峰值 allocated/reserved；CPU 执行时 CUDA 计数为 0。评估保存 `predictions.jsonl` 与 `evaluation_summary.json`。
