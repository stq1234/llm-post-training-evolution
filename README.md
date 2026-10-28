# 大语言模型后训练代码分析与实验

项目资料与实验代码仓库：<https://github.com/stq1234/llm-post-training-evolution>

本作业分支已归档（当前提交 `60d18bb`）：[https://github.com/stq1234/llm-post-training-evolution/tree/main](https://github.com/stq1234/llm-post-training-evolution/tree/main)

本仓库归档后训练代码分析与可复现实验：包括合成中文证据问答数据、PyTorch LoRA 训练/评估、结果分析和本作业实验报告。综述与 DPO 思考题各自保存在课程根目录下的独立作业文件夹，不在本仓库复制。实验实现统一使用 `Qwen/Qwen2.5-0.5B-Instruct`，固定 revision `7ae557604adf67be50417f59c2c2f167def9a775`。

## 目录

| 路径 | 内容 |
|---|---|
| `实验报告.md` | 本仓库对应的源码分析与真实实验报告 |
| `rendered/` | 实验报告本地预览与 Piazza HTML、纯文本及 manifest |
| `docs/SUBMISSION.md` | 三个 HW5 作业的目录分离、Piazza 状态与链接记录 |
| `research/code_analysis.md` | 8 个公开代码仓库、10 个版本快照的实现分析 |
| `research/code_sources.json` | 源码仓库、提交和调用路径索引 |
| `research/code_excerpts/` | 可核对的实现证据摘录 |
| `research/experiment_methods.md` | 实验方法与比较依据 |
| `data/` | 合成数据、固定划分、偏好对及生成清单 |
| `configs/experiment.json` | 模型、训练及评估默认设置 |
| `src/data.py` | 合成数据生成器 |
| `src/modeling.py` | tokenizer、LoRA、策略与 reference 模型加载及目标函数 |
| `src/train.py` | SFT、离线 DPO、OnlineDPO、GRPO 和逐样本评估入口 |
| `src/run_suite.py` | 锁定训练与评估矩阵的串行、可恢复入口 |
| `src/analyze_results.py` | 从已完成日志与预测生成汇总、图表和结果报告 |
| `src/rewards.py` | 结构、答案与证据奖励函数 |
| `src/download_model.py` | 固定 revision 的模型缓存下载器 |
| `src/plot_concepts.py` | 实验训练实现映射图生成器 |
| `src/render_reports.py` | Markdown 报告转 HTML 预览和清单 |
| `tests/` | 数据、奖励与优化目标的单元测试 |
| `figures/` | 实验实现映射图（SVG、PNG、PDF） |
| `results/` | 训练日志、adapter、逐样本 rollout 和评估输出 |

实验结果只从实际生成的 `run_summary.json` 与 `evaluation_summary.json` 汇总；严格主指标按原始输出重新评分。已完成结果的来源和口径见 [`research/experiment_results.md`](research/experiment_results.md) 及 `results/analysis/summary.json`。

## 实验协议

### 数据与划分

`src/data.py` 以固定生成器种子 `20261028` 构造虚构中文短证据问答任务，不使用真实语料、人工标注或外部检索结果。数据包括 512 条 SFT 样本、256 条偏好训练提示及其 256 组成对偏好、96 条 validation 样本，以及各 160 条的 `test_id`、`test_ood` 与 `test_counterfactual`。偏好负例由固定生成器规则构造。数据定义、标签规则、分布与局限见 [`docs/DATA.md`](docs/DATA.md)。

模型选择与训练方案确认仅使用 `validation.jsonl`。固定方案后，三个 test split 仅作最终评估，不用于调参或构造训练偏好。训练入口会拒绝 test/validation 行作为训练输入，评估入口也会拒绝训练行。

### 固定模型与训练分支

所有训练分支采用同一基础模型和 tokenizer revision，BF16 底座、rank 8 / alpha 16 的 LoRA（`q_proj`、`v_proj`），单个 prompt 微批大小默认为 1，上下文上限 256 token，completion 上限 64 token。训练只对 completion token 计算目标；不静默截断超长 prompt 或目标。优化器更新使用梯度裁剪，逐步记录有限梯度状态。`docs/TRAINING.md` 说明训练和评估张量、掩码、目标及日志字段。

离线 DPO、OnlineDPO 和 GRPO 都从相同 seed 的 SFT adapter 初始化。离线 DPO 使用静态 `chosen` / `rejected` 对；policy 和冻结 reference 是同一个 SFT 初始化策略的独立 adapter 状态，二者可共享单个底座。不能通过关闭 SFT adapter、回到原始 Instruct 底座来代替 SFT reference。OnlineDPO 在训练过程中采样并按奖励排序候选。GRPO 每个 prompt 采样 4 个候选，以组内 reward 形成优势；每组 rollout 按配置复用两次更新。`full` 使用格式、答案和证据联合奖励；`format` 消融只检查 JSON/schema 格式，不检查答案或证据正确性。奖励定义见 [`docs/DATA.md`](docs/DATA.md)。

实验采用两个独立训练随机种子：17 和 29。预算均表示 optimizer 更新数；`grad-accum` 表示每次更新累计的微批数，不应描述成训练 epoch。完整配置：

| 分支 | 每 seed 更新预算 | 每次更新 | 额外设置 |
|---|---:|---|---|
| SFT（主方案） | 16 | batch 1，gradient accumulation 4 | `data/train_sft.jsonl`；每 seed 64 次示范抽样，有放回 |
| 离线 DPO | 64 | batch 1，gradient accumulation 4 | `data/preferences_train.jsonl`；reference 为同 seed 的 SFT 策略 |
| OnlineDPO | 64 | batch 1，gradient accumulation 2 | 最多 rollout 128 个 prompt；每 prompt 4 个候选；completion 上限 64；`full` reward |
| GRPO（full） | 128 | batch 1，gradient accumulation 2 | 2 update epochs；每 prompt 4 个候选；completion 上限 64 |
| GRPO（format 消融） | 128 | batch 1，gradient accumulation 2 | 与 full GRPO 同预算，仅将 reward 改为 `format` |

OnlineDPO 的 64 次更新、累计批次 2 与 128 个 prompt rollout 预算相互对应；GRPO 的 128 次更新、2 个 update epochs 与累计批次 2 对应每 seed 最多 128 个 prompt rollout。每个 seed 的偏好和策略优化分支都从该 seed 的 SFT adapter 独立开始。

主方案的 SFT 每 seed 更新 16 步，累计抽取 64 条示范（batch 1、gradient accumulation 4，按训练采样器有放回）。验证预算候选是 16 与 32 个 SFT optimizer updates；在 validation 上以 `format_valid >= 0.95` 且 `joint_correct < 0.95` 为候选规则，并在满足规则的候选中选择最少更新步数。当前协议选择 16。这里的 16/32/256 均为 **SFT 更新步数**，不是生成长度；`max_prompt_length=256`、`max_completion_length=64`、`max_new_tokens=64` 在选择与最终评估中保持不变。validation 指标不作为 test 结果，也不在 README 中填入数值。

可选预算对照不进入两 seed 主矩阵：可运行 `pilot_sft32_seed17`（32 步、128 次示范抽样）供 validation 预算选择，或运行 `sft_long_seed17`（256 步、1,024 次示范抽样）作为单 seed 高预算监督对照。DPO、OnlineDPO、GRPO 和 GRPO-format 始终从主方案的 SFT adapter（16 update）初始化，不以 pilot32 或 sft_long 替代。

### 评估与结果归档

基础模型仅运行一次，路径不带 seed；微调模型按训练分支和 seed 分开。对每个 test split，评估入口使用 greedy decoding，最多生成 64 个 token，默认 batch size 为 8，保存逐样本预测及分组均值。目录约定为：

```text
results/<method>_seed<17|29>/                   # 每次训练的配置、更新日志、摘要与 adapter
results/eval/base/<split>/                      # 基础模型；不加 seed
results/eval/<method>_seed<17|29>/<split>/       # 微调分支的逐样本预测和摘要
```

每个训练目录保存 `run_config.json`、`step_log.jsonl` 和 `run_summary.json`；在线方法还保存 `rollouts.jsonl`。评估目录保存 `predictions.jsonl` 与 `evaluation_summary.json`。汇总多 seed 结果时，先分别保留每个 seed 的指标，再报告跨 seed 均值和波动；不要把所有样本合并后当作独立训练重复。已完成的实验正文位于 `实验报告.md`；本地作业目录和 Piazza 状态见 [`docs/SUBMISSION.md`](docs/SUBMISSION.md)。

### 已完成结果概览

基线、SFT、SFT-256、DPO、OnlineDPO、GRPO 和 GRPO-format 的 validation 与三个 test split 均已评估：48/48 个方法、seed 与划分组合完整，共保存 6,912 条预测；分析清单无缺失预测文件、未完成评估或缺少的训练摘要。项目检查记录为 17 项测试通过。双 seed 方法按两个 seed 的严格联合正确率等权汇总并列出 min–max；基线和 SFT-256 均只有一个 seed。以下联合正确要求输出可按严格 JSON 解析、答案正确且证据编号集合完全匹配：

| 方法 | 同分布测试 | 词汇/措辞留出测试 | 反事实测试 |
|---|---:|---:|---:|
| 基线（n=1） | 0.000 | 0.000 | 0.000 |
| SFT（n=2） | 0.363 [0.338, 0.388] | 0.334 [0.306, 0.362] | 0.322 [0.312, 0.331] |
| SFT-256（n=1） | 0.994 | 0.831 | 0.706 |
| DPO（n=2） | 0.225 [0.175, 0.275] | 0.291 [0.263, 0.319] | 0.159 [0.087, 0.231] |
| OnlineDPO（n=2） | 0.366 [0.325, 0.406] | 0.366 [0.362, 0.369] | 0.378 [0.350, 0.406] |
| GRPO full（n=2） | 0.397 [0.356, 0.438] | 0.366 [0.362, 0.369] | 0.394 [0.369, 0.419] |
| GRPO format-only（n=2） | 0.325 [0.319, 0.331] | 0.334 [0.312, 0.356] | 0.297 [0.263, 0.331] |

本合成任务与当前训练预算下，DPO 相对同 seed SFT 的联合正确率差在三个测试划分均为负（均值分别为 −13.8、−4.4、−16.2 个百分点）；这只支持对本实验设置的结论。OnlineDPO 与完整奖励 GRPO 的三项均值高于格式-only GRPO；按同 seed 配对，full GRPO 相对 format-only 的差为 +7.2、+3.1、+9.7 个百分点。SFT-256 的单 seed 分数较高，但它不是双 seed 结果，不能据此估计跨种子稳定性。完整指标、失败分解、资源记录和错误案例见 [`research/experiment_results.md`](research/experiment_results.md)。

基线的严格格式正确率在三个测试划分均为 0；因此它的严格联合正确率为零不能单独说明答案内容全错。验证集诊断中，基线宽松答案正确率为 0.438，91/96 条原始输出带外围 Markdown 代码围栏。

## GitHub 与 Piazza 提交状态

实验报告与三分支归档均已完成：5.1 `survey`（`48f574c`）、5.2 `main`（`60d18bb`）、5.3 `dpo-discussion`（`d68ef96`）。三个作业均归属 `hw5`，每个帖子实际勾选全部六个标签：`hw1`、`hw2`、`hw3`、`hw4`、`hw5`、`other`。Piazza 综述原帖 [post 168](https://piazza.com/class/mttcaxxvtxe5vj/post/168) 已更新；实验帖 [post 169](https://piazza.com/class/mttcaxxvtxe5vj/post/169) 和 DPO 思考题帖 [post 170](https://piazza.com/class/mttcaxxvtxe5vj/post/170) 已发布并核验。完整映射和核验记录见 [`docs/SUBMISSION.md`](docs/SUBMISSION.md)。

## 环境安装

建议使用 Python 3.12。目标 GPU 环境是具有兼容 NVIDIA 驱动的 Windows 或 Linux CUDA 12.6 主机。仓库的现有项目 `.venv` 使用 `--system-site-packages` 复用了已存在的 CUDA Torch；在新机器上应创建干净环境，不依赖共享 Conda/base 包。

Windows PowerShell（从仓库根目录执行）：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.12.0+cu126"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux CUDA 主机：

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu126 'torch==2.12.0+cu126'
python -m pip install -r requirements.txt
```

`requirements.txt` 固定 PyTorch CUDA wheel、Transformers、Datasets、HF Hub、PEFT、Accelerate、Matplotlib，以及 `render_reports.py` 实测依赖 Markdown 3.4.1 和 BeautifulSoup4 4.12.3。CPU-only 或不同 CUDA wheel 应替换为目标平台的官方 PyTorch wheel，并相应修改 Torch 版本约束。验证安装：

```bash
python -c "import torch, transformers, datasets, peft, accelerate, markdown, bs4; print(torch.__version__, torch.cuda.is_available()); print(transformers.__version__, peft.__version__, accelerate.__version__); print(markdown.__version__, bs4.__version__)"
```

当前公开 Hugging Face 模型不需要登录。模型文件存入当前用户的 Hugging Face 默认缓存，不复制到 Git 仓库。Windows 若不启用 Developer Mode，Hub 可能不能使用 symlink，但模型仍可缓存。

## 复现步骤

以下命令均从仓库根目录运行。Windows PowerShell 使用项目 venv 的解释器；Linux CUDA 环境先按上节激活 `.venv` 并使用 `python`。

### 准备模型、数据与图表

```powershell
.\.venv\Scripts\python.exe -m src.download_model
.\.venv\Scripts\python.exe -m src.data --seed 20261028
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.plot_concepts
```

下载脚本固定模型 ID 与 revision，只取配置、tokenizer 和 `model.safetensors` 必要文件；数据命令按固定 seed 重建 `data/`；图表命令只重建实验实现映射图（SVG、PNG、PDF）。单元测试不启动训练。

### 锁定训练与评估流水线

`src.run_suite` 按 seed 串行执行主训练矩阵和四个划分的评估，校验并跳过已完成输出；遇到不完整或配置不匹配的既有输出会停止，避免把部分运行当作完成。该入口要求先存在 seed 17 的 16-update SFT adapter 和 seed 17 的 256-update SFT 对照 adapter（下方的 SFT 命令可生成），从仓库根目录运行且不接受命令行参数：

```powershell
& .\.venv\Scripts\python.exe -m src.run_suite
```

Linux/macOS 激活 `.venv` 后运行：

```bash
python -m src.run_suite
```

主矩阵的结果汇总入口是 `src.analyze_results`。默认读取所有方法、两个 seed 和 validation/test 划分，将 CSV/JSON 写入 `results/analysis/`，更新结果图和 `research/experiment_results.md`；该命令只读取已有训练与评估产物，不加载模型或启动训练：

```powershell
& .\.venv\Scripts\python.exe -m src.analyze_results
```

Linux/macOS 激活 `.venv` 后运行 `python -m src.analyze_results`。如需更改输入/输出目录或筛选方法、seed，可用 `python -m src.analyze_results --help` 查看支持的参数。

### 运行两 seed 训练矩阵

以下 PowerShell 流程按同一 seed 先训练 SFT，再分别训练 DPO、OnlineDPO、GRPO 与 GRPO-format 消融。各分支复用该 seed 的 SFT 初始化，输出目录彼此独立。长训练由使用者逐条启动和监控；若某次失败，应根据其 run 日志处理后再继续后续步骤。

```powershell
$python = ".\.venv\Scripts\python.exe"
foreach ($seed in @(17, 29)) {
    $sft = "results/sft_seed$seed"
    & $python -m src.train sft --seed $seed --steps 16 --batch-size 1 --grad-accum 4 --input data/train_sft.jsonl --output $sft
    if ($LASTEXITCODE -ne 0) { throw "SFT seed $seed failed" }

    $sftAdapter = "$sft/adapter"
    $dpo = "results/dpo_seed$seed"
    & $python -m src.train dpo --seed $seed --steps 64 --batch-size 1 --grad-accum 4 --input data/preferences_train.jsonl --adapter $sftAdapter --output $dpo
    if ($LASTEXITCODE -ne 0) { throw "DPO seed $seed failed" }

    $online = "results/online_dpo_seed$seed"
    & $python -m src.train online_dpo --seed $seed --steps 64 --batch-size 1 --grad-accum 2 --max-rollout-prompts 128 --max-completion-length 64 --reward-mode full --input data/train_pref.jsonl --adapter $sftAdapter --output $online
    if ($LASTEXITCODE -ne 0) { throw "OnlineDPO seed $seed failed" }

    $grpo = "results/grpo_seed$seed"
    & $python -m src.train grpo --seed $seed --steps 128 --batch-size 1 --grad-accum 2 --update-epochs 2 --max-completion-length 64 --reward-mode full --input data/train_pref.jsonl --adapter $sftAdapter --output $grpo
    if ($LASTEXITCODE -ne 0) { throw "GRPO full seed $seed failed" }

    $grpoFormat = "results/grpo_format_seed$seed"
    & $python -m src.train grpo --seed $seed --steps 128 --batch-size 1 --grad-accum 2 --update-epochs 2 --max-completion-length 64 --reward-mode format --input data/train_pref.jsonl --adapter $sftAdapter --output $grpoFormat
    if ($LASTEXITCODE -ne 0) { throw "GRPO format seed $seed failed" }
}
```

命令中的 `--steps` 是 optimizer updates。配置未显式覆盖的学习率、LoRA 参数、group size 及目标系数来自 `configs/experiment.json`。结果值以每个 run 的配置和日志为准，而不是仅凭命令文本推断。

### 可选 SFT 预算 pilot 与高预算对照

若需保留 16/32 候选的独立运行记录，可对 seed 17 的主方案 checkpoint 与 32-update pilot 分别评估 validation。候选评估统一使用 `max_new_tokens=64`，不改变 prompt/completion 长度。所选主预算为 16；pilot32 不属于主矩阵。256-update 对照同样是 seed 17 单次运行，不纳入两 seed 主方案均值。

```powershell
$python = ".\.venv\Scripts\python.exe"
$sftPilot32 = "results/pilot_sft32_seed17"
& $python -m src.train sft --seed 17 --steps 32 --batch-size 1 --grad-accum 4 --input data/train_sft.jsonl --output $sftPilot32
if ($LASTEXITCODE -ne 0) { throw "SFT pilot32 failed" }
& $python -m src.train evaluate --input data/validation.jsonl --output results/eval/pilot_sft32_seed17/validation --adapter "$sftPilot32/adapter" --batch-size 8 --max-new-tokens 64
if ($LASTEXITCODE -ne 0) { throw "SFT pilot32 validation failed" }

$sftLong = "results/sft_long_seed17"
& $python -m src.train sft --seed 17 --steps 256 --batch-size 1 --grad-accum 4 --input data/train_sft.jsonl --output $sftLong
if ($LASTEXITCODE -ne 0) { throw "SFT long comparison failed" }
```

对 16-update 主 checkpoint 与 pilot32 的 validation 评估都使用 `--max-new-tokens 64`。主 checkpoint 输出路径为 `results/eval/sft_seed17/validation`；32-update pilot 输出路径为 `results/eval/pilot_sft32_seed17/validation`。示例：

```powershell
$python = ".\.venv\Scripts\python.exe"
& $python -m src.train evaluate --input data/validation.jsonl --output results/eval/sft_seed17/validation --adapter results/sft_seed17/adapter --batch-size 8 --max-new-tokens 64
& $python -m src.train evaluate --input data/validation.jsonl --output results/eval/pilot_sft32_seed17/validation --adapter results/pilot_sft32_seed17/adapter --batch-size 8 --max-new-tokens 64
```

如果运行 256-update 对照，其 validation 输出写入 `results/eval/sft_long_seed17/validation`；该模型仍为单 seed 的附加对照，不替换 DPO、OnlineDPO 或 GRPO 的主 SFT 初始化策略。

### 在 validation 确认后运行最终评估

方案确认时可将 `data/validation.jsonl` 评估输出保存到相应方法目录的 `validation/` 子目录。确定超参数、奖励和 checkpoint 后，再运行最终三 split 测试。下面的 PowerShell 函数按要求输出到 `results/eval/<method>_seedN/<split>`；基础模型统一放在 `results/eval/base/<split>`：

```powershell
$python = ".\.venv\Scripts\python.exe"
function Invoke-Eval([string]$method, [string]$adapter) {
    foreach ($split in @("test_id", "test_ood", "test_counterfactual")) {
        $output = "results/eval/$method/$split"
        & $python -m src.train evaluate --input "data/$split.jsonl" --output $output --adapter $adapter --batch-size 8 --max-new-tokens 64
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $method / $split" }
    }
}

Invoke-Eval "base" "none"
foreach ($seed in @(17, 29)) {
    Invoke-Eval "sft_seed$seed" "results/sft_seed$seed/adapter"
    Invoke-Eval "dpo_seed$seed" "results/dpo_seed$seed/adapter/policy"
    Invoke-Eval "online_dpo_seed$seed" "results/online_dpo_seed$seed/adapter/policy"
    Invoke-Eval "grpo_seed$seed" "results/grpo_seed$seed/adapter/policy"
    Invoke-Eval "grpo_format_seed$seed" "results/grpo_format_seed$seed/adapter/policy"
}
# Optional single-seed high-budget supervised comparison, if produced:
if (Test-Path "results/sft_long_seed17/adapter") {
    Invoke-Eval "sft_long_seed17" "results/sft_long_seed17/adapter"
}
```

测试集应在方案冻结后运行。若为 validation 选型使用同一循环，应将 split 列表替换为 `@("validation")`，并输出到对应的 `results/eval/<method>_seedN/validation/`；这些值不能与最终 test 结果混报。

### 渲染报告预览

渲染器生成 HTML、纯文本和 `.manifest.json`，不会改写报告 Markdown。不传 `--image-map` 时，HTML 使用相对路径引用仓库内图片，适合本地预览。要准备发布版本，必须传入从 Piazza 图片上传流程取得的 URL 映射 `--image-map`。若 map 尚缺少某些上传 URL，可同时传入 `--draft-missing-images`，仅为未发布草稿插入显式占位；不能将此草稿用于最终发布。`--part-limit 200000` 保留完整报告为单个 HTML；低于输出大小的阈值会额外生成章节对齐的 `.part-NN.html` 与 `.part-NN.txt`，清单会记录分段状态。

```powershell
.\.venv\Scripts\python.exe -m src.render_reports 实验报告.md --output rendered/实验报告.html --part-limit 200000
.\.venv\Scripts\python.exe -m src.render_reports 实验报告.md --output rendered/piazza.html --image-map docs/piazza_image_map.json --part-limit 200000
```

本地预览输出到 `rendered/实验报告.html`，带 Piazza 图片 URL 的版本输出到 `rendered/piazza.html`；两种版本各自附带 `.txt` 和 `.manifest.json`。修改 `实验报告.md` 后重渲染这两个产物。`--draft-missing-images` 仅在提供 `--image-map` 时适用于未发布草稿，不能用于最终发布 HTML。README 命令只生成本地文件，不会发布 Piazza 帖子。

## 第三方源码参考与本仓库边界

下列固定提交用于理解公开实现的训练调用链；函数级阅读路径、行号和具体职责见 [`docs/CODE_READING.md`](docs/CODE_READING.md)。这些仓库是源码分析对象，不是本项目运行依赖：

| 开源实现 | 固定源码提交 |
|---|---|
| [Hugging Face TRL](https://github.com/huggingface/trl/tree/c886cf23d74acd8dfed95b63943a8dba0f25d85f) | current `c886cf23d74acd8dfed95b63943a8dba0f25d85f`；PPO history `1f8ba929fbb6962249386f7dce20155da8a26cda`；open-r1 dependency `ef4b0b225cf04600332a12a2c83f187623312098` |
| [trlx](https://github.com/CarperAI/trlx/tree/3340c2f3a56d1d14fdd5f13ad575121fa26b6d92) | `3340c2f3a56d1d14fdd5f13ad575121fa26b6d92` |
| [DeepSpeed-Chat / DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples/tree/0c6dd5ed1ec0f0967ae0564534ecfa6f157c6746) | `0c6dd5ed1ec0f0967ae0564534ecfa6f157c6746` |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF/tree/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53) | `dc2a7ad326f619fc5a47737ea47d1920ce2c0b53` |
| [verl](https://github.com/volcengine/verl/tree/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4) | `2ab35292039a51175a4b3d3b3b2d76e951a0d3e4` |
| [slime](https://github.com/THUDM/slime/tree/540ea919e45b8b929198abce008a103be9cfc779) | `540ea919e45b8b929198abce008a103be9cfc779` |
| [open-r1](https://github.com/huggingface/open-r1/tree/5b6ff22b3fb7aa069c54866e517f39dfc3160e09) | `5b6ff22b3fb7aa069c54866e517f39dfc3160e09`；其 GRPO recipe 另固定 TRL `v0.18.0` |
| [MiniMind](https://github.com/jingyaogong/minimind/tree/233d00fa068ec2b8c0d4f07b703934d6391a5a73) | `233d00fa068ec2b8c0d4f07b703934d6391a5a73` |

本仓库的训练脚本和合成数据生成器是独立的低资源最小实现，用于比较所列方法的核心目标和反馈回路；它们不是上述框架的移植，也不构成对 DeepSeek-R1 全量训练流程、规模、数据或算力配置的复现。

## 资源与解释范围

当前实现面向 RTX 3060 Laptop 6 GiB 显存，使用一个 0.5B 模型、LoRA、短上下文、batch 1、梯度累积和梯度检查点。训练所需 GPU 内存和墙钟时间取决于实际序列、奖励生成、显存中其他进程与 attention backend，应从 `run_summary.json` 实测记录报告。环境冒烟推理数据和 6 GiB 资源规划估算分列在 [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)；冒烟测试不等于训练结果。

合成数据规模、格式、评分定义及模板迁移限制见 [`docs/DATA.md`](docs/DATA.md)。开源代码对照的版本快照和实现引用见 [`docs/CODE_READING.md`](docs/CODE_READING.md) 与 [`research/code_analysis.md`](research/code_analysis.md)。
