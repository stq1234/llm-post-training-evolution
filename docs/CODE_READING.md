# 源码阅读记录：后训练实现比较

## 范围和审计口径

- 审阅日期：2026-10-28。
- 研究对象：TRL（当前实现、open-r1 锁定的 v0.18.0 和历史 v0.11.4 PPO）、trlx、DeepSpeed-Chat、OpenRLHF、verl、slime、open-r1、MiniMind。
- 代码入口及调用链记录于 research/code_sources.json；正文草稿为 research/code_analysis.md；短源码证据为 research/code_excerpts/implementation_evidence.md。
- 版本锁定：分别通过本地 git rev-parse HEAD 读取实际检出 commit，并与仓库 git ls-remote origin HEAD 对照。八个公开仓库的当前 HEAD 与远端 HEAD 一致。TRL v0.11.4 使用 tag peeled commit；annotated tag object 与 peeled commit 分开保存。
- 永久源码链接以 GitHub /blob/<commit>/<path> 形式固定到实际审阅提交。分支、release 和训练器默认值会变化，因此不能以当前默认分支回看旧实验。
- 公开实现的功能表述只按对应源代码检查，不从 README、论文摘要或模型结果反推调用链。各仓库的配置和算法选择可能有多分支，正文只说明审阅到的代码路径及其配置选择。
- 没有运行模型训练或项目测试；本工作包限定为源代码阅读与来源核验。临时 sparse checkout 用于读代码及核对行号，分析完成后删除；不修改外部仓库源文件。

## 锁定的源码版本

| 项目快照 | 仓库 | ref | 完整 commit | 备注 |
|---|---|---|---|---|
| TRL current | https://github.com/huggingface/trl | main | c886cf23d74acd8dfed95b63943a8dba0f25d85f | DPO、Online DPO、GRPO |
| TRL dependency for open-r1 | https://github.com/huggingface/trl | v0.18.0 | ef4b0b225cf04600332a12a2c83f187623312098 | open-r1 setup.py pins trl[vllm]==0.18.0；tag object 为 5e2ef1a901f4169c49aa3c351c72a031580ea9b4 |
| TRL PPO history | https://github.com/huggingface/trl | v0.11.4 | 1f8ba929fbb6962249386f7dce20155da8a26cda | v0.11.4 peeled commit；annotated tag object 为 714cd42f67cb2c8bef91546dbb2258e326c03c89 |
| trlx | https://github.com/CarperAI/trlx | main | 3340c2f3a56d1d14fdd5f13ad575121fa26b6d92 | Accelerate PPO |
| DeepSpeed-Chat | https://github.com/deepspeedai/DeepSpeedExamples | master | 0c6dd5ed1ec0f0967ae0564534ecfa6f157c6746 | step2 DPO、step3 PPO |
| OpenRLHF | https://github.com/OpenRLHF/OpenRLHF | main | dc2a7ad326f619fc5a47737ea47d1920ce2c0b53 | Ray/vLLM 训练与 PPO/GSPO 目标 |
| verl | https://github.com/volcengine/verl | main | 2ab35292039a51175a4b3d3b3b2d76e951a0d3e4 | v1 Transfer Queue trainer |
| slime | https://github.com/THUDM/slime | main | 540ea919e45b8b929198abce008a103be9cfc779 | SGLang rollout、Megatron train |
| open-r1 | https://github.com/huggingface/open-r1 | main | 5b6ff22b3fb7aa069c54866e517f39dfc3160e09 | TRL GRPO 配方与奖励 |
| MiniMind | https://github.com/jingyaogong/minimind | master | 233d00fa068ec2b8c0d4f07b703934d6391a5a73 | DPO/PPO/GRPO 本地脚本 |

## 主要阅读路径

下列文件和行范围以锁定提交为准。函数逐项索引、永久链接及各环节角色在 research/code_sources.json。

### TRL 当前

- trl/trainer/grpo_trainer.py：GRPOTrainer（142）；训练器 generation buffer（1557–1598）；采样与环境（2371 起）；group reward、old/ref/sample log-prob（2788–2835）；policy loss、mask 与 reduction（3067–3229）；checkpoint 覆盖（3432）。
- trl/experimental/online_dpo/online_dpo_trainer.py：在线生成与 reward（947–1147）；双样本奖励排序、reference forward、序列 log-prob 求和与 sigmoid/IPO（1149–1285）；checkpoint（1371–1377）。
- trl/trainer/dpo_trainer.py：标准 DPO 的 chosen/rejected 序列 log-prob 及 reference scoring（1337–1415）；checkpoint（1776）。
- trl/trainer/base_trainer.py：TRL _BaseTrainer 基于 Transformers Trainer（63）。

### TRL v0.18.0：open-r1 配方实际锁定的 GRPO 内核

- open-r1/setup.py 第 72 行锁定 trl[vllm]==0.18.0；因此 open-r1 仓库 commit 与该 TRL tag 的组合才是其可复现版本定义，不能将 current main 的新增目标反向归于该配方。
- trl/trainer/grpo_trainer.py：GRPOTrainer（280）；采样 EOS/completion mask、旧策略 log-prob、reward aggregation 与跨进程 gather（983–1216）；当前/reference log-prob 与 grpo/bnpo/dr_grpo reduction（1324–1404）。
- v0.18.0 的审阅范围用于确认 open-r1 真实调用内核；最新 current main 的新增 loss type、tool mask 和更细的采样概率校正仅在 TRL current 记录中报告。

### TRL v0.11.4 历史 PPO

- trl/trainer/ppo_trainer.py：PPOTrainer（117）；生成（462）；step（655）；batched forward 与 response mask（979–1064）；minibatch backward/optimizer（1067–1110）；per-token KL reward（1112–1164）；GAE 与 clipped actor/value loss（1166–1258）。
- 该代码只用于版本演进对比；不得把旧版本 PPOTrainer 的存在表述为当前主分支继续提供相同 API。

### trlx

- trlx/utils/loading.py：通过 registry 注册并选择 PPO trainer 与 pipeline 构造器（3–50）。
- trlx/pipeline/offline_pipeline.py：PromptPipeline 处理提示、保留 metadata 并构造 loader（119–185）。
- trlx/trainer/accelerate_ppo_trainer.py：PPO loss（127–204）；prompt sampling、跨进程 gather、主进程 reward_fn 与 score scatter（251–341）；当前/ref log-prob、value 与 reward 写入 experience（343–524）；模型保存（526）。
- trlx/models/modeling_ppo.py：GAE（136–173）、masked token-average clipped policy/value loss（175–227）。
- trlx/trainer/accelerate_base_trainer.py：多轮 minibatch/microbatch 的 backward、optimizer 与 scheduler（518–610）；Accelerate checkpoint state 在训练过程持久化。

### DeepSpeed-Chat

- applications/DeepSpeed-Chat/dschat/rlhf/ppo_trainer.py：actor generation 和 actor/ref/RM/critic 经验前向（74–179）；KL token reward 与末端奖励（181–194）；GAE、actor/critic 各自 backward/step 及 mixed-precision overflow 对齐（196–320）。
- applications/DeepSpeed-Chat/dschat/rlhf/rlhf_engine.py：actor、reference、reward model 与 critic 创建/共置、ZeRO/offload 配置。
- applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/main.py：提示数据、rollout 缓存、PPO 更新循环及保存（399，513–640）。
- applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/README.md：该项目记录的 batch-size / PPO epoch 不稳定性观察（69–80）；只作为项目报告的经验陈述，不替代参数的源码定义。
- applications/DeepSpeed-Chat/training/step2_dpo_finetuning/main.py：masked target log-prob（219–240）和成对偏好训练循环（456–510）。

### OpenRLHF

- openrlhf/cli/train_ppo_ray.py：Ray strategy 与 vLLM、actor、reference、reward、critic 角色组规划（19–95）。
- openrlhf/trainer/ppo_trainer.py：make_experience -> dynamic balance -> append -> PPO -> vLLM weight broadcast 的训练步（213–264）；checkpoint（384–414）；fit/resume（503–565）。
- openrlhf/trainer/ppo_utils/experience_maker.py：样本拆分（80–96）；Actor、RM、critic、reference 并行计算与 Experience 字段（112–233）；组奖励 estimator 和逐 token GAE（238–387）。
- openrlhf/models/loss.py：全局 token/sample reduction（11–39）；PPO/GSPO、sequence ratio、off-policy gates（116–259）。
- openrlhf/trainer/ray/ppo_actor.py：replay buffer、loss normalization、actor loss 和 KL（153–205、259–359）；保存 actor（608–617）。

### verl

- verl/trainer/main_ppo.py：TaskRunnerV1、trainer_mode 选择、Transfer Queue 和 AgentLoopManager（107–178）。
- verl/trainer/ppo/v1/trainer_base.py：replay batch 到 reward/old/ref/value/advantage/critic/actor 的 _step_once（563–613）；完整 old/ref log-prob 和 rollout correction 分支（1635–1803）；actor update（1828–1871）；checkpoint（949–1010）。
- verl/trainer/ppo/core_algos.py：按 uid/prompt 组统计 outcome reward 的 GRPO estimator（268–335）；五类有效 token / sequence reduction（1140–1206）。
- verl/workers/engine_workers.py：mini-batch update dispatch（276–297）。

### slime

- train.py：rollout manager 生成、actor/critic async_train、training_completed 和 checkpoint 调度（13–83）。
- slime/ray/rollout.py：在线或 retained rollout、重放与 train-data 转换（317–390、503–557）。
- slime/data/batch_builder.py：持久化批次转换计划（48–110）；按 prompt group 对唯一 rollout outcome 做 reward centering/normalization（207–254）。
- slime/rollout/sglang_rollout.py：并发 sample generation、group reward model（388–439）；rollout async queue（492–545）。
- slime/backends/megatron_utils/loss.py：优势估计器选择（725–850）；策略目标及校正（1128–1245）；Megatron loss scale（1495–1535）。
- slime/backends/megatron_utils/actor.py：训练数据适配及 actor/critic 更新（400–485）；保存与推理权重同步（602–640）。

### open-r1

- src/open_r1/grpo.py：数据集映射成对话 prompt、奖励 registry、TRL GRPOTrainer 创建、train/save（73–147）。
- setup.py：依赖锁定 trl[vllm]==0.18.0（72）。
- src/open_r1/rewards.py：答案解析核验、格式奖励、reward registry（40–111、646–690）。
- recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml：vLLM、生成数、prompt/completion 长度与奖励加权（1–52）。
- 实际依赖版本处理 None 奖励时将其转 NaN，再对 reward-function 维度 nansum；全 None 行会以聚合零分参与组统计（TRL v0.18.0 GRPOTrainer 1171–1208）。TRL current main 的 unscorable_mask 则排除全 None 行并将优势设零。该行为差异需随版本说明。
- 代码将 GRPO 核心执行委托给已安装的 TRL；该 recipe 不实现另一个独立 trainer。

### MiniMind

- trainer/train_dpo.py：成对数据、chosen/rejected 与 masked DPO 序列 log-prob、优化器与保存恢复（45–119、173–214）。
- trainer/train_ppo.py：独立 actor-critic PPO 训练入口及 actor/critic optimizer step（239–250、407–409）。
- trainer/train_grpo.py：规则与 reward-model 信号（37–68）；rollout、采样时对数概率、old/reference log-prob、completion mask、组内优势、每序列均值 loss、optimizer 与 checkpoint（71–196、245–322）。
- trainer/rollout_engine.py：Torch 与 SGLang 实现共享的采样结果接口（23–150）。

## 审阅发现与复现提示

1. 训练中可能同时存在 πrollout、πold、πθ 与 πref。除非实现明确采用 bypass 模式，否则不得把 rollout log-prob、PPO old log-prob 和 reference log-prob 合并成一个“旧策略概率”描述。
2. mask 与 loss 分母共同定义目标。报告必须固定 prompt/action/completion mask、EOS 位置、工具 token 是否计入、序列是否被截断，以及 per-token / per-sequence / 固定 horizon 的归约式。
3. 在线生成使用 vLLM/SGLang 时，记录采样时的模型权重版本、token log-prob、sampling temperature、top-p、generation batch 与同步间隔；训练侧出现 importance sampling、truncation、rejection mask 或 score centering 时要作为算法配置报告。
4. 分布式训练应保存全局有效 token/样本计数和并行后端的 loss scaling 方式；OpenRLHF/verl/slime 都把动态 batch、DP 聚合或 Megatron 微批缩放纳入实现。
5. 采样组与 reward 的索引稳定性是 GRPO 的数学前提。随机打乱、跨 rank 切分、异步重放或动态过滤前，必须保留 prompt/group identity；否则组内均值将不再是对应 prompt 的 baseline。
6. DeepSeek-R1 的论文及公开报告不构成其训练器源码。本文没有取得 DeepSeek-R1 官方完整训练源代码，不把其技术描述归属给本节所读的通用开源实现。
7. 源码比较不等于训练可运行性验证；本工作包未安装依赖、未调用 GPU、未训练模型，也没有审定各论文指标的复现结果。
