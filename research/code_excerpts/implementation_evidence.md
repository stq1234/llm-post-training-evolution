# 源码核对用短证据

以下是所锁定提交中的极短代码片段，配合 research/code_sources.json 中的 commit 和永久路径复核。片段只用于说明调用链定位，不替代完整实现。

| 实现 | 源码定位 | 片段 | 核对用途 |
|---|---|---|---|
| TRL current GRPO | trl/trainer/grpo_trainer.py:3074 | mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"] | 策略损失按 response 与工具动作交集屏蔽 |
| TRL v0.11.4 PPO | trl/trainer/ppo_trainer.py:1146 | reward[last_non_masked_index] += score | 标量奖励写至最后有效动作 |
| trlx PPO | trlx/trainer/accelerate_ppo_trainer.py:309–311 | reward_fn should return list of rewards at each token per sample | 奖励函数接口粒度为 token |
| DeepSpeed-Chat PPO | applications/DeepSpeed-Chat/dschat/rlhf/ppo_trainer.py:184–192 | kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs) | reference KL 作为每 token 非分数奖励 |
| OpenRLHF loss | openrlhf/models/loss.py:35 | seq_loss = (loss * loss_mask).sum(dim=-1) / (token_counts + 1e-8) | 每条序列先按有效 token 求均值 |
| verl | verl/trainer/ppo/v1/trainer_base.py:1647 | data["old_log_probs"] = data.pop("rollout_log_probs") | bypass 模式直接复用 rollout 概率 |
| slime | slime/data/batch_builder.py:244 | centered = rewards[positions] - rewards[positions].mean() | 对同一 prompt 组的奖励中心化 |
| open-r1 | src/open_r1/grpo.py:112 | trainer = GRPOTrainer( | GRPO 核心 trainer 由 TRL 提供 |
| MiniMind | trainer/train_grpo.py:124 | advantages = (rewards - mean_r) / (std_r + 1e-4) | 每提示组内标准化标量奖励 |
| open-r1 dependency | setup.py:72 | trl[vllm]==0.18.0 | 该配方使用的 GRPO 内核版本由依赖锁定 |
| TRL v0.18.0 | trl/trainer/grpo_trainer.py:1371–1376 | grpo / bnpo / dr_grpo | 该版本 loss 只提供三种 reduction 分支 |
| TRL v0.18.0 unscorable reward | trl/trainer/grpo_trainer.py:1195–1200 | rewards_per_func.nansum(dim=1) | 全 None 奖励聚合为零并参与组统计 |

## 逐项目核对摘要

- TRL current：在线采样、按 prompt 分组计算奖励、分别记录 old/reference/sampling log-prob；tool mask 会从策略目标排除工具生成。
- TRL v0.11.4：actor、reference 与 value head 对批次前向；每 token KL reward、GAE 与 PPO clipped policy/value loss。
- trlx：跨进程汇总样本，只在主进程调用 token-score reward 函数，随后分发奖励和形成 PPO experience。
- DeepSpeed-Chat：四个逻辑网络构造 PPO experience，标量 RM score 加至结束动作；actor 与 critic 分别优化。
- OpenRLHF：Ray actor group 计算经验，奖励组内归一与可选 GAE/RLOO/GRPO 类基线，PolicyLoss 区分 PPO/GSPO 及全局 loss reducer。
- verl：队列化样本上的 old/ref/value/advantage 字段；bypass 和 decoupled 选择决定旧策略语义；GRPO 与 loss reducer 独立可配。
- slime：SGLang/Ray 生成和 group reward，BatchBuilder 固定样本选择/转换计划，Megatron 执行优势、策略损失与并行缩放。
- open-r1：setup.py 固定 TRL v0.18.0；奖励函数返回 None 后，锁定内核以 nansum 聚合，全 None 行进入零分组统计；current main 对不可评分行的处理不同。
- MiniMind：DPO 使用冻结参考模型计算偏好目标；GRPO 使用 rollout engine 输出旧策略 log-prob、组内奖励优势和响应掩码。
