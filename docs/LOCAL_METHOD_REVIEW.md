# 本地后训练目标实现只读审查

## 范围与结论

本审查只读取 `src/train.py`、`src/modeling.py`、`src/rewards.py`、`configs/experiment.json` 和 `tests/test_objectives.py`，重点核对 SFT completion mask、DPO 的参考/策略适配器、Online DPO 的采样语义、GRPO 的 old/reference 概率、结束与截断掩码、优势归一化、梯度累积及两轮复用。未修改这些训练与测试文件，也未运行 GPU 或测试。根据本地现有记录，17 项 CPU 测试和 8 步 SFT GPU 运行已由主线完成；DPO 与 GRPO 的 GPU 验证仍待进行。

没有发现足以认定为当前确定性算法错误、需要阻止 DPO/GRPO 验证的高置信度缺陷。实现中有两项应在实验解释中明确：GRPO 将达到生成上限而未生成 EOS 的响应仍作为终止样本评分并优化；Online DPO 的 rollout 日志不保存完整 token 序列与逐 token 行为概率，因此不能仅凭日志逐 token 重建采样分布。两项均不等于 DPO/GRPO 损失公式本身错误。

## 目标路径核对

| 环节 | 源码核对 | 审阅结论 |
|---|---|---|
| SFT completion mask | `src/modeling.py:47–102, 124–177`；`src/train.py:266–316` | 使用实际 chat template 构造提示，目标追加 EOS；提示标签为 `-100`，completion 与 EOS 的 response mask 为 1；右侧 padding 的 label 和 mask 均屏蔽。训练通过 `logits[:, :-1]` 对齐 `labels[:, 1:]`，因此第一个 completion token 由最后一个提示位置预测，EOS 参与损失。过长提示或目标显式报错，不静默截断。该路径与测试断言相符。 |
| 离线 DPO | `src/modeling.py:105–116, 190–203, 313–350, 361–373`；`src/train.py:332–408` | 基座上分别挂载从 SFT 加载的 `policy`（可训练）与 `reference`（冻结）LoRA adapter。参考前向在 `no_grad` 与冻结上下文中执行，随后启用 policy 求梯度。对 chosen/rejected 分别累计有效 completion token（包括 EOS）的序列 log-prob 总和，按标准 sigmoid DPO 对比策略与参考的偏好 margin；提示和 padding 不计分。不是按长度平均的 DPO。 |
| Online DPO rollout 与旧策略 | `src/modeling.py:395–450`；`src/train.py:421–489, 492–606` | 生成明确启用采样、温度 1、top-p 1、top-k 0、无重复惩罚；对固定 Qwen 默认配置而言，代码未主动进行 top-p/top-k 截断。一个优化步先生成整批 prompts 的候选，再统一计算偏好对并更新，因此候选来自该更新步开始时的 policy；DPO 目标本身不需要 PPO 式 old-policy 概率比。候选的 policy 概率在训练时重新前向计算，reference 概率由冻结 SFT adapter 计算。 |
| Online DPO 偏好构造 | `src/train.py:443–489, 519–596` | 组内仅取最高分和最低分响应；最高/最低分相同时跳过该提示。非平局组按 pair 数加权累积梯度，tie-only microbatch 不进入损失分母。固定 rollout budget 可使实际更新数低于 steps 上限；程序将实际更新数写入 summary，并记录 budget exhausted 与 skipped ties。该预算策略已由主线确认是有意的实验口径，本审查不将其列为缺陷。 |
| GRPO rollout、old 与 reference | `src/modeling.py:395–450, 206–260, 313–374`；`src/train.py:609–662` | 候选由 policy 生成；更新前逐个响应在冻结 dropout、评估模式下重新计算行为策略 `old_token_logps`，随后切换到独立的冻结 SFT reference adapter 计算 `reference_token_logps`。两类概率分别缓存，更新时与 current log-prob 分别进入 PPO ratio 与 KL 项；old/reference 在损失中 detach。采样概率没有被显式缓存，见后文审查事项。 |
| GRPO mask、优势与损失 | `src/modeling.py:119–187, 206–260`；`src/train.py:665–715` | EOS 在生成响应中被保留并计入 response mask；EOS 后的 padding 不在 completion token 中，右 padding另由 attention mask排除。未遇到 EOS 而达到 `max_new_tokens` 时，截断前缀的所有生成 token 仍有效。奖励按 prompt 组做均值中心化和总体标准差归一，标准差为零时优势为零；policy 与采样 reverse-KL 都先对每条响应的有效 token 求均值，再对响应平均。此处是等响应权重，不是全局有效 token 加权平均。 |
| 梯度累积与两轮复用 | `src/train.py:257–264, 665–760`；`configs/experiment.json` | GRPO 将 `batch_size × grad_accum` 个 prompt 的全部响应纳入一次 optimizer update，逐响应反向并除以总 completion 数，再只执行一次 optimizer step；因此在当前等响应损失定义下，累积缩放一致。`update_epochs=2` 时第二轮沿用同一组 responses、advantages、old 与 reference 概率，`optimizer.zero_grad` 重置梯度，current policy 则已在第一轮更新。此为 PPO 式多轮更新，不是每轮重新采样。 |

## 需要限定的实现事项

### 截断响应作为终止样本

`generate_candidates` 记录 EOS 是否缺失及 `truncated` 标记（`src/modeling.py:430–449`），但 `_make_grpo_rollout_batch` 对每个候选仍调用 `generated_feature` 并缓存所有 token 概率（`src/train.py:621–648`）。更新时只以 completion mask 和 attention mask 过滤 token，不读取 `truncated`（`src/train.py:680–704`）。因此命中生成上限的非 EOS 前缀会得到与 EOS 完成响应同一形式的 outcome reward，并对其已生成 token 施加 policy/KL 损失。对当前短 JSON 任务，这会把被长度上限截断、无法形成完整答案的响应当作一次终止失败或部分成功。

这不是可自动判定的错误：若实验定义明确将长度上限视为终止条件，该实现自洽；若方法要求排除截断轨迹或进行 bootstrap，则当前代码未实现该语义。DPO 及 GRPO 结果应报告截断率，并把截断响应计入方式固定为终止样本，不能在分析时将其当作完整自然终止样本。

### Online DPO 行为概率的审计留存

Online DPO 显式将采样温度、top-p、top-k、beam 与重复惩罚设为近似未截断的采样设置（`src/modeling.py:414–428`）。rollout JSONL 仅写 prompt、候选文本、奖励、索引及截断标记（`src/train.py:449–489`）；它不保存 completion token IDs、逐 token 采样 log-prob 或逐次完整 generation 参数。当前 DPO 目标不使用 old-policy ratio，所以这些概率缺失本身不导致损失公式错误；又因每次策略更新前先完成对应 batch 的采样，样本与更新时 current policy 对齐。限制在于运行后不能从 rollout 日志单独逐 token 核实实际行为分布，且使用 `--model-path` 替换默认模型时，未显式覆盖的 generation-config processors 也可能影响实际采样支持集。若需要离线审计，应保留 token IDs、生成配置以及生成时行为概率；否则结论限于源码设定和该次运行时模型配置。

## 严重度清单

| 严重度 | 问题 | 证据与影响 |
|---|---|---|
| P0/P1：未发现 | 本次范围内未发现会使 SFT、DPO 或 GRPO 目标明显计算错误的确定性缺陷。 | SFT mask、DPO policy/reference adapter 分离、GRPO old/reference 分离及损失梯度方向有相应单元测试和源码路径支持。 |
| P2：方法口径 | GRPO 截断响应按终止前缀正常评分和优化。 | `src/modeling.py:430–449`、`src/train.py:621–704`。报告应说明 cap/EOS 规则并呈现截断率；仅当实验原定排除截断样本时才升级为行为不符。 |
| P2：审计复现 | Online DPO 未在 rollout 日志保存 token IDs、行为 log-prob 和逐批生成参数。 | `src/train.py:449–489`。它不破坏 DPO 当前目标，但降低离线重建采样轨迹的能力；使用非默认模型时应连同其 generation config 保存。 |

## 已有测试覆盖与边界

`tests/test_objectives.py:46–115` 覆盖 SFT 提示屏蔽、EOS 计分、过长样本拒绝、移位后的 completion/padding mask；`117–127` 检查 DPO 梯度方向；`129–165` 检查总体标准差组优势、GRPO 的梯度方向、old/reference detach 及 reverse-KL 单点计算。它们未直接覆盖真实 PEFT 多 adapter 的切换与冻结、Online DPO 同一 rollout batch 内策略版本不变、GRPO 跨更新复用缓存策略概率、截断样本口径或多组变长响应的累积梯度等集成行为。根据主线提供的状态，CPU 与 SFT GPU 已有结果；本审查没有复跑这些测试，也不把单元目标测试视作 DPO/GRPO GPU 集成验证。

## 审查结论

在默认配置和当前实验口径下，代码实现了 SFT completion-only 监督、以 SFT LoRA 为固定参考的 DPO/Online DPO，以及区分 old/reference 策略概率的 GRPO。GRPO 两轮复用缓存样本符合 PPO 式更新结构，梯度累积在等响应均值损失下缩放正确。剩余需要在后续实验记录中固定的，是截断响应的终止处理以及 Online DPO 行为概率日志不足；二者均应作为解释和复现边界记录，不据此推断尚未运行的 DPO/GRPO GPU 结果。
