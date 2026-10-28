# 实验报告非结果章节独立审读

> 历史审读记录：本检查发生在结果章节完成前。以下“未读取结果区”和逐行位置反映当时版本，行号不用于定位当前报告。

## 范围与结论

只读检查了当前根目录 `实验报告.md` 的方法、数据、设置、讨论边界和参考文献部分（当时未读取结果区；当时稿件尚无 Results 章节）、`src/train.py`、`src/modeling.py`、`docs/LOCAL_METHOD_REVIEW.md` 及 `configs/experiment.json`。未读取 `results/` 或 `tests/`，未运行训练、评估、测试或任何 GPU 工作；未改报告、源码或配置。

总体结论：协议参数和本地 DPO/Online DPO/GRPO 的主要计算路径表述大体对应配置与实现，特别是固定 SFT reference、old/reference 概率分工、GRPO 组总体标准差、逐回答损失归约、两次复用 rollout 及截断处理均有明确说明。未发现需要停止正在运行实验的代码—方法严重缺陷。但提交前应修正一处确定的预处理事实错误，并澄清 SFT 损失的实际归约，否则报告公式会暗示不同的长度加权目标；另有两项理论/复现表述应收紧。

## 应提交前修正的问题

### P2：把右侧 padding 写成左侧 padding

- 历史报告位置：`实验报告.md:282`，“训练序列采用左侧 padding 对齐批次”。
- 实现证据：`src/modeling.py:124–140` 的 `collate_features` 在 `input_ids`、`labels` 和 `response_mask` 末尾追加 padding，并构造 `attention_mask=[1]*n+[0]*padding`，这是右侧 padding。`docs/LOCAL_METHOD_REVIEW.md` 的 SFT mask 段也明确记为右 padding。
- 建议：将“左侧”改为“右侧”。提示/填充不计入 response mask 的其他说明与代码相符，无需改动。

### P2：SFT 展示的 token 求和式未说明实际按样本长度平均

- 历史报告位置：`实验报告.md:38–44` 将 SFT 写为 `-∑_t log πθ(y_t|...)`；`286` 及表 5 第 `346–347` 行称 completion-only token NLL，但未说明本地样本归约。
- 实现证据：`src/train.py:294–299` 使用 `F.cross_entropy(..., ignore_index=-100)` 的默认 mean reduction。配置 batch size 为 1，所以每个 microbatch 的 loss 是该条回答有效 token 上的均值；梯度累积再将各 microbatch loss 等权相加并除以 4。实际每步目标是四条有放回抽取回答的“逐回答 token 均值”的平均，而不是每条回答 token 负对数似然之和，也不是将四条回答所有 token 合并后统一求均值。回答长短因而不会按有效 token 数线性增权。
- 建议：保留式 (1) 作为序列负对数似然的定义时，应紧接着区分训练 reduction，并给出实际目标：
  `L_SFT = E_(x,y)[(1/T_y) Σ_t -log πθ(y_t | x,y_<t)]`，再注明 batch size 1 下梯度累积对每条样本均值等权平均。这样和代码完全对应，不需改实现或协议。

## 需要收紧的理论与复现论述

### P2/P3：DPO 的偏好模型假设未在方法中交代

- 历史报告位置：`实验报告.md:66–75` 和 `288–295` 给出正确的 reference-adjusted logistic DPO 目标，也指出偏好对为规则合成、不是人类标注；但没有说明理论目标将偏好标签建模为 latent reward 差的 Bradley–Terry/logistic 选择概率，并以 reference-relative KL-regularized reward-optimal policy 参数化。
- 建议：在首个 DPO 公式前用一两句列明这些是 DPO 推导的建模前提，而非本实验已验证的事实；补充就近的 `[9]` 引用。继而说明本研究确定性规则标签只是这些偏好对目标的操作化代理，不能据此声称满足人类偏好模型。公式不需改写。

### P2/P3：rollout 日志能核查文本样本，但不能精确重放采样 token 序列

- 报告位置：表 2 第 `158` 行说可“以运行日志和逐样本 rollout 记录确认各方法消费的训练数据”；`362–363` 又要求报告逐样本/截断记录。
- 实现证据：`src/train.py:449–468` 将 Online DPO 的候选文本、奖励、索引和截断标记写入 JSONL；GRPO 在 `652–660` 还写入候选的 old/reference 序列 log-prob 汇总值和优势。候选 `completion_ids` 只在内存中用于构造 loss，没有被这些 rollout 记录持久化。`docs/LOCAL_METHOD_REVIEW.md` 已指出 Online DPO 缺少 token IDs、逐 token 行为概率和逐批生成配置。
- 建议：将表 2 该行限缩为“日志可核对提示、解码文本、奖励和截断记录”；如称可精确审计或重放轨迹，则须另存 token IDs 和生成/RNG 状态。也可指出 GRPO 磁盘日志只存 old/reference 序列汇总值，逐 token 缓存仅在训练进程内使用。报告第 `299` 行已经准确说明 Online DPO 的限制，建议把范围说明扩展到 GRPO，避免读者误以为 GRPO rollout 可从文件精确重建。

## 其他核对结果

- **最终配置与方法预算一致。** `configs/experiment.json` 的 SFT 16 步、DPO 64 步/累积 4、Online DPO 累积 2/至多 128 prompt、GRPO 128 步/累积 2/每批复用两次，以及 `group_size=4`、reward、clip、beta 和优势 epsilon 均与表 5 相符。seed 17 是文件默认值，报告的 seed 29 属于运行时 seed 覆盖；`sft_long` 256 步 seed 17 单次控制与主矩阵分开报告也符合锁定协议。
- **在线组数核算正确。** Online DPO 每次最多生成两个 prompt 组，每组四个候选，128 个 prompt 上限对应至多 512 个候选；并列组消耗 prompt budget、使有效更新数少于 64 的限定也与 `train_online_dpo` 相符。GRPO 累积两个 prompt 组、每组四答、每份 rollout 连续优化两次，128 次更新对应 64 份 rollout、128 个 prompt 组和 512 个候选。报告已正确避免把这与 Online DPO 或静态 DPO 说成等生成/等计算量比较。
- **GRPO 目标和 mask 说明基本吻合。** `group_advantages` 采用 `unbiased=False` 的总体标准差并加 epsilon；`grpo_loss` 以 old log-prob 作 ratio 分母、以 frozen reference 构造采样 KL，逐回答按有效 token 数平均后再对回答平均。新 rollout 仅在两次更新复用边界重新生成。代码与报告都明确 EOS 计分；达到上限未出 EOS 的前缀仍作为终止样本接受奖励和策略/KL 损失，非 bootstrap/排除口径已被披露。
- **DPO 计算叙述正确。** policy/reference 均为对应 seed 的 SFT adapter，reference 冻结； chosen/rejected 按响应 token log-prob 总和计算，含 EOS、屏蔽 prompt/padding；在线 DPO 用当前策略采样并按规则奖励极值配对，但目标不使用 PPO old-policy ratio。
- **八种实现的比较达到实质性代码分析深度。** 总表之后逐个说明 TRL、trlx、DeepSpeed-Chat、OpenRLHF、verl、slime、open-r1 与 MiniMind 的数据/rollout、概率对象、掩码/归约或系统边界，并以固定版本源码链接支撑。鉴于本作业明确要求八实现代码比较，这部分整体不属于空泛逐篇摘要。表 2 的恢复日志、样本留存等内容略偏工程记录；建议仅在篇幅需要时压缩与算法目标、梯度尺度或复现实验解释无直接关系的恢复细节，保留版本锁定、归约、old/reference 和采样边界等核心分析。
- **未见应移出报告的明显审计元数据。** 当前审读范围未见文件 hash、逐文件字节数、机器绝对路径、调试日志、提交/验收操作记录。精确源码 revision 用于来源定位和复现，适合留在引用中。

## 提交建议

先修正右 padding 错字及 SFT reduction 描述；补足 DPO 偏好模型假设与就近文献引用；收紧 rollout 日志可复现范围。其余非结果章节没有发现需要阻止后续正式结果撰写的重大数学或协议矛盾。以上判断是静态文本—源码—配置审读，不构成运行结果、GPU 集成验证或数据生成器核验。
