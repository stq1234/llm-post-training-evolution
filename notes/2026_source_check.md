# 2026 年论文来源核验

核验日期：2026-10-28。书目信息取自各论文 arXiv 原始摘要页；方法和实验范围取自相应 arXiv HTML 全文。以下均按当前可见的 arXiv 版本引用。现有 `survey_sources.json` 对这六条大多只记年份、题名和摘要层面的概述，重写时应补作者和具体版本日期；未在原始页面查到正式出版信息的，不标作会议或期刊论文。

## 1. Reinforcement-aware Knowledge Distillation for LLM Reasoning — [31] / S39

**原始来源：** [arXiv 摘要与版本记录](https://arxiv.org/abs/2602.22495)；[论文全文](https://arxiv.org/html/2602.22495)。

**书目：** Zhaoyang Zhang、Shuli Jiang、Yantao Shen、Yuting Zhang、Dhananjay Ram、Shuo Yang、Zhuowen Tu、Wei Xia、Stefano Soatto；2026-02-26 提交 v1，当前记录为 2026-06-17 修订的 v3。题名准确。页面提供 arXiv 版本记录；不要仅凭全文关键词“ICML”写成已发表 ICML 论文。

**可安全引用：** 作者提出 RL-aware distillation（RLAD），其中 TRRD 在学生自身 rollout 上使用 PPO/GRPO 式 clipped likelihood ratio，并以旧学生策略与教师策略的混合分布为锚点；方法不是简单追加教师 KL。作者在 K&K 逻辑推理及 Skywork-OR1 数学推理设置报告与 GRPO、KDRL 等基线的比较结果；可表述为“在论文给定模型、数据和训练条件下报告改进”，不可外推为所有蒸馏或所有推理任务的普遍优势。

**现有条目核对：** S39 对 student rollout、teacher/old mixture 和 trust-region 的概括准确。建议补明样例范围：逻辑实验包含 Qwen3-0.6B/1.7B 学生，教师为 Qwen3-8B；复杂数学实验使用 105K Skywork-OR1 训练样本及不同学生/教师配置，运行于 H200。摘要的“consistently outperforms”应保留为作者报告，不替代逐设置的数值比较。

## 2. Understanding Knowledge Distillation in Post-Training: When It Helps and When It Fails — [32] / S40

**原始来源：** [arXiv 摘要与版本记录](https://arxiv.org/abs/2606.22942)；[论文全文](https://arxiv.org/html/2606.22942)。

**书目：** Xin Liu、Simin Ma、Shujian Liu、Song Wang、Sathish Reddy Indurthi、Haoyun Deng、Lu Wang、Kaiqiang Song；2026-06-22 提交，当前为 v1。题名准确，当前原始记录是 arXiv 预印本版本。

**可安全引用：** 该研究在 Tülu 3 的 939K 指令数据上构造 10K 至全量的训练规模，比较 SFT 与 GKD，并在 BBH、GPQA、IFEval、InFoBench、MMLU-Pro 五项评测上报告结果：低数据区间（少于 80K）KD 优于 SFT，10K 设置的平均优势最高约 5 个百分点；数据增加后优势收窄，个别规模下 SFT 反超。该趋势依赖教师能力：作者另报告更强的 instruction-tuned 教师在大数据区间恢复收益，因此不能概括成“数据多时 KD 失效”。

**现有条目核对：** S40 的“Tülu 3 低数据时 KD 胜过 SFT、数据增加后差距缩小”是有依据的；重写应加上“该实验比较的教师/学生与五项基准范围”，并保留更强教师这一条件。论文还在低资源翻译、对话摘要、ARC-Challenge 评测两阶段合成数据策略；若正文讨论该策略，需单独限定为这些任务与本文具体设置。

## 3. FutureWorld: A Live Reinforcement Learning Environment for Predictive Agents with Real-World Outcome Rewards — [36] / S38

**原始来源：** [arXiv 摘要与版本记录](https://arxiv.org/abs/2604.26733)；[论文全文](https://arxiv.org/html/2604.26733)。

**书目：** Zhixin Han、Yanzhi Zhang、Chuyang Wei、Maohang Gao、Xiawei Yue、Kefei Chen、Yu Zhuang、Haoxiang Guan、Jiyan He、Jian Li、Yitong Duan、Yu Shi、Mengting Hu、Shuxin Zheng；2026-04-29 提交 v1，当前为 2026-05-15 修订的 v4。题名准确。arXiv 页面的 comments 仍注明代码将于之后发布、实验正在进行；应称作者报告的预印本结果，不称已独立复现或成熟通用平台。

**可安全引用：** FutureWorld 描述了一种按日收集真实事件预测问题、保存含搜索行动的 rollout、在预期结果可取得后回填 Brier 奖励并回放更新的闭环；实现基于修改后的 verl-tool-future 和 GRPO。作者报告三种开源 agent 的连续训练实验，并比较连续 8 天的检查点在同一组 500 个预测问题上的表现。该结果仅支持论文设置下的事件预测实验；计划时点仍无法检索到真值的问题会被丢弃，论文在五天观察中报告该比例平均为 35.65%，所以不应将结果写成广泛适用于长期代理任务的结论。

**现有条目核对：** S38 对延迟回填和重放的描述准确，特定事件预测域的边界也合适。建议补充日级延迟、未解析样本丢弃和 arXiv comments 中的代码/实验状态；避免据其抽象设计推断公开代码可复现性。

## 4. LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking — [37] / S42

**原始来源：** [arXiv 摘要与版本记录](https://arxiv.org/abs/2604.15149)；[论文全文](https://arxiv.org/html/2604.15149)。

**书目：** Lukas Helff、Quentin Delfosse、David Steinmann、Ruben Härle、Hikaru Shindo、Patrick Schramowski、Wolfgang Stammer、Kristian Kersting、Felix Friedrich；2026-04-16 提交 v1。题名准确；当前页面为 arXiv 预印本记录。

**可安全引用：** 在单一 SLR-Bench 域（基于火车/车辆标签的归纳逻辑规则学习）中，论文用 Isomorphic Perturbation Testing 比较普通外延 verifier 与保持逻辑同构关系的 verifier，报告部分 RLVR 模型使用逐实例标签枚举等捷径通过外延检查，而未学到可迁移规则。受控训练仅比较 OLMo-3-7B-Think-DPO 在两种奖励信号下各约 500 步的结果；这是关于该基准与该训练设置的证据，不足以断言所有 RLVR 或数学推理都发生 reward hacking。

**现有条目核对：** S42 对 SLR-Bench、标签枚举和 verifier 漏洞的描述基本准确；限制句“单一领域、模型系列与有限 7B 训练实验”有歧义，应改为“评测覆盖 GPT-5、OLMo-3、Ministral、GPT-4 等多种模型，但任务域仅为 SLR-Bench；因果控制训练只用 OLMo-3 7B”。论文明确把对数学、因果、溯因等其他推理域的泛化列为未解问题；对 GPT-5 系列只能黑盒评测。

## 5. Kimi K2.5: Visual Agentic Intelligence — [40] / S43

**原始来源：** [arXiv 摘要与版本记录](https://arxiv.org/abs/2602.02276)；[论文全文](https://arxiv.org/html/2602.02276)。

**书目：** arXiv 作者栏为 Kimi Team（包含大量署名贡献者）；2026-02-02 提交 v1，当前为 2026-08-07 修订的 v2。题名准确。宜按团队作者的技术报告/ arXiv 版本引用；正文中属于作者自报的模型与内部评估结果。

**可安全引用：** K2.5 报告描述在可验证推理/agentic 任务上的规则 outcome reward、通用任务上的 GRM、视觉任务专用 reward，以及 Agent Swarm 的 PARL 组合奖励（任务表现、子代理实例化/并行、完成率）。DEP 的“约为纯文本训练效率的 90%”是论文所述多模态训练基础设施结果；应注明这是作者给定系统和训练设定下的相对效率表述。

**现有条目核对：** S43 的奖励构成和 PARL 项目有正文支持，但必须分开说明不同机制的对象与证据层次。最重要的归属修正：报告所说 Toggle 平均减少 25–30% 输出 token，是在 **Kimi K2 Thinking** 上评估，不是 Kimi K2.5 本身的 token 减量结果（全文 §4.4）。可写成 K2.5 技术报告介绍/讨论的既有 token-efficient RL 方法；不得把这组数值归为 K2.5 实验。90% 则属于 K2.5 的 DEP 多模态训练效率描述，不能和 Toggle 数字并列成同一类 K2.5 效果。

## 6. On Language Drift during RLVR Post-Training — [42] / S41

**原始来源：** [arXiv 摘要与版本记录](https://arxiv.org/abs/2610.02015)；[论文全文](https://arxiv.org/html/2610.02015)。

**书目：** Michael Sullivan、Alexander Koller；2026-10-01 提交 v1。题名准确，为新近 arXiv 预印本；引用时应显式注明版本日期。

**可安全引用：** 理论部分在论文的形式化设置中假设 outcome-only、最终答案正确性二元奖励，并将 RLVR 与 SFT 训练过程建模为单调改进的 checkpoint 序列；在这些定义和假设下，作者证明 RLVR 的可达语言漂移没有有限上界，并给出限制漂移会限制最大期望奖励的定理。实证部分只使用 1–1.5B 的 Llama、Gemma、Qwen 系列模型和 GSM8K 一种带 CoT 数据，比较 SFT/RLVR 共 30 次训练；Llama/Gemma 观察到更明显漂移，Qwen 的收益主要来自行为 sharpen（学会在答案后停止），相对未显著漂移。

**现有条目核对：** S41 的题名、日期、理论/实验类别及“不作领域共识”的边界正确，但“漂移约束与奖励有权衡”必须注明是理论结论而非实证结果：作者明确表示未对该定理做经验验证。不要写成所有 RLVR 都必然导致漂移、前沿模型实验已验证该权衡，或漂移约束已被证明必然降低现实模型表现；论文称 RLVR permits drift，实证条件是任务超出基础模型已有行为（novel task）。

## 给重写稿的书目与用语核对清单

1. 六条 arXiv 链接和题名均与原始页面相符；`survey_sources.json` 当前记录年份但未完整记录作者、精确提交/修订日期和版本。书目可用“第一作者 et al.”，Kimi 用团队作者 Kimi Team，并在需要时标明 v1/v2/v3/v4。
2. 将所有 2026 来源描述为“作者在该版本报告/提出”，而非领域定论。arXiv 版本记录提供日期，但不能据其自行推断同行评审状态；无确证时不添加会议录信息。
3. 必须修正两处易误归属：Kimi K2 Thinking 的 25–30% Toggle 结果不能归为 K2.5；Verifier Gaming 的单一限制是任务域，不是所评模型只有一个系列。
4. 对 KD 和 language drift 保留条件：KD 的大数据收益取决于教师是否提供额外知识；language-drift 的理论与其一数据集、小模型实证不可互换。
