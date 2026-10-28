# 来源证据对照

本表将正文数字引文与原始来源逐条对应。证据描述为对已阅读论文、技术报告或官方项目材料的概述；适用范围和版本差异列在边界项。2026 年预印本按研究报告结论陈述，不作为已定论。

## [1] Training Language Models to Follow Instructions with Human Feedback (2022)

**来源：** [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)；来源编号 S03。

**支持的主张：** InstructGPT 建立 SFT、偏好奖励建模与 PPO 的分阶段 RLHF；特定提示分布下 1.3B 模型比 175B GPT-3 更受偏好。

**核验到的证据：** 论文 §§3–4 描述示范、成对比较和未标注提示数据，以及 SFT、RM、PPO/KL、PPO-ptx 流程；评价章节报告 1.3B 人类偏好结果。

**边界与版本：** 性能主张限定其 API 提示分布、评审方式和实验模型。

## [2] Learning to Summarize from Human Feedback (2020)

**来源：** [https://arxiv.org/abs/2009.01325](https://arxiv.org/abs/2009.01325)；来源编号 S01。

**支持的主张：** 偏好比较可用于拟合摘要奖励模型，再以强化学习优化策略。

**核验到的证据：** 原论文摘要和方法先收集人类比较、拟合奖励模型，再以强化学习微调摘要模型。

**边界与版本：** 2020 年前史研究，任务为摘要。

## [3] WebGPT: Browser-assisted Question-answering with Human Feedback (2021)

**来源：** [https://arxiv.org/abs/2112.09332](https://arxiv.org/abs/2112.09332)；来源编号 S02。

**支持的主张：** 浏览器辅助问答可把检索行动、证据引用与回答质量纳入人类反馈训练；拒绝采样可用奖励模型筛选候选。

**核验到的证据：** 论文描述浏览器交互示范、模仿学习、奖励模型、人类偏好评价与筛选候选回答的实验。

**边界与版本：** 仅支持论文定义的浏览器问答任务。

## [4] Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback (2022)

**来源：** [https://arxiv.org/abs/2204.05862](https://arxiv.org/abs/2204.05862)；来源编号 S04。

**支持的主张：** 帮助性与无害性可以作为可分别讨论的人类偏好目标。

**核验到的证据：** 论文描述依据人类偏好训练 helpful/harmless 助手，并讨论两种目标间的权衡。

**边界与版本：** 不能把特定安全评测视为全面安全保证。

## [5] Constitutional AI: Harmlessness from AI Feedback (2022)

**来源：** [https://arxiv.org/abs/2212.08073](https://arxiv.org/abs/2212.08073)；来源编号 S05。

**支持的主张：** 宪法原则可用于自我批评和修订示范，AI feedback 可用于偏好奖励与 RL。

**核验到的证据：** 论文方法描述原则驱动的自我批评/修订，以及基于 AI 反馈的偏好建模和强化学习。

**边界与版本：** 原则仍需由设计者选定，反馈质量受模型评判能力影响。

## [6] Scaling Laws for Reward Model Overoptimization (2022)

**来源：** [https://arxiv.org/abs/2210.10760](https://arxiv.org/abs/2210.10760)；来源编号 S06。

**支持的主张：** 优化不完美代理奖励过度可能使黄金奖励停滞或下降。

**核验到的证据：** 合成偏好实验改变奖励模型规模、训练数据与优化程度，报告代理分数继续升高时 gold reward 可能回落，并分析 KL 正则。

**边界与版本：** 证据来自论文指定的摘要偏好和合成评价设置。

## [7] Direct Preference Optimization: Your Language Model is Secretly a Reward Model (2023)

**来源：** [https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)；来源编号 S07。

**支持的主张：** DPO 在 Bradley–Terry 与 KL 正则假设下把奖励表示为相对参考策略对数概率，并用离线偏好分类目标训练。

**核验到的证据：** 论文从 KL 约束最优策略推导隐式奖励和成对损失，实验比较摘要、情感控制等设置下 DPO 与 PPO-RLHF。

**边界与版本：** 理论依赖偏好似然假设；实验不证明所有任务普遍优于显式 RL。

## [8] SLiC-HF: Sequence Likelihood Calibration with Human Feedback (2023)

**来源：** [https://arxiv.org/abs/2305.10425](https://arxiv.org/abs/2305.10425)；来源编号 S08。

**支持的主张：** SLiC-HF 将偏好信息转成序列似然校准/排序约束。

**核验到的证据：** 论文方法定义 likelihood calibration 与偏好目标，在摘要等任务报告人评、自动指标和 RLHF 基线对比。

**边界与版本：** 不同模型、数据及摘要设置不能直接作横向结论。

## [9] RRHF: Rank Responses to Align Language Models with Human Feedback without Tears (2023)

**来源：** [https://arxiv.org/abs/2304.05302](https://arxiv.org/abs/2304.05302)；来源编号 S09。

**支持的主张：** RRHF 用多来源响应排序和排名损失提高偏好响应的似然。

**核验到的证据：** 论文方法收集同一提示的多个回答，按偏好/奖励排序并使用 ranking loss 结合语言建模训练。

**边界与版本：** 效果受候选响应来源和质量限制。

## [10] A General Theoretical Paradigm to Understand Learning from Human Preferences (2023/2024)

**来源：** [https://arxiv.org/abs/2310.12036](https://arxiv.org/abs/2310.12036)；来源编号 S10。

**支持的主张：** ΨPO 框架比较 RLHF、DPO 与 IPO；IPO 使用有限目标以抑制特定有限数据下的过度偏好间隔。

**核验到的证据：** 论文理论框架推导不同偏好优化目标，分析 IPO 与 DPO/RLHF，并讨论偏好数据过拟合。

**边界与版本：** arXiv 2023 首版、AISTATS 2024 发表；理论结论依其有限数据假设。

## [11] Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms (2024)

**来源：** [https://arxiv.org/abs/2406.02900](https://arxiv.org/abs/2406.02900)；来源编号 S26。

**支持的主张：** DPO 等直接偏好方法也可能因过度优化及未覆盖区域外推失准而退化。

**核验到的证据：** 论文比较策略偏移与外部偏好分数，并用树状 MDP 分析直接对齐隐式奖励的 OOD 外推。

**边界与版本：** 证据覆盖论文模型与简化问题设置。

## [12] KTO: Model Alignment as Prospect Theoretic Optimization (2024)

**来源：** [https://arxiv.org/abs/2402.01306](https://arxiv.org/abs/2402.01306)；来源编号 S19。

**支持的主张：** KTO 可由单回答 desirable/undesirable 标签训练，不要求偏好对。

**核验到的证据：** 论文给出 prospect-theoretic utility 目标和正负单回答 KTO 损失及实验。

**边界与版本：** 效果依具体标签、模型规模、数据和评价设定。

## [13] ORPO: Monolithic Preference Optimization without Reference Model (2024)

**来源：** [https://arxiv.org/abs/2403.07691](https://arxiv.org/abs/2403.07691)；来源编号 S20。

**支持的主张：** ORPO 将 chosen 回答的 SFT 负对数似然与 chosen/rejected odds-ratio 偏好损失合并为单阶段目标；方法不需要单独的 SFT warm-up，也不使用独立参考模型。

**核验到的证据：** 原文第 4.2 节将目标写为 SFT loss 与 odds-ratio loss 的加权和；引言称无需 SFT warm-up。实验章节报告从预训练 Phi-2、Llama-2、Mistral 检查点直接使用偏好数据训练的设置。

**边界与版本：** EMNLP 2024 版本；另有工作为控制比较将 ORPO 应用于 SFT 检查点，因此“无 SFT warm-up”是方法无需前置阶段，不是每个实验都禁止采用 SFT 初始化。论文特定设置不能决定普遍优劣。

## [14] SimPO: Simple Preference Optimization with a Reference-Free Reward (2024)

**来源：** [https://arxiv.org/abs/2405.14734](https://arxiv.org/abs/2405.14734)；来源编号 S21。

**支持的主张：** SimPO 以长度归一化平均 log-prob 作为隐式奖励并加偏好目标间隔；原论文报告的 Base 与 Instruct 配置均从 SFT 检查点开展偏好训练。

**核验到的证据：** 论文给出 reference-free objective 与 length-normalization 消融；实验设置明确说明 Base 配置先在 UltraChat 上训练 SFT 模型，Instruct 配置使用现成指令模型作为 SFT 检查点。

**边界与版本：** 奖励定义不含显式参考策略项，但这不意味着训练不使用 SFT 初始化；论文比较 Base 与 Instruct 配置并指出初始化质量会影响结果。预印本 v3（2024-11-01）。

## [15] Direct Language Model Alignment from Online AI Feedback (2024)

**来源：** [https://arxiv.org/abs/2402.04792](https://arxiv.org/abs/2402.04792)；来源编号 S22。

**支持的主张：** OAIF 每轮从当前策略采样候选，再由语言模型判别器给出偏好。

**核验到的证据：** 论文描述在线 AI feedback 流程，并报告多任务人评与离线 DAP/RLHF 基线比较。

**边界与版本：** AI judge 质量依赖其提示和校准。

## [16] Human Alignment of Large Language Models through Online Preference Optimisation (2024)

**来源：** [https://arxiv.org/abs/2403.08635](https://arxiv.org/abs/2403.08635)；来源编号 S23。

**支持的主张：** 在线 IPO 与偏好博弈存在理论关系；在线 DPO 与在线 IPO 一般并非同一目标。

**核验到的证据：** 论文推导在线 IPO 动力学/驻点并比较 DPO 与 IPO 在偏好设置下的目标差异。

**边界与版本：** ICML 2024 论文结论需按其理论条件理解。

## [17] Self-Rewarding Language Models (2024)

**来源：** [https://arxiv.org/abs/2401.10020](https://arxiv.org/abs/2401.10020)；来源编号 S25。

**支持的主张：** 同一语言模型可用作 judge 产生自奖励，再以迭代 DPO 训练。

**核验到的证据：** 论文报告判别提示自评分数、迭代 DPO 和特定基准的指令跟随/自评分变化。

**边界与版本：** 生成器与 judge 可能共享盲点；结果不构成普遍自改进保证。

## [18] Self-Play Preference Optimization for Language Model Alignment (2024)

**来源：** [https://arxiv.org/abs/2405.00675](https://arxiv.org/abs/2405.00675)；来源编号 S24。

**支持的主张：** SPPO 通过自博弈向近似偏好均衡更新；实验使用偏好 oracle。

**核验到的证据：** 论文构造 self-play objective 和均衡分析；实验采用 PairRM 等偏好模型评价策略。

**边界与版本：** 不能表述为完全无外部评价器；其声称不依赖 GPT-4 补充回答/偏好标签。

## [19] UltraFeedback: Boosting Language Models with Scaled AI Feedback (2023)

**来源：** [https://arxiv.org/abs/2310.01377](https://arxiv.org/abs/2310.01377)；来源编号 S11。

**支持的主张：** UltraFeedback 以大规模多维 AI 反馈构造偏好训练数据。

**核验到的证据：** 数据论文报告约 250k 指令及超过百万条 GPT-4 反馈评分/解释，并说明据此筛选和训练模型。

**边界与版本：** AI 反馈不等同于独立人类评价。

## [20] RLAIF: Scaling Reinforcement Learning from Human Feedback (2023)

**来源：** [https://arxiv.org/abs/2309.00267](https://arxiv.org/abs/2309.00267)；来源编号 S12。

**支持的主张：** AI 标注器可用于 RLAIF 并可与人类反馈在特定任务上直接比较。

**核验到的证据：** 论文在摘要、helpful/harmless 对话任务比较人类与 PaLM 2 AI 标注器，使用策略梯度更新并报告评价。

**边界与版本：** 结论只覆盖该模型、提示、任务和基线。

## [21] Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing (2024)

**来源：** [https://arxiv.org/abs/2406.08464](https://arxiv.org/abs/2406.08464)；来源编号 S29。

**支持的主张：** Magpie 通过对齐模型模板续写合成大规模指令数据并筛选子集。

**核验到的证据：** 论文摘要说明，以模板前缀诱导已对齐的 Llama-3-Instruct 自行生成 400 万条指令及其对应回答，再从中选择 30 万条高质量实例；比较实验使用筛选后的数据微调 Llama-3-8B-Base。

**边界与版本：** 合成指令分布继承生成教师和筛选过程；论文报告的是 400 万条指令及对应回答、筛选后的 30 万条实例。

## [22] Let’s Verify Step by Step (2023)

**来源：** [https://arxiv.org/abs/2305.20050](https://arxiv.org/abs/2305.20050)；来源编号 S13。

**支持的主张：** 过程监督在论文所测 MATH 子集上优于结果监督；PRM800K 提供步骤级标签。

**核验到的证据：** 论文比较 outcome 与 process supervision 的数学筛选表现，并介绍发布约 800k 条步骤级人类反馈。

**边界与版本：** 结果限定特定模型、标注协议与 MATH 子集。

## [23] Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations (2023)

**来源：** [https://arxiv.org/abs/2312.08935](https://arxiv.org/abs/2312.08935)；来源编号 S14。

**支持的主张：** 自动生成逐步监督可训练过程奖励模型，并用于候选重排或 PPO。

**核验到的证据：** 论文介绍 Math-Shepherd 的自动逐步标注、PRM 训练及重排序/PPO 应用实验。

**边界与版本：** 自动标签和 verifier 也可能带误差。

## [24] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (2024)

**来源：** [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)；来源编号 S27。

**支持的主张：** GRPO 对同一问题采样多条回答，以组内奖励均值/标准差形成相对优势，可省略单独 critic。

**核验到的证据：** DeepSeekMath 定义 GRPO、组采样、相对优势及 KL 项，并在数学任务进行训练比较。

**边界与版本：** 这是策略优化器；结果依赖论文数据、模型与奖励函数。

## [25] Tülu 3: Pushing Frontiers in Open Language Model Post-Training (2024/2025)

**来源：** [https://arxiv.org/abs/2411.15124](https://arxiv.org/abs/2411.15124)；来源编号 S28。

**支持的主张：** Tülu 3 组合 SFT、DPO 和 PPO 式 RLVR，使用确定性 verifier 评价答案及约束。

**核验到的证据：** 技术报告详述开源配方、数据、训练与去污染评价；RLVR 阶段在可核验任务调用程序校验器。

**边界与版本：** arXiv 2024、COLM 2025 版本；结果不可当作跨论文算法常数。

## [26] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2025)

**来源：** [https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)；来源编号 S34。

**支持的主张：** R1-Zero 从基础模型直接用准确性/格式规则奖励 RL；R1 加冷启动和多阶段训练，并蒸馏约 800k 轨迹。

**核验到的证据：** 技术报告分别列出两个配方、训练阶段和观察到的可读性/语言混杂问题；蒸馏章节报告轨迹量与学生规模。

**边界与版本：** 作者对“涌现”行为的描述是单一报告观察，不是普遍定理。

## [27] Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? (2025)

**来源：** [https://arxiv.org/abs/2504.13837](https://arxiv.org/abs/2504.13837)；来源编号 S36。

**支持的主张：** 部分 RLVR 的 pass@1 提升可能来自对基础模型已有轨迹的概率重分配。

**核验到的证据：** 论文在数学与视觉推理设置比较大采样量 pass@k，并报告高 k 下基础模型与 RL 模型差异。

**边界与版本：** 这是研究解释，受模型、基准与估计预算限制。

## [28] DAPO: An Open-Source LLM Reinforcement Learning System at Scale (2025)

**来源：** [https://arxiv.org/abs/2503.14476](https://arxiv.org/abs/2503.14476)；来源编号 S35。

**支持的主张：** DAPO 报告 Clip-Higher、动态采样、token-level loss 和超长响应处理等改动；在论文 AIME24/Qwen2.5-32B 设置报告 avg@32 50。

**核验到的证据：** 论文列出方法组件、逐步消融及与 R1-Zero-Qwen-32B 和朴素 GRPO 对比。

**边界与版本：** 数值受题集、温度、32 次采样和奖励规则限制。

## [29] Orca: Progressive Learning from Complex Explanation Traces of GPT-4 (2023)

**来源：** [https://arxiv.org/abs/2306.02707](https://arxiv.org/abs/2306.02707)；来源编号 S18。

**支持的主张：** GPT-4 复杂解释轨迹可作为小模型的指令微调/蒸馏数据。

**核验到的证据：** Orca 报告以 GPT-4 解释轨迹和任务执行痕迹训练小模型，并在多项基准评估。

**边界与版本：** 学生基准表现不证明教师相同的推理机制或泛化能力。

## [30] DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales (2023)

**来源：** [https://arxiv.org/abs/2308.01320](https://arxiv.org/abs/2308.01320)；来源编号 S30。

**支持的主张：** 端到端 RLHF 系统要组织 SFT、奖励建模和 PPO 训练流水线，系统实现影响规模和开销。

**核验到的证据：** DeepSpeed-Chat 论文介绍阶段流程、ZeRO 等实现及多尺度性能测试。

**边界与版本：** 吞吐和扩展性数字取决于硬件、模型和并行配置。

## [31] Reinforcement-aware Knowledge Distillation for LLM Reasoning (2026)

**来源：** [https://arxiv.org/abs/2602.22495](https://arxiv.org/abs/2602.22495)；来源编号 S39。

**支持的主张：** Reinforcement-aware KD 在学生当前 rollouts 上使用教师信号与 trust-region 约束。

**核验到的证据：** 2026 预印本提出 on-policy 蒸馏目标并报告推理后训练实验。

**边界与版本：** 新近预印本；证据依论文模型与任务。

## [32] Understanding Knowledge Distillation in Post-Training: When It Helps and When It Fails (2026)

**来源：** [https://arxiv.org/abs/2606.22942](https://arxiv.org/abs/2606.22942)；来源编号 S40。

**支持的主张：** Tülu 3 数据上的实验报告低数据量时 KD 相对 SFT 收益明显，增加数据后差距收窄。

**核验到的证据：** 2026 预印本比较数据量、教师能力及 SFT/KD 后训练结果。

**边界与版本：** 新近预印本，不能外推为普遍蒸馏规律。

## [33] Toolformer: Language Models Can Teach Themselves to Use Tools (2023)

**来源：** [https://arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761)；来源编号 S15。

**支持的主张：** 工具使用训练学习何时调用 API、传何参数以及如何使用结果。

**核验到的证据：** Toolformer 论文用少量 API 示例生成调用候选，再以自监督损失筛选有益调用，测试多种工具。

**边界与版本：** 结论限于其工具集合和测试任务。

## [34] SWE-bench: Can Language Models Resolve Real-World GitHub Issues? (2023)

**来源：** [https://arxiv.org/abs/2310.06770](https://arxiv.org/abs/2310.06770)；来源编号 S17。

**支持的主张：** SWE-bench 把真实 GitHub issue 转成代码库级修复任务。

**核验到的证据：** 论文说明由 12 个 Python 仓库的 2,294 个问题构成，使用 issue 和测试评估模型补丁。

**边界与版本：** 是代码环境任务，不等同偏好训练结果。

## [35] τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains (2024)

**来源：** [https://arxiv.org/abs/2406.12045](https://arxiv.org/abs/2406.12045)；来源编号 S33。

**支持的主张：** τ-bench 评价带用户、工具和领域规则的多轮代理行为，并用 pass^k 衡量重复成功可靠性。

**核验到的证据：** 论文构造航空/零售环境、用户模拟和数据库状态检查，定义多次运行指标。

**边界与版本：** 模拟环境和领域覆盖有限。

## [36] FutureWorld: A Live Reinforcement Learning Environment for Predictive Agents with Real-World Outcome Rewards (2026)

**来源：** [https://arxiv.org/abs/2604.26733](https://arxiv.org/abs/2604.26733)；来源编号 S38。

**支持的主张：** FutureWorld 保存预测轨迹，待现实事件结果出现后回填 outcome reward 并用于更新。

**核验到的证据：** 2026 原始预印本描述 live event prediction 环境和延迟结果奖励回放机制。

**边界与版本：** 特定事件预测域，尚不能说明通用长期代理 RL 已成熟。

## [37] LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking (2026)

**来源：** [https://arxiv.org/abs/2604.15149](https://arxiv.org/abs/2604.15149)；来源编号 S42。

**支持的主张：** 特定归纳规则实验显示，verifier 覆盖漏洞可能让部分 RLVR 策略采用标签枚举捷径。

**核验到的证据：** SLR-Bench 分别测试 extensionally/isomorphically verifier，报告受控模型上的捷径行为。

**边界与版本：** 作者限定单一领域、模型系列与有限 7B 训练实验。

## [38] WebArena: A Realistic Web Environment for Building Autonomous Agents (2023/2024)

**来源：** [https://arxiv.org/abs/2307.13854](https://arxiv.org/abs/2307.13854)；来源编号 S16。

**支持的主张：** WebArena 提供多个可自托管网站以评估长程网页任务；论文报告 GPT-4 成功率 14.41%、人类 78.24%。

**核验到的证据：** 论文定义网站环境和 812 项任务，并在结果中报告 GPT-4 代理与人类基线的端到端成功率。

**边界与版本：** 数值只对应论文环境版本、代理和基线设置。

## [39] The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity (2025)

**来源：** [https://arxiv.org/abs/2506.06941](https://arxiv.org/abs/2506.06941)；来源编号 S37。

**支持的主张：** 可控组合谜题实验观察到推理模型表现随复杂度变化，并在高复杂度区出现失效。

**核验到的证据：** 论文改变任务复杂度和推理 token 预算，分析不同模型的成功率和输出行为。

**边界与版本：** 仅说明论文的谜题任务族，不概括全部推理能力。

## [40] Kimi K2.5: Visual Agentic Intelligence (2026)

**来源：** [https://arxiv.org/abs/2602.02276](https://arxiv.org/abs/2602.02276)；来源编号 S43。

**支持的主张：** Kimi K2.5 联合文本—视觉 RL、生成式奖励、可验证结果、视觉任务奖励与智能体并行调度奖励。

**核验到的证据：** 2026 原始技术报告 §§4.4–4.5、§5.2 描述多来源 SFT、规则/GRM/视觉/预算奖励、PARL 的 performance/parallel/finish 奖励、Toggle 与 DEP。

**边界与版本：** 作者技术报告和内部基准；25%–30% token 降幅、90% 训练效率等结果限该报告设定。

## [41] HybridFlow: A Flexible and Efficient RLHF Framework (2024)

**来源：** [https://arxiv.org/abs/2409.19256](https://arxiv.org/abs/2409.19256)；来源编号 S32。

**支持的主张：** HybridFlow 将训练与生成控制、资源布局和权重切换作为系统设计对象。

**核验到的证据：** 论文提出 hybrid controller 与 3D-HybridEngine，并评估生成/训练阶段的资源调度。

**边界与版本：** 系统收益依集群、模型和 workload 设置。

## [42] On Language Drift during RLVR Post-Training (2026)

**来源：** [https://arxiv.org/abs/2610.02015](https://arxiv.org/abs/2610.02015)；来源编号 S41。

**支持的主张：** RLVR 训练可能造成文本推理语言形式漂移，漂移约束与奖励有权衡。

**核验到的证据：** 2026 年 10 月原始预印本提供理论分析与文本推理模型实验。

**边界与版本：** 范围限于论文特定文本推理任务/模型；不作为领域共识。

## [43] OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework (2024)

**来源：** [https://arxiv.org/abs/2405.11143](https://arxiv.org/abs/2405.11143)；来源编号 S31。

**支持的主张：** OpenRLHF 协调训练、推理及多模型组件，支持 RLHF 与偏好优化流程。

**核验到的证据：** 论文描述基于 Ray、vLLM、DeepSpeed 的架构与 RLHF、DPO、拒绝采样接口。

**边界与版本：** 以论文代码版本描述能力；实现版本会演进。
