# 最终实验结果报告独立核验

## 范围

对照当前版本 `实验报告.md` 的摘要、表 6–11、案例、讨论与结论，以及 `results/analysis/summary.json`、相关 CSV、`results/analysis/error_cases.jsonl` 和 11 个原始 `run_summary.json`。只读检查并查看了报告引用的图文件；未浏览网页、运行代码/测试或使用 GPU，也未更改报告、代码或数据。本结论对应本次核验时的工作区版本。

## 结论

摘要、表 6–11、主要结果叙述和结论中的数值及 seed 汇总口径，与当前 analysis summary/CSV 和原 run summary 总体一致。joint 正确定义、baseline/SFT-256 的单次运行标记、两种子结果的均值及 min–max、DPO 负结果、训练/测试截断分母、DPO log-prob margin 和资源峰值均通过抽查。案例描述与 `error_cases.jsonl` 对应；所引用的图文件存在，图号、图注顺序一致；Markdown 中恰有 17 个完整的 display-math 区块。

当前需协调一项反事实指标的内部口径：报告在数据说明和指标表承诺“答案是否随事实变化”，但结果只报告 80 对的双题 joint 成功率，且结果章节明确称其不量化答案翻转一致性。该指标不在当前 `counterfactual_by_seed.csv` 中。它是报告内部范围承诺与已呈现分析的缺口；本审读不将其归为外部课程规范。报告可纳入单独补算的 answer-flip 诊断，并注明严格答案/证据联合正确仍是另一指标。

## 核验记录

### 表 6、表 7、摘要及 DPO/GRPO 结果叙述

- 摘要和表 6 的三组划分 joint 均值、双种子 min–max 均与 `metrics_seed_mean_range.csv` 相符。三个测试集均为每 seed 160 条；分析清单显示预测文件完整、无缺失或不完整评估。
- 报告的 n 口径一致：SFT、DPO、在线 DPO、GRPO-full 与 GRPO-format 均为 seed 17/29；baseline 与 SFT-256 为单次运行。表 6/7 明确区分了这些条件，表 9 的完成种子列也显示相应的 1 或 2。
- joint 定义写为合法 JSON、答案字符串正确、证据编号集合与必要证据精确匹配；这与 analysis summary 中 strict 主指标相符。围栏宽松解析只作为诊断单列，没有混入主指标。
- 表 7 的四类任务 joint 均值和范围匹配 summary。讨论中“主方法两步关系/加法均不高于 0.100，SFT-256 除外”与该表相符；不可回答任务达到 1.000 的方法也列述正确。
- DPO 相对同 seed SFT 的 joint 差分别为 −0.13750、−0.04375、−0.16250，六个 seed×split 差均为负；正文取三位小数后的 −0.138、−0.044、−0.162 与 analysis CSV 一致。完整奖励 GRPO 与格式奖励 GRPO 的均值差 0.071875、0.03125、0.096875，正文的 0.072、0.031、0.097 正确，且两个 seed 在三划分上的差都为正。
- 基线围栏诊断的宽松答案正确率 0.400/0.156/0.350、围栏规范化 joint 0.044/0.006/0.200 与 summary 相符；验证集 91/96 条有外层围栏也与 analysis manifest 相符。

### 表 8 案例

表 8 的四例都能在 `error_cases.jsonl` 找到相同方法、种子、任务、gold 和预测：baseline 的围栏 JSON；SFT-17 两步题漏掉证据 3 并多引 4；DPO-17 不可回答保管者题猜出“苔雀0P6”；SFT-256-17 证据正确但把 9 算成 8。报告对错误的概括没有超出这些样本所支持的描述，并提醒案例不用于估计错误频率。

### 表 9 反事实成对统计

`counterfactual_seed_mean_range.csv` 显示每个 seed 均有完整 80 对，报告表 9 的双题 joint 成功率和范围均匹配。SFT-256 的 0.413 对应 33/80=0.4125 的三位小数舍入；SFT-16、DPO、在线 DPO、GRPO-full 和 GRPO-format 的均值/范围也一致。报告正确将每对两个成员都 strict-joint-correct 定义为 pair success，而没有把 160 条样本当作 160 个独立 pair。

当前 CSV 只包含 `paired_double_joint_success`，没有 answer-flip consistency。报告的 3.2 数据说明与表 4 的指标职责表承诺分析答案是否随事实改变；6.3 只展示 pair-complete 成功率，正文还明确指出它不衡量答案翻转。建议待独立补算诊断生成后，将其定义、分母、seed 汇总和解释边界并入报告；否则需在最终范围说明中消除未兑现的内部承诺。

### 表 10 DPO 轨迹

按 β=0.1 从日志中的 chosen/rejected 相对参考 reward 还原后的数值与 `dpo_logp_margin_summary_by_seed.csv` 一致：seed 17 的初始/最终 chosen −0.031/−0.520、rejected −0.123/−16.868、margin 0.093/16.348；seed 29 为 −0.000/−1.623、0.114/−23.071、−0.114/21.448。叙述正确指出 chosen 项也下降、rejected 项下降更多，并未把该训练轨迹当作导致测试退化的因果证据。

### 表 11 资源与截断

表 11 的阶段更新数、耗时、generated/updated token 均值和范围与 `run_resources_seed_mean_range.csv` 及原始 run summary 相符；表注说明双种子方法报均值及 min–max，SFT-256 为 n=1。Online DPO 的 52 次更新和表述中的 75–81 个可用偏好对也与两次 run summary/analysis 行一致。GRPO updated tokens 约为同 seed generated tokens 的两倍，与每组 rollout 复用两个更新的报告口径相符。

将原始 `peak_cuda_allocated_bytes` / `peak_cuda_reserved_bytes` 按二进制 GiB（除以 1024³）换算后，所有运行的 allocated 最大值为 1.475625 GiB（报告 1.476 GiB），reserved 最大值为 2.234375 GiB（报告 2.234 GiB），匹配 analysis CSV。表 11 各方法 allocated 的均值和范围也匹配。表内列名明确标出 allocated memory；正文的 reserved 峰值另行说明。

训练 rollout 分母检查：Online DPO 两个 seed 合计 128×4×2=1,024 completions，文件中恰有 1 条截断；GRPO-full 与 GRPO-format 各自均有 1,024 条候选且各为 0 条截断。最终测试生成按每个模型/seed 的三个 160 条 split 计 480 条：baseline 有 3/480 条截断；DPO seed 29 有 5/480 条（test-ID 2 条、反事实 3 条）；其余已训练条件为零。报告的分母和归属正确。

### 图、公式与术语

- 报告中图 1 为实现映射，随后依次为图 2 联合正确率、图 3 任务分层/错误分解、图 4 DPO margin、图 5 资源与训练轨迹；相应 PNG 文件均存在，编号和图注顺序吻合。当前图示标题、错误条范围/均值、训练曲线和 DPO 终点与相应表格/分析一致。
- 全文有 34 个独立 `$$` 定界行，即 17 组成对的 display-math 区块；成对结构完整。SFT per-completion token mean、DPO BT/reference 公式及 GRPO 总体标准差和损失式均已纳入。
- 建议把 3.5、6.1 中的“证据精确率”（尤其表 6 文字叙述，数值实际为 `evidence_exact`）统一改称“证据集合精确匹配率”。由于报告另列 `evidence_f1`，用“精确率”容易被理解为 precision，而非 exact-set match。

## 建议交付前处理

1. 协调反事实 answer-flip 指标承诺与结果呈现；纳入补算值或收窄报告范围。
2. 将 `evidence_exact` 的中文名称改为“证据集合精确匹配率”，与 evidence F1 区分。

除上述两项外，本次抽查未发现摘要、表 6–11、案例或结论存在影响提交的数值/统计口径错误。核验为静态文件对照，不构成对 GPU 复跑或训练过程的独立复现。
