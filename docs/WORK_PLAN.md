# 工作计划与验收

## 目标

1. 完整中文学术综述：2023—2026 年后训练目标、算法、数据与奖励、训练流程、实现转换；向前追溯必要基础。方法维度综合分析，保留来源证据，绘制技术谱系、训练回路等图，形成对照表及不少于约 30 篇相关原始文献。
2. 多项目源码分析与 6GB 显存上的真实后训练实验，报告联结理论、代码路径、实验设计、结果和讨论；至少覆盖 6 个有实质差异的开源实现，避免只列 README。
3. 两道 DPO 思考题：在线 DPO 的可行形式与前提；推理模型为何继续采用在线 RL。按题作答，公式、反例、方法比较和文献支持。
4. GitHub 全量归档后，在 Piazza 发布三个富文本帖子，并验证全文、公式、图片和标题。

## 固定信息

- 仓库：https://github.com/stq1234/llm-post-training-evolution （用户提供 .git 远程地址）。
- 帖子标题原文：关于大语言模型后训练技术演进的综述；大语言模型后训练技术演进代码分析与任务实操；DPO与其他强化学习训练对比的思考。
- 三个帖子均归属 HW5，在 Piazza 选择标签 `hw5`；课程标签集合为 `hw1` 至 `hw5` 及 `other`。标题与本地正文入口见 [`reports/README.md`](../reports/README.md)。
- 本机 NVIDIA RTX 3060 Laptop 6GB。训练 GPU 由主线独占调度。
- Markdown 为定稿；正式正文不包含执行障碍、内部检查、哈希或机器路径。

## 分工与依赖

- 主线：范围、实验设计与整合、真实训练验收、图表质量、GitHub、Piazza。
- 文献包：原始论文及技术演进证据、综述初稿。
- 代码包：锁定公开版本并读取真实函数路径，提供方法到代码映射。
- 思考题包：两问题的深入论证及原始来源。
- 环境包：维护可复现 README/requirements、项目 venv 与固定模型缓存；GPU 仅做短推理冒烟验证，不训练。训练和评估 GPU 调度归主线/训练包。

## 实验决策与验收

使用固定的小型因果 LM、合成中文证据问答任务及固定划分。主训练矩阵为 seed 17、29 两次独立运行：SFT 16 update / grad accumulation 4；离线 DPO 64 / 4；OnlineDPO 64 update、最多 128 prompt、completion 上限 64、grad accumulation 2；GRPO 128 update、2 update epochs、grad accumulation 2，并做 format-only 奖励消融。SFT 的可选选择 pilot 为 seed 17 的 32 update；可选 `sft_long_seed17` 为 256 update 高预算监督对照。16、32、256 均表示 SFT optimizer updates，不改变 prompt/completion/max-new-token 上限。命令、reference-policy 语义、评估目录和复现步骤以根目录 `README.md` 为准。最终比较包含基础模型、SFT、离线偏好优化及在线反馈方法；validation 只用于方案确认，test splits 在方案固定后评估。保留原始逐样本预测、rollout、更新日志和可复用绘图代码，关键统计可重算。

## 当前状态

- **已完成的资料与工程准备**：综述正文在 [`reports/01_综述/报告.md`](../reports/01_综述/报告.md)，DPO 思考题在 [`reports/03_DPO思考题/回答.md`](../reports/03_DPO思考题/回答.md)；源码分析与实验正文入口为 [`reports/02_代码分析与实验/报告.md`](../reports/02_代码分析与实验/报告.md)。三个帖子均属 HW5，标题和正文入口集中列于 [`reports/README.md`](../reports/README.md)。`research/code_analysis.md` 与来源索引覆盖 8 个公开仓库、10 个版本快照。实验数据、训练日志、评估预测及结果分析均已生成；实验报告正在整合已核验结果。
- **环境准备**：项目 venv 和依赖版本已记录在 `requirements.txt`；固定 revision 的 Qwen2.5-0.5B-Instruct 已缓存，完成 CPU 配置/tokenizer 加载及最多 32-token GPU 推理冒烟验证。冒烟测试不是训练实验。当前 venv 复用了系统包，`pip check` 暴露了既有共享依赖冲突；PEFT/Accelerate 导入通过，未改写 base。新机器的干净安装步骤见 `README.md`。
- **训练与评估已完成**：validation 与三个测试划分共 48/48 个方法、seed、split 评估均完整，产生 6,912 条预测；`results/analysis/summary.json` 没有缺失预测文件、不完整评估或缺失训练摘要。主要方法均运行 seed 17、29；基线与 256-update SFT 高预算对照 `sft_long_seed17` 为单次运行。Validation 预算规则选中 SFT-16。项目检查记录为 17 项测试通过（本次文档更新未重跑）。方法指标和资源实测见 [`research/experiment_results.md`](../research/experiment_results.md)。
- **结果要点**：严格联合正确要求 JSON 格式、答案和证据编号集合同时正确。两 seed 的 SFT 平均联合正确率在同分布、措辞留出、反事实测试分别为 0.363、0.334、0.322；DPO 为 0.225、0.291、0.159，配对差在三项划分都低于同 seed SFT。OnlineDPO 为 0.366、0.366、0.378，full-reward GRPO 为 0.397、0.366、0.394；full 相对 format-only GRPO 的配对差分别为 +7.2、+3.1、+9.7 个百分点。`sft_long_seed17` 得分较高，但仅一个 seed，作为单次高预算对照，不代表稳定的跨 seed 结论。完整数值与边界见结果文档。
- **待完成工作与发布状态**：将已核验结果及相应图表整合进实验报告，定稿后重新渲染 HTML；GitHub 仓库尚未推送。Piazza 的 01 full 仍是草稿，`loops` 图片上传连续三次返回 `Failed to load file data`，因此帖子尚未发布；需先解决图片上传并复核正文、标题和图片，再进行发布。未发布草稿不视为已提交成果。

## 发布前检查

原文标题逐字匹配；摘要后的仓库入口格式；完整正文无占位符；图表有编号和解释；文献报告结果与本实验实测结果明确区分；实验数值可从归档日志/评估文件重算；公式实际渲染；图片通过编辑器上传；归档链接可访问；删除本任务临时材料并停止测试服务。
