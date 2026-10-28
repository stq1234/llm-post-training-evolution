# 后训练技术演进综述

- Piazza 帖子独立标题：关于大语言模型后训练技术演进的综述
- 作业归属：HW5；该帖实际勾选 `hw1`、`hw2`、`hw3`、`hw4`、`hw5`、`other` 六个标签
- 本作业分支已归档：[https://github.com/stq1234/llm-post-training-evolution/tree/survey](https://github.com/stq1234/llm-post-training-evolution/tree/survey)
- 正文：[`综述论文.md`](综述论文.md)
- 正文规模：约 2.45 万汉字、26 个公式、46 篇参考文献、10 张表和 4 张图。
- 支撑材料：`notes/` 中的来源与证据记录；`figures/` 中的技术谱系、奖励、训练回路和 R1 图，由本目录 `src/plot_concepts.py` 与 `data/concept_figures.json` 维护。

综述正文、来源记录和图表均单独维护在本作业目录；实验代码与结果归档在相邻的作业 5.2 目录。

## 渲染

正文和图片定稿后，可从本目录运行下列命令，使用相邻 5.2 项目的 venv 和渲染器。重写完成前不要生成 HTML。

```powershell
& "..\作业5.2_后训练代码分析与实验\.venv\Scripts\python.exe" "..\作业5.2_后训练代码分析与实验\src\render_reports.py" "综述论文.md" --output "rendered\综述论文.html" --part-limit 200000
```
