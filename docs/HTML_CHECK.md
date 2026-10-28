# 报告 HTML 转换核验

## 范围与产物

本次历史核验使用 `src/render_reports.py` 将当时的 Markdown 转为本地 HTML 预览、完整纯文本和 JSON manifest；旧 HTML 中的图像 URL 为本机 `file:///`。当前渲染器不提供 `--image-map` 时使用仓库相对图像路径，主线定稿后会重新渲染。没有打开浏览器或 Piazza，也没有上传或重绘插图；若提供的 URL map 尚缺图片，`--draft-missing-images` 仅用于未发布草稿，不能用于最终发布 HTML。

以下字节数、分片统计、内容计数及 manifest 状态是报告迁入独立子目录前的历史核验记录，不代表当前源文件重渲染后的字节数。源文件现位于 canonical 报告路径；HTML 和纯文本的既有输出文件名保留，主线将在内容定稿后重渲染并复核。

本地预览命令（项目虚拟环境，未提供 `--image-map`，图片以相对路径引用）：

```powershell
.\.venv\Scripts\python.exe -m src.render_reports reports/01_综述/报告.md --output reports/html/01_survey.html
.\.venv\Scripts\python.exe -m src.render_reports reports/03_DPO思考题/回答.md --output reports/html/03_dpo_questions.html
```

| 报告 | 完整 HTML | 完整纯文本 | manifest | 公式／表格／图像 | 分片 |
|---|---|---|---|---|---|
| `reports/01_综述/报告.md` | `reports/html/01_survey.html`（64,999 B，迁移前记录） | `reports/html/01_survey.txt` | `reports/html/01_survey.manifest.json` | 21 组数学定界符、3 张表、2 幅本地图像 | 14 片；均按完整章节边界切分，最大 15,819 B |
| `reports/03_DPO思考题/回答.md` | `reports/html/03_dpo_questions.html`（33,831 B，迁移前记录） | `reports/html/03_dpo_questions.txt` | `reports/html/03_dpo_questions.manifest.json` | 52 组数学定界符、1 张表、无图像 | 完整 HTML 小于 50,000 B，无需拆分 |

综述的分片 HTML 与对应纯文本位于 `reports/html/01_survey.part-NN.html` 和 `reports/html/01_survey.part-NN.txt`。每个片段均为独立完整 HTML 文档，并按 Markdown 二级标题保留完整章节；分片文本连接后与完整文本 manifest 的正文字符顺序一致（忽略空白差异）。纯文本文件在图像原位置加入 `[图片插入点：替代文字]` 标记，便于粘贴时定位。

## 核验结果

- 两份完整 HTML 及全部综述分片均包含 `doctype`、`html`、`head`、`body`，可作为独立 HTML 文档解析。
- 标题层级与 Markdown 渲染结果一致：综述 27 个标题，DPO 思考题 4 个标题。表格分别为 3 张和 1 张；Markdown 中的 57 个综述链接、29 个思考题链接均保留到 HTML `href`。
- 从源文抽取的数学片段逐一出现在 HTML `body` 文本中：综述 21 个，思考题 52 个；反斜杠、上下标及 `$...$`／`$$...$$` 定界符均保留。HTML 中没有将这些公式置于 `code` 元素；两份源文没有围栏代码块或行内代码公式。
- 所有正文引用编号均能在转换后的正文文本中找到。完整纯文本 manifest、分片纯文本及 HTML 均无截断；manifest 的 `parts_text_matches_full` 为 `true`。
- 本次按源文最新修订重渲染综述；ORPO 可从预训练模型联合适配与偏好优化、SimPO 偏好阶段从 SFT 检查点出发的区别，已出现在完整 HTML、对应章节分片及方法表中。
- 综述的 2 幅图均能定位到“9 训练系统决定算法能否落地”章节和第 10 片；文件存在，HTML 中使用 `file:///` 本地图像 URL，纯文本保留替代文字定位标记。DPO 思考题未引用图像。
- 执行 `python -m src.render_reports` 成功生成上述文件；另以项目虚拟环境执行渲染核验与 `py_compile` 检查，均通过。

## 边界

HTML 中的数学保留为原始 LaTeX 定界符和公式文本，未转成图片或浏览器渲染结果。本次只验证源文到静态 HTML／纯文本的保留情况；Piazza 中的数学排版与图片 URL 替换仍由主线在发布环节核验。
