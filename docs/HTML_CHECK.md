# 实验报告渲染产物

本仓库只渲染根目录的 `实验报告.md`。纯文字综述和 DPO 思考题分别由各自作业目录管理；本仓库不存放它们的 HTML、纯文本或 manifest。

`rendered/实验报告.html` 是指向仓库本地图片的预览版；`rendered/piazza.html` 使用 `docs/piazza_image_map.json` 中的上传 URL。两者各自带有同名 `.txt` 和 `.manifest.json`。编辑实验报告后，在仓库根目录重渲染：

```powershell
.\.venv\Scripts\python.exe -m src.render_reports 实验报告.md --output rendered/实验报告.html --part-limit 200000
.\.venv\Scripts\python.exe -m src.render_reports 实验报告.md --output rendered/piazza.html --image-map docs/piazza_image_map.json --part-limit 200000
```

不提供 `--image-map` 时，HTML 用相对路径引用本地图片。若上传 URL 映射暂缺图片，可在提供 `--image-map` 时增加 `--draft-missing-images` 生成未发布草稿占位；含占位符的草稿不能用于 Piazza 最终提交。
