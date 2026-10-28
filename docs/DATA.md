# 合成证据问答数据

## 数据来源与可复现生成

本数据集完全由 `src/data.py` 的规则生成器构造，是用于后训练方法对比的**虚构中文短证据问答**。它不含真实语料、人工标注或外部检索结果，不能据此推断模型在真实领域文档上的表现。数据生成只依赖 Python 标准库，不需要模型、网络或 GPU。默认随机种子为 `20261028`；从项目目录运行 `python -m src.data` 可重新写出 `data/` 下的 JSONL 和清单。也可用 `--seed` 与 `--output-dir` 指定种子和输出目录。

生成器先抽取一个小型符号世界，再将事实渲染为编号材料；`metadata.latent` 保留关系、实体、事实编号和查询。`derive_label(row)` 从这些事实和查询重新推导答案及最小支持证据，目标标签不是随机指定的。训练输入使用 `prompt`，监督目标使用紧凑的 `target`；`metadata` 仅供审计和重算标签，不应拼接进模型输入。

## 划分

| 文件 | 条数 | 用途 |
| --- | ---: | --- |
| `data/train_sft.jsonl` | 512 | 监督微调 |
| `data/train_pref.jsonl` | 256 | 构造偏好对的独立提示池 |
| `data/preferences_train.jsonl` | 256 | 可直接用于 DPO 的 chosen/rejected 对 |
| `data/validation.jsonl` | 96 | 训练过程中的验证 |
| `data/test_id.jsonl` | 160 | 同分布最终测试 |
| `data/test_ood.jsonl` | 160 | 留出措辞和实体词表的测试 |
| `data/test_counterfactual.jsonl` | 160 | 80 个成对最小反事实样本 |

每个问答划分中的四种任务条数相同；`two_hop` 内部的关系链查询和数值加法查询也等量。每条提示含 4—6 条编号短材料。`train_sft`、`train_pref` 及其他普通划分不会复用提示；SFT 与偏好训练提示互不相同。偏好文件的 `chosen` 是标准目标，`rejected` 由同一提示自动构造并带有 `error_type`，错误类型为答案错误、证据错误、无依据猜答或过度拒答。不可回答样本不构造语义正确的“过度拒答”负例。

采用 Qwen2.5-0.5B-Instruct tokenizer（revision `7ae557604adf67be50417f59c2c2f167def9a775`）测量，全体提示最长为 216 个 Unicode 字符；直接编码为 153 token，加入模型 chat template 和 generation prompt 后为 182 token，低于配置中的 256-token 提示上限。

模型选择与超参数调整只使用 `validation`；三个测试划分只用于最终报告，不参与模型选择。

所有记录遵循统一结构：

```json
{"id":"train_sft-0000","split":"train_sft","task_type":"lookup","prompt":"仅依据编号材料作答……","answer":"青色","evidence":[2],"target":"{\"answer\":\"青色\",\"evidence\":[2]}","metadata":{"generator":"synthetic-evidence-qa-v1","template_family":"id","latent":{"facts":[],"query":{}}}}
```

`evidence` 使用提示中从 1 开始的材料编号。生成的 `target` 是无空格 JSON，键顺序固定为 `answer`、`evidence`；这是数据序列化形式，不构成评分对空格或键顺序的要求。提示要求给出全部必要编号，不列无关材料。无法从材料确定答案时，`answer` 为 `未知`、`evidence` 为空列表。反事实样本另带共享的 `pair_id`。

## 任务与答案推导

- `lookup`：从一条与问题实体及关系完全匹配的事实中查找属性；其他材料作为无关或同关系干扰项。
- `two_hop`：一半把“物品—保管者—存放处”两条关系连接起来，一半引用两条数量事实做整数求和。
- `unanswerable`：问题要求的关系对目标实体缺失；材料含有该实体的其他属性及干扰实体的事实，因此不能据常识补答。
- `negation`：候选名单事实只界定可能人选及名单边界，不表示名单中必有人持有目标物；另一条独立事实说明恰有一位保管者。再结合两名候选的否定事实，通过排除得到唯一剩余候选，因此标签的最小证据包含候选名单、唯一性和两条否定事实。

`test_ood` 使用其他实体词表与未在训练划分出现的指令/问句表达，但仍保留相同任务逻辑、编号材料形式和答案格式。因此它测量的是词汇与模板迁移，不代表结构上全新的推理任务；模板族有限，模型仍可能利用任务格式线索。`test_counterfactual` 每对保持问题与其他事实不变，只替换一个事实值或关系，并要求答案随之变化：包括颜色替换、第二跳存放处或数量替换、缺失关系补入，以及否定候选关系替换。不可回答任务的反事实伙伴在加入所问关系后会变成可回答，但保留原任务族标签以标识成对来源。

## 奖励与偏好接口

`src/rewards.py` 的 `score_response(text, row)` 先做严格 JSON/schema 检查，再分别返回格式正确、答案正确、证据精确匹配、证据 F1、联合正确、适用时的正确拒答及总奖励。格式检查要求一个恰含 `answer` 和 `evidence` 两键的 JSON 对象，答案为非空字符串，证据为无重复的正整数数组；拒绝代码围栏、对象外文本、重复 JSON 键、额外字段、错误类型及当前行材料编号范围之外的引用。JSON 内外空白和键顺序不影响格式判定。`score_response` 可依据当前行检查证据编号上界；`format_only_reward(text)` 没有 gold row，只检查 JSON/schema 结构，不检查材料编号上界。若答案可解析但证据越界或结构错误，仍保留答案正确性诊断。总奖励为

$$R=0.10\,I_{format}+0.45\,I_{answer}+0.45\,I_{answer\land evidence\_exact}.$$

因此只给出正确答案但引用材料错误时得分为 0.55，不会因列出大量证据而获得联合正确奖励。`format_only_reward(text)` 是 0/1 格式消融目标；`lenient_answer(text)` 只供错误分析，不参与训练奖励。`src/data.py` 的 `make_preference(row, rng)` 可用有种子的 `random.Random` 在训练提示上复现偏好拒答类型；验证和测试提示不得用于构造训练偏好。

公式中的每个指示量由相应布尔值转换为浮点数：`True` 计为 `1.0`，`False` 计为 `0.0`。证据精确匹配按去重后的编号集合比较，同时要求模型没有重复列号；证据次序不影响集合匹配。

## 复现和核验

```powershell
python -m src.data --seed 20261028
python -m unittest discover -s tests -p 'test_data_rewards.py' -v
```

`data/manifest.json` 记录生成器版本、数据/偏好随机种子、各划分规模与任务分布、唯一提示数及偏好错误类型计数。测试从 latent 重新计算标签，检查划分规模、任务平衡、提示去重、反事实变化和奖励边界。
