# 本地训练环境诊断

检查日期：2026-10-28（Asia/Shanghai）。本文区分首次只读诊断、随后经授权的环境准备，以及一次短推理冒烟测试。没有开展训练；显存、吞吐和时长估算仍与本机实测值分开标注。

## 可复用环境

项目训练与推理应使用项目虚拟环境；它通过 `--system-site-packages` 复用 Anaconda base 中已验证的 CUDA Torch：

```text
D:\北京大学\硕士\大语言模型与应用\作业5_后训练技术演进\.venv\Scripts\python.exe
Python 3.12.7
PyTorch 2.12.0+cu126（PyTorch CUDA runtime 12.6）
Transformers 4.51.3
Datasets 4.3.0
huggingface_hub 0.31.2
Matplotlib 3.9.2
PEFT 0.15.2
Accelerate 1.6.0
```

`torch.cuda.is_available()` 返回 `True`。PyTorch 能识别 1 张 NVIDIA GeForce RTX 3060 Laptop GPU，计算能力为 8.6，设备显存为 6 GiB；FP16 与 BF16 支持探测均为 `True`。`nvidia-smi` 报告驱动版本 596.08、驱动侧 CUDA 版本 13.2。CUDA 13.2 是驱动信息，不能据此当作 PyTorch runtime 版本；本环境中的 Torch wheel 标记为 `cu126`。

项目 `.venv` 中只新增了 `peft==0.15.2` 与 `accelerate==1.6.0`；安装过程复用 base 中的 Torch、Transformers 等依赖，没有升级或改写 base。`trl` 与 `bitsandbytes` 未安装。完整版本记录见 [requirements.txt](../requirements.txt)。

现有 Python 环境如下：

| 环境 | Python | 已观察到的相关组件 | GPU 训练可用性 |
|---|---:|---|---|
| 项目 `.venv` | 3.12.7 | 继承 base 的 Torch 2.12.0+cu126、Transformers 4.51.3、Datasets 4.3.0、HF Hub 0.31.2；独立安装 PEFT 0.15.2、Accelerate 1.6.0 | CUDA 可用，建议路径 |
| `base` | 3.12.7 | Torch 2.12.0+cu126、Transformers 4.51.3、Datasets 4.3.0、HF Hub 0.31.2；没有 PEFT 或 Accelerate | CUDA 可用，保留为共享底层 |
| `swe` | 3.10.19 | Datasets 4.4.1、HF Hub 1.1.7；无 Torch、Transformers、PEFT、TRL、Accelerate | 不能直接用 GPU 训练 |
| `wechat` | 3.8.20 | 未发现 Torch、Transformers、Datasets、PEFT、TRL 或 Accelerate | 不能直接用 GPU 训练 |

Conda 环境列表仍只含 `base`、`swe`、`wechat`；项目 `.venv` 是独立的 Python venv。首次诊断时，对课程目录的直接子目录检查了常见 `.venv`、`venv`、`env` 和 `conda-meta` 标记，未发现相邻作业专用虚拟环境。

项目 venv 的 `pip check` 有非零结果，原因是 `--system-site-packages` 暴露了 base 中既有的若干依赖冲突：例如 Datasets 4.3.0 要求 Requests 至少 2.32.2、base 为 2.31.0；另有 codebleu/tree-sitter、conda/pluggy、conda-repo-cli 与 selenium/urllib3 等约束不一致。PEFT 与 Accelerate 的导入和指定版本检查通过，`pip check` 输出没有报告它们自身的缺失约束。遵照环境要求，没有为此升级共享 base；若后续数据加载实际触发 Requests 兼容问题，应由主线在项目 venv 内评估局部修复。

## 资源与缓存

系统可见内存约 31.7 GiB，首次诊断时可用约 15.5 GiB。C 盘当时可用约 69.9 GiB，D 盘约 283.0 GiB。模型缓存根目录为 `C:\Users\-stq-\.cache\huggingface\hub`；首次诊断发现 `google-bert/bert-base-chinese` 的一个快照（约 0.38 GiB），未发现小型 Qwen 模型。之后 Qwen2.5-0.5B-Instruct 固定 revision 下载到默认缓存，模型目录实占约 999,586,387 字节（约 0.93 GiB）。

GPU 资源快照：RTX 3060 Laptop GPU 共 6144 MiB，检查时使用 0 MiB、利用率 0%，未发现运行中的 GPU 进程。显存快照是瞬时值，后续可能随桌面应用变化。

## 小型中文因果模型端点

首次只查询 Hugging Face 公共模型 API metadata，并对配置和权重 URL 发送 HTTP HEAD；没有获取权重内容。之后按固定 revision 下载 Qwen2.5-0.5B-Instruct；Qwen3-0.6B 仍只做了 metadata/HEAD 探测。两者均返回公开、未 gated 的 metadata，配置和单文件 `model.safetensors` 的 HEAD 均为 200。权重大小来自 CDN HEAD 响应，实际 Qwen2.5 文件长度在下载后再次匹配；这些文件大小不等于加载后的显存占用。

| 模型 | API 状态与修订 | HEAD 权重大小 | 判断 |
|---|---|---:|---|
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 200；SHA `7ae557604adf67be50417f59c2c2f167def9a775`；metadata 标记未 gated；已下载至默认缓存 | 988,097,824 B（约 0.92 GiB） | 已完成 tokenizer/config 与 GPU 生成冒烟验证 |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 200；SHA `c1899de289a04d12100db370d81485cdf75e47ca`；metadata 标记未 gated | 1,503,300,328 B（约 1.40 GiB） | 备选，仍在目标规模内，磁盘和显存开销更高 |

两者都以一个 `model.safetensors` 文件发布；相应 `model.safetensors.index.json` 的 HEAD 返回 404，与 API 文件列表中存在单权重文件、没有索引文件相符。下载由 [src/download_model.py](../src/download_model.py) 执行，固定上述 commit SHA，只下载 7 个配置、tokenizer 和 safetensors 文件，未将权重复制到仓库。首次下载约 1 分 56 秒；Hugging Face Hub 缺少可选 `hf_xet`，因此退回普通 HTTP。Windows 缓存不支持 symlink，Hub 发出空间效率警告，但本次缓存目录实测大小约 0.93 GiB。

## 模型加载与 GPU 冒烟测试（实测）

使用项目 `.venv`、本地缓存和固定 revision。CPU 侧加载配置与 tokenizer 用时 0.288 秒；Qwen2 模型配置、词表和 tokenizer 均能解析。FP16 权重加载并移动到 GPU 用时 4.870 秒。贪心生成设置 `max_new_tokens=32`，本次实际生成 32 token，用时 1.632 秒（约 19.6 输出 token/s）。PyTorch 记录的生成峰值为 961.1 MiB allocated、1,014.0 MiB reserved；显存释放清理后同一进程剩余分配 8.1 MiB，进程结束后的 `nvidia-smi` 显示 0 MiB、利用率 0%，没有运行进程。

冒烟加载期间 Transformers 发出 `Sliding Window Attention is enabled but not implemented for sdpa` 警告；因此一次短生成只证明本地模型可加载和生成，不证明该 attention 配置对正式长序列实验正确。训练或评测时应显式选择经核实的 attention 实现（可检查 `attn_implementation="eager"`），并记录实际配置。生成提示词中的 “LoRA” 被模型解释为 LoRa 无线通信；该输出不是任务准确率或后训练效果证据。此次时延与显存数据是单次冷启动冒烟测试，不是训练吞吐基准。

## 6 GiB 显存上的最小可行训练方案

下表是基于 0.5B 参数模型、单卡 RTX 3060 Laptop、短序列和小批次的**规划估算**，不是本机训练基准。实际峰值与速度会受笔记本 GPU 功耗档位、驱动、数据长度、padding、注意力实现、生成长度及训练库版本影响；建议为桌面占用和 CUDA 临时张量留至少约 0.5–1 GiB 余量。

| 方法 | 建议的起始配置 | 预计显存峰值（估算） | 吞吐/耗时（估算） |
|---|---|---:|---|
| LoRA SFT | Qwen2.5-0.5B-Instruct；FP16 或 BF16；LoRA rank 8；micro-batch 1；最大长度 512；梯度检查点；梯度累积 8–16；SDPA；`use_cache=False` | 约 2.5–4.5 GiB | 约 50–200 个训练 token/s；若 1,000 条样本平均 256 个有效 token，单轮约 22–85 分钟 |
| LoRA DPO | 使用成对的 prompt/chosen/rejected；micro-batch 1 pair；序列长度 512；梯度检查点；policy 从 SFT 策略初始化，reference 固定为同一 SFT 初始化策略 | 约 3.5–5.5 GiB，接近显存上限 | 约 25–100 个 chosen/rejected 合计 token/s；1,000 对、两侧各平均 256 token，单轮约 1.4–5.7 小时 |
| 在线 GRPO | micro-batch 2 条生成（每 prompt 2 个候选）；prompt 上限约 128 token、completion 上限约 64 token；梯度检查点；不加载 critic 或独立 reward model，优先用任务可验证的轻量奖励 | 约 4.5–6 GiB，可能 OOM | 生成阶段约 15–50 个输出 token/s（两个候选合计）；1,000 prompt × 2 × 64 token 的 rollout 约 0.7–2.4 小时，另加策略更新时间 |

推荐先做 SFT。LoRA 冻结基础模型参数，只为低秩适配器维护梯度和优化器状态；0.5B 模型的半精度权重文件约 0.92 GiB，因此无需先依赖 4-bit 才能尝试单样本、512-token 的微调。启用梯度检查点、使用批次 1 和短序列可限制激活内存。若出现 OOM，依次将最大长度降到 384/256，再确认无其他 GPU 进程；不要先增加批次。

DPO 需要高质量偏好对。reference policy 必须是 SFT 初始化策略的冻结版本。若两者共享底座，reference adapter/权重状态仍须表示 SFT 后策略；简单禁用 SFT adapter 会回退到原始 Instruct 底座，不能作为等价 reference。adapter 复制、合并或独立参考模型的实现由训练工作包决定；本环境包未实现 DPO。`trl` 当前未安装。

GRPO 是三者中风险最高的方案：即使模型小，在线生成、多个候选的 log-prob 与训练激活会共同占用显存。若课程要求真实在线组相对策略梯度，可先以 2 个候选和 64-token completion 做一个极小可验证运行；如果峰值超限，再评估 QLoRA 或 CPU offload，但本机尚未检查这些依赖的安装可用性。奖励函数应与任务对应，并记录 rollout 数、奖励统计和实际 GPU 峰值，不能将以上估算当作实验结果。

## 检查范围与来源

- 本机 Python、PyTorch、包版本、CUDA 设备与 dtype 支持：通过各 Python 可执行文件直接查询；PEFT/Accelerate 导入和版本已验证。
- GPU 利用率、显存、驱动：`nvidia-smi` 只读查询，诊断末尾复查仍为 0 MiB / 0%。
- 内存、磁盘、Conda 环境和模型缓存：Windows CIM / Conda 元数据 / Hugging Face 默认缓存目录只读查询。
- 模型可访问性：`https://huggingface.co/api/models/{repo}` metadata GET；`config.json`、`model.safetensors` 和权重索引使用 HEAD。Qwen2.5 按固定 revision 下载并在本地完成一次模型生成；没有下载 Qwen3 权重。
- 训练显存和吞吐：机制与模型规模的保守规划估算，未在本机运行训练或做训练速度基准；短生成的实际耗时和峰值另列于冒烟测试一节。
