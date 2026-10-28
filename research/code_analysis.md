# 后训练方法的源码实现比较

## 1. 范围与阅读口径

本节从训练代码而非项目简介出发，比较八个公开实现：Hugging Face TRL、CarperAI trlx、DeepSpeed-Chat、OpenRLHF、verl、THUDM slime、Hugging Face open-r1 与 MiniMind。对 TRL 同时阅读历史 PPO 版本、open-r1 实际钉住的 v0.18.0 GRPO 内核和当前训练器：历史 PPO 取 v0.11.4，DPO、在线 DPO 与当前 GRPO 取当前代码；其余仓库以审阅时的公开主分支源码为准。每个实现均沿入口、输入数据、采样、奖励或偏好信号、优势构造、损失聚合、梯度更新及检查点保存追踪。链接固定到对应源文件版本，便于定位到实现细节。

比较时须区分三类策略分布。记 πθ 为当前待更新策略，πold 为形成 PPO 更新批次时的行为策略，πref 为固定参考策略。πold 用于重要性比率 πθ/πold；πref 通常用于 KL 约束或 DPO 的偏好对比。两者概念不同，即使实现通过共享底座、禁用适配器或复用同一个模型对象来省显存，也不能把它们在数学目标中的作用混为一谈。纯 DPO 没有 PPO 意义上的旧策略，也不需要价值网络；在线 DPO 另有采样策略，但仍以参考策略的序列概率差构造偏好目标。

有效响应掩码同样属于目标定义的一部分。提示 token、右侧 padding、EOS 后 token、工具输出、被拒绝的截断轨迹以及训练推理引擎不一致所过滤的 token，决定哪些对数概率进入优势、KL 和梯度。即使“算法名称”和剪切系数相同，掩码位置和损失分母不同，更新仍不是同一个目标。

## 2. 实现结构概览

| 实现 | 本次读到的方法 | 采样与信号来源 | 优势或偏好目标 | 损失权重与并行特征 |
|---|---|---|---|---|
| TRL 历史/当前 | v0.11.4 PPO；当前 DPO、在线 DPO、GRPO | PPO 接收响应与分数；在线 DPO/GRPO 可本地或由 vLLM 采样，调用奖励函数 | PPO 用 GAE；GRPO 按同一提示分组标准化奖励；在线 DPO 将两条采样按奖励排序 | PPO 与 GRPO 都显式使用响应掩码；当前 GRPO 支持 token/sequence IS、多种损失归一化和采样偏差修正 |
| trlx | PPO | Accelerate 分发提示和采样；主进程计算奖励后 scatter | 价值网络估计 GAE；参考模型 KL 以逐 token 惩罚加入奖励 | token 掩码平均；Accelerate 多进程、梯度累积、mini/micro-batch |
| DeepSpeed-Chat | PPO、DPO | PPO 在 actor 上采样，独立奖励模型打分；DPO 使用成对偏好数据 | PPO 用 GAE；DPO 直接比较 chosen/rejected 对数概率比 | token 掩码均值；DeepSpeed Actor/Critic、ZeRO 与可选卸载 |
| OpenRLHF | PPO、RLOO、组归一化策略梯度、GSPO 等配置项 | Ray/vLLM 生成，奖励模型或外部奖励服务评分 | 经验构造阶段支持 GAE、RLOO、组归一化等估计器 | 既有 token-mean，也有逐样本 token-mean；支持序列级 GSPO 与训练/采样偏差校正 |
| verl | PPO、GRPO 等优势估计器，v1 同步/异步训练 | AgentLoop 与推理客户端生成，经 Transfer Queue 回传 | 配置选择 GAE 或组相对优势；可区分 rollout 与旧策略对数概率 | token/序列多种全局归约；Ray worker、FSDP/Megatron 与队列式数据传递 |
| slime | PPO、GRPO、GSPO、CISPO、REINFORCE++ 等 | SGLang/Ray 异步 rollout，BatchBuilder 转为 Megatron 训练批次 | 组奖励后处理或逐 token GAE；支持多种策略目标 | 样本列表、变长 token 与 loss mask；Megatron 张量/流水线并行及权重版本同步 |
| open-r1 | 基于 TRL v0.18.0 的 GRPO 配方 | 数据集映射成对话提示，TRL 执行采样；奖励注册表含准确率、格式等 | GRPO 的分组基线与三类损失归一化由其锁定的 TRL 实现负责 | 仓库代码主要负责数据、奖励及训练配置；示例配置启用 vLLM |
| MiniMind | DPO、PPO、GRPO 训练脚本 | DPO 读 chosen/rejected；GRPO 用可插拔 PyTorch 或 SGLang rollout engine | DPO 用冻结参考模型；GRPO 采用组内均值/标准差奖励优势 | GRPO 对每条回复先做 token 平均、再做样本平均；单机 PyTorch/DDP，提供完整状态恢复 |

表中的方法名只描述仓库中可选的实现分支，不表示所有组合均经过同等验证，也不意味着这些框架复现了某篇论文的全部实验条件。尤其是 open-r1 是建在 TRL 之上的任务与奖励配方，而不是另一套独立的 GRPO 数学内核。

## 3. 各仓库的调用链与实现选择

### 3.1 TRL：历史 PPO、当前 GRPO 与在线 DPO

TRL v0.11.4 的 PPOTrainer.step 接收已经编码的 query、response 和外部 score。它在 step 内组织输入批次、计算行为策略和参考策略对响应 token 的 log-prob、价值与掩码；compute_rewards 将 log πold−log πref 乘 KL 系数作为逐 token 非分数奖励，并把标量分数放到最后一个有效响应 token。compute_advantages 通过价值序列执行 GAE，并对掩码范围内的优势 whitening；loss 对当前与旧策略概率取比、剪切策略代理目标，同时使用剪切价值损失；train_minibatch 调用 Accelerate backward、梯度裁剪、优化器 step 和清零梯度。其检查点复用 Trainer/Accelerate 状态保存机制。这条路径的关键并非 PPO 名称，而是 response mask 同时约束终止位置的奖励、GAE 递推统计与 actor/critic 损失。[PPO step 与样本检查](https://github.com/huggingface/trl/blob/1f8ba929fbb6962249386f7dce20155da8a26cda/trl/trainer/ppo_trainer.py#L655-L740)；[参考 KL、GAE 与 PPO 损失](https://github.com/huggingface/trl/blob/1f8ba929fbb6962249386f7dce20155da8a26cda/trl/trainer/ppo_trainer.py#L1112-L1258)；[响应掩码和 minibatch 更新](https://github.com/huggingface/trl/blob/1f8ba929fbb6962249386f7dce20155da8a26cda/trl/trainer/ppo_trainer.py#L979-L1110)。

当前 TRL 将训练器拆成更适合策略采样的路径。GRPOTrainer 的 _generate_and_score_completions 读取 prompt，可按数据行实例化工具环境，生成 completions，计算一个或多个 reward function，并按 num_generations 将奖励重排为提示组。无分数样本以 NaN 排除组统计，优势由组奖励减组均值构造；是否再除组标准差由 scale_rewards 控制。损失侧把 completion_mask 与 tool_mask 相乘，因此工具执行产生的文本不会自动被当作策略动作优化。ref 模型或 PEFT 参考 adapter 提供逐 token KL 项；old_per_token_logps 供 PPO 式重要性比率使用；若 vLLM 的采样概率与训练模型概率不同，则另保留 sampling_per_token_logps，并可选择按 token 或整条响应修正、截断或屏蔽权重。GRPO、BNPO、DR-GRPO、DAPO、CISPO、SAPO、VESPO 和 LUSPO 分别使用不同损失分母，不能只用“组相对奖励”概括其梯度尺度。[生成、分组奖励和优势](https://github.com/huggingface/trl/blob/c886cf23d74acd8dfed95b63943a8dba0f25d85f/trl/trainer/grpo_trainer.py#L2371-L2835)；[旧策略、采样策略、参考策略和 loss mask](https://github.com/huggingface/trl/blob/c886cf23d74acd8dfed95b63943a8dba0f25d85f/trl/trainer/grpo_trainer.py#L3067-L3154)；[损失类型与分母](https://github.com/huggingface/trl/blob/c886cf23d74acd8dfed95b63943a8dba0f25d85f/trl/trainer/grpo_trainer.py#L3156-L3229)。

在线 DPO 走的是另一条数据流。OnlineDPOTrainer 对每个 prompt 采样两条候选，逐条调用奖励函数；无 EOS 时可扣分；随后按奖励高低选出 chosen/rejected，并分别求掩码后的策略序列 log-prob 总和与参考策略总和。两组差之差构成 DPO logit，最后应用 sigmoid 或 IPO 目标。这里没有价值网络、GAE 或逐 token advantage；奖励仅用于决定偏好次序，偏好幅度不直接作为梯度权重。由于对数概率采用序列求和，候选长度会影响差值；改成 token 均值会改变偏好目标，不能视为数值上的等价归一化。[Online DPO 采样与奖励排序](https://github.com/huggingface/trl/blob/c886cf23d74acd8dfed95b63943a8dba0f25d85f/trl/experimental/online_dpo/online_dpo_trainer.py#L1149-L1238)；[掩码序列概率与偏好损失](https://github.com/huggingface/trl/blob/c886cf23d74acd8dfed95b63943a8dba0f25d85f/trl/experimental/online_dpo/online_dpo_trainer.py#L1242-L1285)。

因此 TRL 适合作为算法原型和可配置基准，但复现时至少需要锁定 TRL 版本、采样器、completion 长度、EOS 与工具掩码、reward 聚合、loss_type、importance_sampling_level、gradient accumulation 与生成批次。GRPOTrainer 的保存覆盖先写模型卡再转交 Transformers Trainer 持久化训练状态；DPO 与在线 DPO 也有相应的 checkpoint 覆盖。[GRPO checkpoint](https://github.com/huggingface/trl/blob/c886cf23d74acd8dfed95b63943a8dba0f25d85f/trl/trainer/grpo_trainer.py#L3432-L3438)；[Online DPO checkpoint](https://github.com/huggingface/trl/blob/c886cf23d74acd8dfed95b63943a8dba0f25d85f/trl/experimental/online_dpo/online_dpo_trainer.py#L1371-L1377)。只记录 β、ε 或 num_generations，不足以确定优化器实际看到的样本权重。

### 3.2 trlx：Accelerate 下的 PPO 样本生成与集中式奖励

trlx.utils.loading 通过注册表按配置名称提供 trainer 与 pipeline 构造器；PromptPipeline 对提示编码，并在批处理时保留额外字段，使其可以作为 reward function 的 metadata 传递。AcceleratePPOTrainer 由模块导入注册，prompt iterator 的具体构造与采样训练可替换，而 PPO experience 和 loss 结构留在 PPO trainer 内。[注册表入口](https://github.com/CarperAI/trlx/blob/3340c2f3a56d1d14fdd5f13ad575121fa26b6d92/trlx/utils/loading.py#L3-L50)；[PromptPipeline 编码和批处理](https://github.com/CarperAI/trlx/blob/3340c2f3a56d1d14fdd5f13ad575121fa26b6d92/trlx/pipeline/offline_pipeline.py#L119-L185)。

trlx 的 AcceleratePPOTrainer.make_experience 从 PromptPipeline 取提示，经模型生成后跨进程 gather 提示和 completions。奖励函数只在主进程执行，契约要求返回 token 级分数；主进程把分数 pad 后 scatter 回各进程。随后模型正向计算当前策略的行为 log-prob、参考 log-prob 与价值，利用 score mask 排除无效位置，将参考 KL 惩罚加入奖励并在 EOS 位置注入标量结果，最终写入 rollout store。这个“主进程评分—分发 token 奖励”的约束会影响奖励服务的吞吐、数据对齐和异常传播，特别是奖励函数返回序列级 score 时，框架通过末 token 放置而非自动分摊来完成 credit assignment。[PPO rollout、reward_fn 与 score 分发](https://github.com/CarperAI/trlx/blob/3340c2f3a56d1d14fdd5f13ad575121fa26b6d92/trlx/trainer/accelerate_ppo_trainer.py#L251-L341)；[KL 与终止奖励写入样本](https://github.com/CarperAI/trlx/blob/3340c2f3a56d1d14fdd5f13ad575121fa26b6d92/trlx/trainer/accelerate_ppo_trainer.py#L440-L524)。

更新阶段的 loss 从样本存储中的 old log-prob、old value、reward 计算 GAE；按照 causal 或 seq2seq 结构构造 response mask，当前模型输出再送给 PPO clipped policy/value loss。两类损失都按 mask 中有效 token 的计数归一，而不是先对每条响应求均值再对样本平均。BaseTrainer.learn 支持多个 PPO inner epochs、mini/micro-batch 和梯度累积，逐 microbatch 调用 accelerator.backward，完成后 optimizer 与 scheduler 各更新一次。复现者若将分布式 reward 改为每进程本地调用，必须确认 reward 与跨进程 gather 后样本顺序一致，否则不是纯粹的性能替换。[PPO GAE 与 token masked loss](https://github.com/CarperAI/trlx/blob/3340c2f3a56d1d14fdd5f13ad575121fa26b6d92/trlx/models/modeling_ppo.py#L136-L227)；[microbatch backward 与 optimizer 更新](https://github.com/CarperAI/trlx/blob/3340c2f3a56d1d14fdd5f13ad575121fa26b6d92/trlx/trainer/accelerate_base_trainer.py#L551-L588)。

### 3.3 DeepSpeed-Chat：拆分式 actor、critic、奖励模型与 DPO 阶段

DeepSpeed-Chat 的 step3 PPO 以 actor 生成 response，调用 actor、冻结 reference、reward model 和 critic 四路前向得到 log-prob、参考 log-prob、序列末端奖励分数和值。compute_rewards 对动作 token 加入 KL 惩罚，并把裁剪后的奖励模型分数放到最后一个有效动作位置；train_rlhf 按有效结束位置清零后续奖励和值，执行 GAE。Actor 和 critic 先后各自反向与 step，可由配置让两者在混合精度 overflow 时同步跳过，以免一个更新而另一个未更新。PPO actor loss 是掩码 token 加权均值，critic 使用 clipped value loss。主循环先生成并存入 MiniDataset，再遍历 PPO epoch，最后保存 actor 与 critic 状态；项目官方训练文档报告，其成功运行使用生成 batch 等于训练 batch、ppo_epochs=1 且 generation_batches=1，并指出增大 PPO 数据复用时行为策略 log-prob 与更新策略 log-prob 会迅速分离。该经验说明更新复用次数和 rollout 批次本身就是稳定性条件，而非无关工程参数。[经验构造和终止奖励](https://github.com/deepspeedai/DeepSpeedExamples/blob/0c6dd5ed1ec0f0967ae0564534ecfa6f157c6746/applications/DeepSpeed-Chat/dschat/rlhf/ppo_trainer.py#L137-L194)；[PPO actor/critic 更新及 overflow 对齐](https://github.com/deepspeedai/DeepSpeedExamples/blob/0c6dd5ed1ec0f0967ae0564534ecfa6f157c6746/applications/DeepSpeed-Chat/dschat/rlhf/ppo_trainer.py#L196-L306)；[step3 主循环与检查点](https://github.com/deepspeedai/DeepSpeedExamples/blob/0c6dd5ed1ec0f0967ae0564534ecfa6f157c6746/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/main.py#L513-L640)；[官方稳定性观察与推荐批次关系](https://github.com/deepspeedai/DeepSpeedExamples/blob/0c6dd5ed1ec0f0967ae0564534ecfa6f157c6746/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/README.md#L69-L80)。

同一仓库 step2 的 DPO 使用 preference pair。get_batch_logps 以 label mask 排除 prompt 和 padding，累计目标 token log-prob；训练循环对 policy 与 reference 分别前向，计算 chosen 和 rejected 的序列 log-ratio 后形成 DPO 损失。这里 reference 是静态偏好锚点，policy 以成对数据直接更新，无需 PPO old policy 或 critic。官方代码的阶段式组织便于对照 SFT、奖励模型、PPO、DPO，但各阶段数据与模型检查点必须逐项匹配；只说明“用了 DeepSpeed-Chat”仍无法得知采取哪条训练阶段或 PPO 调度参数。[DPO log-prob 计算与更新循环](https://github.com/deepspeedai/DeepSpeedExamples/blob/0c6dd5ed1ec0f0967ae0564534ecfa6f157c6746/applications/DeepSpeed-Chat/training/step2_dpo_finetuning/main.py#L219-L240)。

### 3.4 OpenRLHF：Ray 角色组、可替换优势估计器与归约策略

OpenRLHF 的 train_ppo_ray 构造 vLLM 生成引擎、actor/reference/reward/critic Ray ActorGroup，可按 placement group 将角色共置。ExperienceMaker 为 rollout 构造 action、attention mask，将序列送入 actor 得到行为 log-prob，分别分发到 reward、critic 与 initial/reference 角色；experience 记录 action_log_probs 与 base_action_log_probs。随后奖励被按每个 prompt 的 n_samples_per_prompt 重排，可选择 RLOO leave-one-out、去组均值、组标准差归一或全局均值等 baseline；对需要 critic 的配置，则以 GAE 递推优势。因而“GRPO”与“PPO”在该框架中既受 advantage estimator 影响，也受 actor loss 的 policy_loss_type 影响，需分别记录。[Actor/RM/critic/ref 前向及经验字段](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_utils/experience_maker.py#L112-L233)；[组奖励与优势估计](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_utils/experience_maker.py#L238-L387)。

PolicyLoss 显示该项目的目标范围比标准 PPO clipping 更广：可使用 token PPO，也可按有效响应 token 的平均 log-ratio 形成 GSPO 的 sequence ratio；off-policy correction 可从 πtrain/πrollout 的概率比或 sampled-token divergence 估计偏差，并按 token 或 sequence 颗粒度屏蔽/截断。更容易被实验忽略的是 aggregate_loss：token_level_loss=True 时全局有效 token 总和除以全局 token 数；False 时先计算单样本 token 均值，再按全局样本数平均。两者回答的是不同的统计问题：前者让长响应贡献更多动作项，后者给予每条非空响应相同总权重。配合动态批处理与 DP 梯度平均时，代码显式传入全局计数及 dp_size，避免局部 shard 长度改变梯度尺度。[token/sequence 归约](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/models/loss.py#L11-L39)；[PPO/GSPO 与 IS 校正](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/models/loss.py#L116-L259)。

训练器把生成、experience 制作、动态样本负载平衡、actor/critic 训练、权重广播至 vLLM 和 KL 控制器更新显式分成阶段；checkpoint 同步 actor 与 critic 客户端状态。适合大规模采样不意味着适合小显存单机：Ray 角色规划、vLLM 权重同步、奖励/critic 模型与动态批处理共同带来额外运行依赖和调度复杂度。[训练步、vLLM 权重同步与保存](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_trainer.py#L213-L264)；[PPO actor 微批次反向](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ray/ppo_actor.py#L259-L359)。

### 3.5 verl：Transfer Queue、训练模式与全局样本归约

本次按当前主树的 v1 入口审阅，而不是将旧 Ray trainer 的设计误称为唯一实现。Hydra main_ppo 创建 v1 TaskRunner，TaskRunner 根据 trainer_mode 选取同步/异步 trainer，初始化 Transfer Queue 与 AgentLoopManager；每次 step 从 replay buffer 取批次，按需计算奖励、平衡 DP 批次，计算 old_log_prob、可选 ref_log_prob 和 value，再构造 advantage、更新 critic 与 actor。样本 key/fields 在 worker 与训练器之间通过队列传递，采样服务和训练角色可以使用不同 worker 组合。[v1 入口与 AgentLoopManager 初始化](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/trainer/main_ppo.py#L107-L178)；[每批次从 reward 到 actor update 的顺序](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/trainer/ppo/v1/trainer_base.py#L563-L613)。

其策略概率字段至少有三种语义：rollout_log_probs 是实际采样服务概率，old_log_probs 是 PPO 优化窗口的 anchor，ref_log_prob 是参考策略概率。bypass 模式把 rollout 概率直接作为 old；decoupled 模式会重新计算旧策略作为稳定 anchor，形成 rollout、old 与当前 πθ 三个策略的分离。可选 rollout correction 再通过 importance sampling/rejection 或统计指标处理训练推理偏差。这一实现分支会改变 ratio 的分母及失配纠正，单凭“策略更新用了 PPO clip”无法重建目标。[old policy 的 bypass/decoupled 语义](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/trainer/ppo/v1/trainer_base.py#L1635-L1694)；[ref log-prob 与优势数据字段](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/trainer/ppo/v1/trainer_base.py#L1696-L1803)。

verl 的 GRPO estimator 将每条响应 token reward 求和后，按 uid 聚合提示组；标准化开关关闭时对应不除以标准差的变体。单条响应组成的组使用均值零、标准差一，避免空基线导致数值异常。loss 聚合提供 token-mean、token-sum、seq-mean-token-sum、固定 horizon 的 seq-mean-token-sum-norm 与 seq-mean-token-mean。函数注释特别区分 FSDP 和 Megatron 中 loss 反向所需缩放，意味着同一归约式经不同后端/微批数后还需确认尺度。[GRPO 组统计与 response advantage](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/trainer/ppo/core_algos.py#L268-L335)；[全局 loss aggregation](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/trainer/ppo/core_algos.py#L1140-L1206)。检查点包含 actor、critic、数据加载器状态并支持远端位置与保留数量策略。[Checkpoint 保存](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/trainer/ppo/v1/trainer_base.py#L949-L1010)。

### 3.6 slime：异步 rollout、可重放数据与 Megatron 训练

slime 将 rollout 服务与训练引擎分层：train.py 每轮向 Ray RolloutManager 请求生成批次，再将引用传给 actor 或 critic 模型执行异步训练；周期性保存训练状态并同步 actor 权重到推理端。RolloutManager 可从在线生成、调试数据或恢复记录读取样本，BatchBuilder 再把生成记录转换为训练张量并按数据并行切分。对在线样本，BatchBuilder 以 rollout id、分组 id、样本索引和数据处理配置形成可复放计划；重试时必须复用原始选择和转换方案，避免在崩溃恢复中悄悄改变批次。[主训练循环、权重加载和检查点调用](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/train.py#L13-L83)；[RolloutManager 生成与恢复](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/ray/rollout.py#L317-L390)；[批次计划和奖励后处理](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/data/batch_builder.py#L48-L110)；[按提示组处理唯一 rollout 奖励](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/data/batch_builder.py#L207-L254)。

在 SGLang 异步 rollout 中，样本任务可以并发生成、执行奖励模型与全组评分；采样程序保留 rollout id、结束状态及权重版本等信息。RolloutBatch 进入 Megatron actor 后，根据配置选择行为 log-prob、ref log-prob、KL、values、response length 与 loss mask；优势函数支持 GRPO/GSPO/CISPO、PPO 和 REINFORCE++ 等。policy_loss_function 再按配置计算 clipped PPO、序列级目标或 importance sampling 修正，loss_function 对 Megatron microbatch 和并行组缩放。权重版本、组采样随机种子、分段轨迹和恢复计划都是异步系统可复现性的必要数据，不可只保存最终文本。[异步生成及整组 reward](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/rollout/sglang_rollout.py#L388-L439)；[rollout 入口与任务收集](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/rollout/sglang_rollout.py#L492-L545)；[优势估计器分派](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/backends/megatron_utils/loss.py#L725-L850)；[策略损失与并行归约](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/backends/megatron_utils/loss.py#L1128-L1245)；[actor 训练与模型保存](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/backends/megatron_utils/actor.py#L400-L485)。

这种流水线可提高大规模推理与训练的设备利用率，但更依赖样本版本管理、rollout/训练策略偏差纠正、队列持久性和 checkpoint 的联合语义。项目近期代码也明确区分了训练完成事件和训练状态已持久保存；恢复时使用同一批原始样本和转换计划，与仅从“最后一个 checkpoint”重启的简单本地训练并不相同。

### 3.7 open-r1：把任务奖励接入通用 GRPO，而非自定义优化器

open-r1 的 grpo.py 将数据集中的 problem 转成用户消息格式，构造 tokenizer、模型和 reward 函数注册表，再将它们交给 TRL GRPOTrainer。其 setup.py 将 trl[vllm] 固定为 0.18.0，所以这一配方实际调用的 GRPO 内核应以 TRL v0.18.0 的源码解释，而不能用当前 TRL main 的新增代码替代。该版本把 completion mask 截在首个 EOS，可配置屏蔽没有 EOS 的截断响应；对全局 gather 后的奖励按每组 num_generations 重排并计算均值、标准差，传入优势；loss 内含 GRPO、BNPO、DR-GRPO 三种归一化分母，beta 非零时还以 reference log-prob 构造逐 token KL 惩罚。其重要性比率在生成时缓存 old_per_token_logps，训练时与当前策略逐 token 对比。accuracy_reward 解析模型输出的 LaTeX 答案并与 gold 核验；format_reward 检查 think/answer 标签和输出结构；其他奖励函数可处理长度、推理步骤、重复和代码执行。配置中的 Qwen2.5-1.5B 示例采用 16 个 generation、1024 个最大 completion token，并启用 vLLM。因此该仓库的主要贡献是奖励、数据、工程配置和任务配方；GRPO 的优化内核来自其固定依赖的 TRL 版本。入口调用 Trainer.train 后记录训练指标、保存 Trainer state 与模型；保存策略亦由传入的训练配置控制。[依赖锁定](https://github.com/huggingface/open-r1/blob/5b6ff22b3fb7aa069c54866e517f39dfc3160e09/setup.py#L72)；[数据映射及 GRPOTrainer 初始化](https://github.com/huggingface/open-r1/blob/5b6ff22b3fb7aa069c54866e517f39dfc3160e09/src/open_r1/grpo.py#L73-L124)；[训练、state 与模型保存](https://github.com/huggingface/open-r1/blob/5b6ff22b3fb7aa069c54866e517f39dfc3160e09/src/open_r1/grpo.py#L123-L147)；[v0.18.0 的实际采样、奖励组统计与 loss](https://github.com/huggingface/trl/blob/ef4b0b225cf04600332a12a2c83f187623312098/trl/trainer/grpo_trainer.py#L983-L1216)；[v0.18.0 的 old/ref log-prob、KL 与三类归一化](https://github.com/huggingface/trl/blob/ef4b0b225cf04600332a12a2c83f187623312098/trl/trainer/grpo_trainer.py#L1324-L1378)；[可验证答案与格式奖励](https://github.com/huggingface/open-r1/blob/5b6ff22b3fb7aa069c54866e517f39dfc3160e09/src/open_r1/rewards.py#L40-L111)；[示例 generation 与长度配置](https://github.com/huggingface/open-r1/blob/5b6ff22b3fb7aa069c54866e517f39dfc3160e09/recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml#L1-L52)。

复现实验时，必须把配方配置和 TRL 依赖版本一起锁定。奖励函数内容、reward_weights、completion 最大长度、生成数和 vLLM 采样分布都是算法有效输入；改动奖励语义或版本后，不能将结果解释为对原配方的直接复现。此处审阅的是 open-r1 公开配方及其实际调用的 TRL 接口，不是 DeepSeek-R1 私有/未公开训练实现的替代证据。

### 3.8 MiniMind：适合教学和本地改造的直接训练回路

MiniMind 在同一仓库提供 train_dpo.py、train_ppo.py 与 train_grpo.py。DPO 脚本载入 policy 与冻结参考模型，DPODataset 提供 chosen/rejected 输入与 response mask，logits_to_log_probs 先计算 token 对数概率，dpo_loss 依据掩码求序列概率和及 reference-adjusted preference margin，然后使用 AdamW、梯度累积、AMP 与保存恢复状态。它直接展示了 DPO 的数据接口、两路前向和端到端反向过程，方便小规模验证，但也意味着独立复制一份完整 reference 权重会占用相当显存。[DPO 目标及 mask 使用](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/train_dpo.py#L45-L94)；[冻结参考模型与 DDP 初始化](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/train_dpo.py#L173-L214)。

GRPO 脚本以 rollout_engine 统一 PyTorch 原生生成和 SGLang HTTP 采样接口，采样时保留每 token log-prob、prompt 长度和 completion mask。奖励结合格式/长度/重复规则与 reward model 分数，按每个提示的 n 个回答减组均值并除以组标准差，随后比较当前模型与采样时 old_logps 的概率比，并用参考模型 KL 与 clipped ratio 形成 token loss。loss 先在每条序列上按 completion_mask 求平均，再在样本维平均，所以每个回答总权重相同；该分母不同于 TRL 的 BNPO token 全局平均，也不同于 DR-GRPO 固定最大长度分母。反向后执行裁剪、optimizer/scheduler step，并在保存点刷新 rollout engine 的 policy。checkpoint 同时保存权重及优化器、调度器、epoch/step 状态，便于恢复训练。[奖励和组归一化](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/train_grpo.py#L37-L68)；[rollout、掩码、ratio 与 GRPO 损失](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/train_grpo.py#L71-L150)；[RolloutEngine 与策略同步接口](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/rollout_engine.py#L23-L94)；[断点恢复、混合精度与 DDP](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/train_grpo.py#L245-L322)。

## 4. 三个对复现影响最大的实现分歧

### 4.1 “旧策略”不是“参考策略”

PPO 的 ratio 必须明确分母是哪一时刻的策略。DeepSpeed-Chat 在一次采样后保存 actor log-prob，之后以它作为 old log-prob；ref log-prob 则进入 KL 惩罚。OpenRLHF 的 Experience 对象把 actor 行为概率和 base/reference 概率分开保存。verl 的 decoupled mode 更进一步区分采样策略、一次 batch 的 PPO anchor 与当前参数；bypass 模式将前两者合并。TRL 当前 GRPO 可能还同时保存 sampling log-prob、old training log-prob、reference log-prob。将这些张量名都简写为 old/ref 而不说明创建时刻、梯度是否关闭和更新时长，会使 off-policy 修正与 KL 正则被错误比较。

### 4.2 token 均值与序列均值改变样本权重

令第 i 个回答有效长度为 Ti、每 token 损失为 ℓit。token 平均按所有有效动作求和除以总 token 数，样本权重与长度近似成正比；先对每条序列求均值再对序列平均时，短回答与长回答的总梯度权重更接近。TRL grpo 的每序列 token 均值、TRL bnpo 的有效 token 总均值、DR-GRPO 的 batch size 乘固定完成长度分母、OpenRLHF 的 token_level_loss 开关、verl 的五类聚合、MiniMind 的每序列均值，均可导致不同更新幅度。多卡时还要使用全局分母补偿 DDP/FSDP 的梯度平均，不能在每个 rank 上各自归一后就假定等价。复现表至少应列出 loss reducer、分母、完成 token 定义、是否跨 rank 聚合以及末尾不足批次如何处理。

### 4.3 推理引擎产生的样本可能已是 off-policy 数据

当模型在专用 vLLM 或 SGLang 服务采样、训练在另一套 kernel/精度或更新后的权重上计算 log-prob 时，采样策略与训练策略可能存在差异。TRL 在 sampling_per_token_logps、old_per_token_logps 和 importance_sampling_ratio 之间显式校正；OpenRLHF 对 rollout/train 概率差提供 token/sequence gating；verl 支持 bypass 与 decoupled old-policy 模式和 rollout correction；slime 记录采样 weight_version 并在异步数据缓冲区管理批次；MiniMind SGLang engine 从服务返回采样 token log-prob。重要性比率的乘法、裁剪或 rejection mask 会同时改变偏差与方差，特别是序列级比率对长回答更敏感。报告必须注明采样引擎、权重同步频率、rollout-logprob 来源与校正方式。

### 4.4 离线 DPO 与在线 DPO 的数据语义

标准 DPO 的输入是一组已经存在的偏好回答。策略和参考模型分别对 chosen、rejected completion 计算条件序列概率，并以两者相对参考模型的对数比差构造分类式目标。数据偏好关系因此先于本轮梯度更新；如果偏好来自人工标注、合成 judge 或规则筛选，数据生产链和训练器可以独立重放。TRL 与 DeepSpeed-Chat 的 DPO 代码均以 completion label mask 排除 prompt 和 padding，且把响应 token log-prob 求和。该求和不是任意的编码便利：在不同长度候选间，较长序列的累计负对数概率尺度更大，因而实现损失、数据截断和 prompt 模板共同构成可观察的训练条件。

在线 DPO 将偏好生成带入训练循环，但没有把奖励值本身作为策略梯度。TRL 在线版本对当前策略采样两条回复，逐条调用奖励函数并比较得分；奖励大小只决定哪条是 chosen、哪条是 rejected，随后仍依据 policy/reference 序列概率差更新。它仍需定义参考模型来源、两条回复的采样温度、奖励相等时的 tie 规则、EOS 缺失惩罚与采样对数概率如何使用。与 PPO 相比，它省去 value head、GAE 和 old-policy clipping，却增加在线生成和奖励评估成本；与离线 DPO 相比，它的偏好数据会随当前策略变化。在线 DPO 和 PPO 因而都是在线采样，但一个以排序过的 pair 做对数似然分类，另一个以奖励、优势及概率比构造策略梯度，不能按“均使用奖励模型”简单归为同一目标。[TRL 标准 DPO 概率比与 reference scoring](https://github.com/huggingface/trl/blob/c886cf23d74acd8dfed95b63943a8dba0f25d85f/trl/trainer/dpo_trainer.py#L1337-L1415)；[DeepSpeed-Chat 的成对偏好更新](https://github.com/deepspeedai/DeepSpeedExamples/blob/0c6dd5ed1ec0f0967ae0564534ecfa6f157c6746/applications/DeepSpeed-Chat/training/step2_dpo_finetuning/main.py#L219-L240)。

### 4.5 GRPO 组基线中的小样本方差差异

GRPO 把同一 prompt 的多条回答视作局部比较集合，优势取各自 reward 减去组基线。不同实现对分组 identity、std 估计与 edge case 有实质不同的处理。TRL 以生成顺序和 num_generations 重塑奖励，支持组标准差归一、只减均值或按 batch 估计尺度；样本不可评分时由 NaN-aware 均值排除，并使其优势归零。verl 则利用 uid 逐条恢复提示组，配置 norm_adv_by_std_in_grpo 控制除组标准差；仅有一个成员时直接用均值零、标准差一，避免无法定义样本标准差。MiniMind 使用 std(unbiased=False)，即组内总体标准差。OpenRLHF 的 torch.std 使用框架默认样本标准差；TRL v0.18.0 的 std_grouped_rewards 也通过未指定 correction 的 torch.std 计算，所以 open-r1 锁定该版本时继承同一约定。若组内只采两条回复，样本标准差相对总体标准差会按平方根二扩大；保持 reward 不变、只切换 std 估计约定，就会改变每一批优势和策略梯度大小。

slime 的样本可被多轮对话切成多个训练 segment，因此不以 segment 数直接作为组内独立回答数；BatchBuilder 按 rollout id 恢复唯一生成结果，检查同一 rollout 的分段样本必须具有同组身份和一致 outcome reward，再对唯一 rollout reward 做中心化/可选标准化，并把相同结果映射回各训练片段。若把 segment 展平后重新分组，长对话会因产生更多片段而获得额外权重，提示组的统计对象也会变成 segment 而非采样回答。复现 GRPO 时至少应保存每个生成回答的 prompt id、rollout id、raw reward、组均值/标准差、优势以及每个训练 segment 对应的动作掩码。[TRL 的可评分组处理](https://github.com/huggingface/trl/blob/c886cf23d74acd8dfed95b63943a8dba0f25d85f/trl/trainer/grpo_trainer.py#L2797-L2835)；[verl 的单样本组与 std 处理](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/trainer/ppo/core_algos.py#L304-L335)；[slime 对 unique rollout 统计](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/data/batch_builder.py#L207-L254)。

### 4.6 角色分离、队列与 checkpoint 的可复现性

大规模 RLHF 系统不只是把一个 for-loop 横向扩容。OpenRLHF 和 DeepSpeed-Chat 显式分配 actor、reference、reward model 与 critic，分别在 rollout 和优化阶段执行不同前向；这能让模型共置、卸载和并行策略被单独配置，但 checkpoint 需同步检查 actor、critic、优化器以及 vLLM 权重。verl 将 rollout data、log-prob、reward、advantage 等字段写入 Transfer Queue，训练器可恢复 replay buffer 与输入数据加载状态；批次键和字段版本是训练语义一部分。slime 将原始 rollout 及转换计划持久化，并在重放时保留 batch identity；代码注释特别区分 training_completed（释放队列容量）与训练参数和进度已经被联合 checkpoint 的含义。

这一差别对复现实验尤为重要：如果生成服务先于 actor 更新多个批次，后续样本可能对应旧的权重版本；如果进程在转换 raw rollout 后、actor 保存前退出，重新采样不会产生相同数据。仅恢复参数 checkpoint 并从数据集游标继续，不能保证复现原训练轨迹。必须明确记录的是“可继续优化”还是“可逐样本重放”，以及如何处理未完成 rollout、reward RPC 超时、权重更新迟到与 optimizer/critic 恢复的不一致。同步的 MiniMind 训练脚本在本地实验中更容易固定每批次只采样一次、更新一次并立刻保存状态；它降低了调度复杂度，但也没有异步方案的吞吐收益。[verl 的 replay buffer 选择与异步设置](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/trainer/ppo/v1/trainer_base.py#L140-L183)；[slime 对训练完成与持久化状态的区分](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/train.py#L56-L83)。

### 4.7 奖励函数、数据模板与被评估行为

奖励不是算法之外的配置附件，而是将任务语义写入目标的接口。open-r1 的 accuracy reward 只有在 gold 答案可解析且输出能按规则提取时才返回数值；无法解析时返回 None。其固定依赖 TRL v0.18.0 会先把 None 转为 NaN，再对奖励函数维度执行 nansum；当所有奖励函数均无分时，聚合奖励变为零并进入组均值/标准差计算，代码同时发出 warning。TRL current main 则通过 unscorable_mask 将全 None 样本排除在 NaN-aware 组均值之外，并把其优势置零。这个差异意味着 open-r1 的配方不能直接套用 current main 的不可评分处理结论。格式奖励则检查 think/answer 结构，代码奖励可并发执行测试用例。其配置把准确率、格式和标签计数赋以明确权重，三个函数的和会改变组内 reward 方差，进而改变 GRPO advantage，而不是只影响日志。[TRL v0.18.0 的 None 处理和 nansum](https://github.com/huggingface/trl/blob/ef4b0b225cf04600332a12a2c83f187623312098/trl/trainer/grpo_trainer.py#L1171-L1208)；[TRL current main 的 unscorable mask](https://github.com/huggingface/trl/blob/c886cf23d74acd8dfed95b63943a8dba0f25d85f/trl/trainer/grpo_trainer.py#L2800-L2835)。

MiniMind 的 GRPO 奖励则把长度、思考区间长度、思考结束标签次数、重复惩罚和奖励模型得分相加。规则分数与 reward model 分数的取值尺度若不校准，某一项可能主导组排序；格式项的重叠也可能重复奖励同一种行为。因此比较方法时要保存奖励分项和聚合前原始值，并说明哪些奖励可程序核验、哪些依赖另一个语言模型。统一 prompt 模板、chat template、答案解析器、EOS/截断处理和奖励版本后，才能将差异归因于 DPO/PPO/GRPO 的优化结构；否则数据与奖励变动可能比 loss type 更能解释性能差异。[open-r1 可核验奖励及不可评分返回](https://github.com/huggingface/open-r1/blob/5b6ff22b3fb7aa069c54866e517f39dfc3160e09/src/open_r1/rewards.py#L40-L111)；[MiniMind 规则与奖励模型得分组合](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/train_grpo.py#L37-L68)。

## 5. 面向 6 GB 显存本地实验的设计建议

在 6 GB RTX 3060 Laptop 上，优先选 MiniMind 这类显式单机训练回路作为方法承载骨架，先比较 DPO 与一个极小批次的 GRPO，而不是先引入 Ray、Megatron、ZeRO 多角色系统。DPO 无在线生成，训练前可预计算冻结 reference 的偏好概率，或采用 PEFT adapter 切换/参考策略实现以减少双模型常驻；必须保留 chosen/rejected completion mask，且不改变用于正文报告的序列 log-prob 求和定义。若参照 MiniMind 原版 reference 模型双份加载，先选小参数量模型并缩短 prompt 与 answer，再测显存。

GRPO 每轮同时涉及策略 rollout、奖励计算、策略反向与参考 KL。建议用 prompt batch 1–2、每提示 2 条 completion、短回答上限（例如 64–128 token）开始，梯度累积只扩大优化批次而不增加同一时刻的采样张量；先用原生 PyTorch rollout 减少推理服务和权重同步复杂度。若需要 SGLang，应确保其采样概率及权重版本可记录，并在训练前做单批次 log-prob 对齐检查。reward model 若与 actor 同时驻留会进一步挤占显存，可选择轻量/CPU 奖励服务或任务可核验的规则奖励；但三者回答的是不同奖励来源，报告应明确区分，不能把规则奖励结果说成奖励模型反馈。

预实验应先运行单个前向和一组采样，确认 dtype、attention mask、EOS 位置、padding 与 log-prob 对齐，再逐步扩充更新数。保留逐样本原始输出、逐响应长度、reward 组成、KL、有效 token 数及峰值显存；这些记录可发现完成掩码错位、组内采样顺序破坏或 rollout 权重滞后。6 GB 只是设备容量，不等价于某个配置可运行的保证；实际可行范围还取决于参数量、优化器状态、激活检查点、序列长度、是否冻结/量化参数、采样引擎和 reward 模型。

## 6. 结论

源码审阅显示，后训练算法从论文目标到可复现实验之间存在一层不可忽略的训练器语义：优势估计器决定基线和归一范围；old 与 reference 概率的构造决定 clipping 与 KL 的对象；completion/action mask 决定参与优化的 token；loss reduction 决定长短样本的相对权重；分布式全局计数、rollout 权重版本与恢复数据计划又决定同一配置在多设备执行时是否保持目标不变。TRL 和 open-r1 提供较高层的算法与奖励配方；trlx、DeepSpeed-Chat 保留相对直接的 PPO actor/critic 路径；OpenRLHF、verl、slime 则进一步把生成服务、策略概率校正、全局批处理及分布式同步纳入训练方法本身；MiniMind 展示了较紧凑的单机可改造回路。

本章只依据所列公开仓库在对应锁定版本中的代码路径作出结论。仓库名、模型结果或论文叙述不能证明使用了某个具体训练实现。DeepSeek-R1 的报告和论文可用于讨论方法背景，但本次没有把其方法描述当作公开训练源码证据，也不将其他开源框架的代码反向归属为 DeepSeek-R1 官方实现。

## 参考文献

1. Schulman, J. et al. Proximal Policy Optimization Algorithms. 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347).
2. Schulman, J. et al. High-Dimensional Continuous Control Using Generalized Advantage Estimation. 2015. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438).
3. Rafailov, R. et al. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290).
4. Shao, Z. et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300).
5. Liu, Z. et al. Understanding R1-Zero-Like Training: A Critical Perspective. 2025. [arXiv:2503.20783](https://arxiv.org/abs/2503.20783).
6. Zheng, C. et al. Group Sequence Policy Optimization. 2025. [arXiv:2507.18071](https://arxiv.org/abs/2507.18071).
7. Hugging Face. TRL source repository and trainers. [Repository](https://github.com/huggingface/trl).
8. CarperAI. trlx source repository. [Repository](https://github.com/CarperAI/trlx).
9. Microsoft. DeepSpeed-Chat source within DeepSpeedExamples. [Repository](https://github.com/deepspeedai/DeepSpeedExamples).
10. OpenRLHF contributors. OpenRLHF source repository. [Repository](https://github.com/OpenRLHF/OpenRLHF).
11. Volcengine. verl source repository. [Repository](https://github.com/volcengine/verl).
12. THUDM. slime source repository. [Repository](https://github.com/THUDM/slime).
13. Hugging Face. open-r1 source repository. [Repository](https://github.com/huggingface/open-r1).
14. Jingyao Gong. MiniMind source repository. [Repository](https://github.com/jingyaogong/minimind).
