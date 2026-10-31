# TODO — ADOPD v2 efficient reasoning

更新时间：2026-10-31 02:20 UTC。集群：2 × 8 H100（Ray head + workera），所有产物都在容器内。

## 进行中

- [ ] Qwen3-8B 4k（256 × 56，训练中 eval 跑 32k/16k/forced）—— 01:38 UTC 开始，预计 ~07:00 完成
- [ ] Qwen3-8B 2k（同上）
- [ ] Qwen3-8B base 的 step-0 eval：3 个 budget prompt × 32k/16k/forced（`tmp/eval_merged8b/base_step0.yaml`）
- [ ] Qwen3-14B teacher cache（teacher TP 1，16 个副本）
- [ ] Qwen3-14B 8k：只做 32k eval，step 0 含 plain（base model）对照

以上由 `tmp/chain_8b_14b_v2.sh` 串行执行；启动阶段失败（端口竞争）自动重试 3 次；每个 run 结束后删除 FSDP 分片、保留 HF 权重。

## 待办

- [ ] 把 8B 训练中的 `eval_32k/`、`eval_16k/`、`eval_forced/` 以及 8B base 的 step-0 结果，按 `eval_avg8_*` 命名写入三个 8B run，和 4B 面板一致
- [ ] 14B 的 32k 结果同样写成 `eval_avg8_32k/`（step 0 含 `_plain`）
- [ ] 汇总 4B / 8B / 14B 的长度缩短与 accuracy 表，更新 Google Slides（第 21/24/25 页仍是 16k 口径）
- [ ] 端口竞争的根治：verl 在临时端口段（32768–63999）里选 `MASTER_PORT` 后再 bind，会被其他进程抢占（`EADDRINUSE` / TCPStore 30 分钟超时）。给 worker group 传 `master_port_range`（< 32768），并保证并发创建的 group 不重叠
- [ ] 把可复用的运维脚本从 `tmp/`（gitignored）提升到 `scripts/`：eval 矩阵生成、wandb 合并日志、FSDP 分片清理、本文的表格生成（`tmp/report_tables.py`）
- [ ] 两个 `test_sbatch_reads_yaml_allocation_and_launches_all_nodes` 测试依赖 `/scratch` 路径，在本集群上一直失败
- [ ] worker 磁盘：`/opt/tiger/refine_opsd_persist`（1.36T，非本项目）是否删除待定

---

## 实验配置

**方法（ADOPD v2）**
- student：带 budget prompt（“reasoning budget B tokens”）生成。推理到 B 个 token 仍未闭合时强制插入 `</think>`，再生成答案；总长 ≤ 16384。
- teacher：冻结的同一个 base model。teacher 的完整推理轨迹预先离线生成并缓存（最长 16k，seed 42）。
- loss：
  - **answer-only KL**：只在 student 的答案 token 上，用 sampled-token k1 估计，对齐到“teacher 在自己完整推理之后”给该答案的分布。
  - **长度项**：对 student 推理 token 做 single-rollout policy gradient，cost = λ · L_trace / B，λ = 0.1。
  - 每个 prompt 一条 rollout（g1），不用 group。

**训练设置（所有 run 相同）**

| 项 | 值 |
|---|---|
| 数据 | DAPO-Math-17k 英文子集，14,016 prompt，1 epoch |
| batch | 256 prompt / step，56 步；一次 rollout 一次更新（mini-batch = 全 batch，ppo_epochs 1） |
| 优化 | lr 1e-6，warmup 25 步，weight decay 0.01，grad clip 1.0，无 ref KL |
| 长度 | prompt ≤ 2048，response ≤ 16384，teacher 轨迹 ≤ 16384 |
| 采样 | temperature 1.0，top_p 1.0 |
| 并行 | 4B：每个 vLLM 副本 1 卡（16 副本 × 16 prompt）；8B/14B：student/teacher TP 2（8 副本 × 32 prompt） |
| 框架 | verl 0.8（vendored）、vLLM 0.11（FLASH_ATTN + CUDA graph）、FSDP |

**评估协议**
- 数据：avg@8，6 个数据集，汇总按题数加权：AIME25（30）、AIME26（30）、HMMT26（33）、AMC23（40）、GPQA 普通化学（20）、GPQA 普通物理（19）。
- 三种 eval budget，都用 budget prompt，并且都从同一条 32k rollout 得到：
  - `32k`：rollout 原样。
  - `16k`：截断到 16384 token。
  - `forced`：在训练 budget B 内已闭合则原样截到 16k；否则保留前 B 个推理 token，插入 `</think>`，只续写答案（≤ 16k）。
- plain（不加 budget prompt）只在 base model（step 0）上跑，作为主对照。
- 结果写进对应训练 run 的 wandb，前缀 `eval_avg8_{32k,16k,forced}/`，横轴 `train_step`。

**wandb run**（项目 `li-jiang-research/efficient-reasoning`）

| run | id |
|---|---|
| 4B 8k | `21df9cfe` `qwen3_4b_adopd_v2_sample_g1` |
| 4B 4k | `5f2273ed` `qwen3_4b_adopd_v2_sample_g1_b4096` |
| 4B 2k | `b1c97764` `qwen3_4b_adopd_v2_sample_g1_b2048` |
| 8B 8k | `3a0ce5dd` `qwen3_8b_adopd_v2_sample_g1_b8192_bs256` |

## 已完成实验

### 训练指标（前 5 步 vs 最后 5 步的平均）

- train acc：训练 rollout 的答案正确率。
- resp len = reasoning + answer。
- forced：被强制插入 `</think>` 的比例。
- s/step：不含 eval 的单步耗时。

| run | steps | train acc | resp len | reasoning | answer | forced | answer KL | len cost | s/step |
|---|---|---|---|---|---|---|---|---|---|
| 4B 8k 256x56 | 1–5 | 0.759 | 8399 | 6675 | 1723 | 0.52 | 0.085 | 0.081 | 261 |
| 4B 8k 256x56 | 52–56 | 0.874 | 5960 | 5015 | 944 | 0.22 | 0.061 | 0.061 | 255 |
| 4B 4k 256x56 | 1–5 | 0.705 | 6200 | 3878 | 2321 | 0.82 | 0.100 | 0.095 | 250 |
| 4B 4k 256x56 | 52–56 | 0.812 | 4666 | 3470 | 1195 | 0.58 | 0.073 | 0.085 | 245 |
| 4B 2k 256x56 | 1–5 | 0.634 | 5187 | 2036 | 3150 | 0.97 | 0.109 | 0.099 | 249 |
| 4B 2k 256x56 | 52–56 | 0.750 | 3342 | 1990 | 1352 | 0.90 | 0.073 | 0.097 | 231 |
| 8B 8k 256x56 | 1–5 | 0.722 | 9683 | 7012 | 2671 | 0.59 | 0.086 | 0.086 | 224 |
| 8B 8k 256x56 | 52–56 | 0.890 | 7082 | 5674 | 1408 | 0.31 | 0.066 | 0.069 | 197 |
| 8B 8k 128x112（消融） | 1–5 | 0.709 | 9955 | 7130 | 2823 | 0.62 | 0.091 | 0.087 | 168 |
| 8B 8k 128x112（消融） | 108–112 | 0.833 | 4882 | 3758 | 1122 | 0.16 | 0.070 | 0.046 | 142 |
| 8B 4k 128x112（消融） | 1–5 | 0.692 | 8063 | 4006 | 4056 | 0.91 | 0.096 | 0.098 | 157 |
| 8B 4k 128x112（消融） | 108–112 | 0.864 | 4600 | 3442 | 1156 | 0.61 | 0.065 | 0.084 | 126 |

总耗时：
- 4B 每个 run 5.1–5.8 h（中途 eval 按旧的 16k avg@4 口径跑）。
- 8B 8k 256×56 共 4.6 h，含 3 次 32k/16k/forced eval，每次约 24 分钟。

### Qwen3-4B eval 矩阵（acc / 平均长度）

base（step 0，plain，32k）：acc 0.699，长度 13,692。step 0 = base model 加对应 run 的 budget prompt。

| run | eval | step 0 | step 20 | step 40 | step 56 | Δlen step 56 vs base 同 prompt | Δlen step 56 vs base plain(32k) |
|---|---|---|---|---|---|---|---|
| 4B 8k | 32k | 0.690 / 12.5k | 0.677 / 12.6k | 0.677 / 11.9k | 0.665 / 11.4k | -8.6% | -16.6% |
| 4B 8k | 16k | 0.586 / 10.1k | 0.583 / 10.3k | 0.592 / 9.9k | 0.608 / 9.5k | -6.2% | — |
| 4B 8k | forced | 0.601 / 8.5k | 0.598 / 8.5k | 0.612 / 8.0k | 0.605 / 7.6k | -10.4% | — |
| 4B 4k | 32k | 0.678 / 11.8k | 0.677 / 12.0k | 0.680 / 12.0k | 0.672 / 11.6k | -1.8% | -15.4% |
| 4B 4k | 16k | 0.601 / 9.9k | 0.597 / 10.0k | 0.591 / 9.9k | 0.589 / 9.6k | -2.9% | — |
| 4B 4k | forced | 0.570 / 6.4k | 0.554 / 6.3k | 0.543 / 6.1k | 0.551 / 5.6k | -12.1% | — |
| 4B 2k | 32k | 0.677 / 12.1k | 0.687 / 12.0k | 0.680 / 11.9k | 0.674 / 12.0k | -0.6% | -12.2% |
| 4B 2k | 16k | 0.592 / 9.9k | 0.598 / 10.0k | 0.592 / 9.9k | 0.597 / 9.9k | +0.2% | — |
| 4B 2k | forced | 0.528 / 5.5k | 0.528 / 5.5k | 0.493 / 4.5k | 0.477 / 4.1k | -26.1% | — |

base 的 plain 在 16k 下：acc 0.584，长度 10.9k。

### Qwen3-8B 8k（256 × 56）

base（step 0，32k）：plain acc 0.706 / 14,103；8k prompt acc 0.693 / 13,497。

| eval | step 20 | step 40 | step 56 | step 56 forced 比例 | Δlen step 56 vs base plain(32k) |
|---|---|---|---|---|---|
| 32k | 0.709 / 13.5k | 0.703 / 13.2k | 0.690 / 12.6k | — | -10.5% |
| 16k | 0.588 / 10.7k | 0.593 / 10.6k | 0.602 / 10.3k | — | — |
| forced | 0.614 / 9.4k | 0.616 / 9.2k | 0.621 / 8.9k | 0.54 | — |

按数据集（32k，step 56 vs base plain）：

| 数据集 | base acc | step 56 acc | 长度 base | 长度 step 56 | Δlen |
|---|---|---|---|---|---|
| aime25 | 0.658 | 0.671 | 17,929 | 16,130 | -10.0% |
| aime26 | 0.704 | 0.650 | 16,484 | 15,023 | -8.9% |
| hmmt26 | 0.405 | 0.443 | 20,152 | 18,318 | -9.1% |
| amc23 | 0.975 | 0.931 | 8,807 | 7,985 | -9.3% |
| gpqa_chemistry_general | 0.588 | 0.588 | 10,461 | 8,824 | -15.6% |
| gpqa_physics_general | 0.862 | 0.809 | 8,777 | 7,116 | -18.9% |

8B base 的 16k / forced 结果要等 8B base step-0 eval 跑完。

### 消融：8B 128 × 112（每步 128 prompt、112 步，同样 1 epoch）

eval 为 32k、avg@8、budget prompt。权重已删，eval 结果保留。

| run | step | acc | 长度 | Δacc vs base plain | Δlen vs base plain |
|---|---|---|---|---|---|
| 8B 8k 128×112 | 20 | 0.707 | 13,366 | +0.001 | -5.2% |
| 8B 8k 128×112 | 60 | 0.691 | 12,425 | -0.015 | -11.9% |
| 8B 8k 128×112 | 100 | 0.691 | 11,542 | -0.015 | -18.2% |
| 8B 8k 128×112 | 112 | 0.547 | 6,895 | -0.159 | -51.1% |
| 8B 4k 128×112 | 20 | 0.722 | 13,515 | +0.016 | -4.2% |
| 8B 4k 128×112 | 60 | 0.698 | 12,911 | -0.008 | -8.5% |
| 8B 4k 128×112 | 100 | 0.701 | 12,041 | -0.005 | -14.6% |
| 8B 4k 128×112 | 112 | 0.685 | 11,879 | -0.021 | -15.8% |

更新次数翻倍后，8k 在最后十几步塌缩（训练 rollout 长度从 10k 掉到 3.7k，forced 比例从 0.71 降到 0.09）；4k 没有塌。

### 目前的结论

- 训练时强制截断后，学到的主要是**更短的答案**：4B 2k 的答案长度从 3150 降到 1352。forced 比例下降不多（4B 2k 从 0.97 到 0.90），说明模型没怎么学会在 budget 内自己结束思考。
- 自由生成（32k）下的缩短，主要来自 budget prompt 本身加上 8k run 的训练：
  - 相对 base plain：4B 8k/4k/2k 为 −16.6% / −15.4% / −12.2%，8B 8k 为 −10.5%。
  - 相对“base + 同一 budget prompt”：只有 8k 还有明显缩短（4B −8.6%，8B −6.5%）。
- forced eval（训练时的使用方式）下，长度缩短最明显，4B 2k 达到 −26%。代价是 accuracy 随 budget 变紧而下降：4B 2k 从 0.528 降到 0.477。8B 8k 的 forced acc 在 step 20 → 56 间反而上升（0.614 → 0.621）。

## 已完成（工程）

- [x] eval variants：一条 32k rollout 同时得出 32k/16k/forced（`adopd_eval_variants`）
- [x] teacher cache 改为 driver 统一写到 head（节点本地磁盘）
- [x] 4B eval 矩阵合并成 10 个 job（原 32 个），结果写入 3 个 4B 训练 run

## 注意事项

- 8B/14B 每个 rollout 副本占 2 张卡（TP 2），需要 `per_device_batch_size: 32` 才能和 4B 一样每步 256 prompt。
- 训练长度 16k、eval 长度 32k：verl 只把 eval response pad 到训练长度、不截断，所以 eval agent 返回截断后的 response，长度指标用完整 rollout 长度。
- wandb 里另有 5 个旧的 avg@4 eval-only run 和 1 个中止的 run，按要求保留未删。
