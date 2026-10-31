# TODO — ADOPD v2 efficient reasoning

更新时间：2026-10-31 21:30 UTC。集群：2 × 8 H100（Ray head + workera），所有产物都在容器内。

## 进行中

- [ ] Qwen3-14B 4k → 2k（256 × 56，32k eval，step 0 不 eval）—— 20:43 UTC 开始（`tmp/chain_14b_more.sh`）
- [ ] CRISP 基线（`tmp/chain_crisp.sh`，在 14B 之后自动开始，约 11/01 08:00 UTC）：
  - 4B 冒烟测试：2 次更新，在第 2 次更新前刷新 teacher；不通过就停下。
  - 4B → 8B → 14B 正式训练；每个 run 结束后在 step 20/40/56 做 eval（不加 budget prompt；32k，以及在 8k/4k/2k 处强制 `</think>`），然后删除 FSDP 分片。

两个串联脚本都在启动阶段失败时自动重试 3 次。

## 待办

- [ ] CRISP 的 eval 结果写入各自训练 run 的 wandb（`eval_avg8_32k/`、`eval_avg8_forced_b*/`），并补进本文
- [ ] 14B 4k/2k 的结果写入 wandb（`eval_avg8_32k/`），并补进本文
- [ ] 更新 Google Slides（第 21/24/25 页仍是 16k 口径），加入 4B/8B/14B 矩阵、跨模型对比、base forced 对比和实例分析
- [ ] 端口竞争的根治：verl 在临时端口段（32768–63999）里选 `MASTER_PORT` 后再 bind，会被其他进程抢占（`EADDRINUSE` / TCPStore 30 分钟超时）。给 worker group 传 `master_port_range`（< 32768），并保证并发创建的 group 不重叠
- [ ] 把可复用的运维脚本从 `tmp/`（gitignored）提升到 `scripts/`：eval 矩阵生成、wandb 合并日志、FSDP 分片清理、表格生成（`tmp/report_tables.py`）、实例分析（`tmp/case_study/`）
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
| 8B 4k | `155bdc61` `qwen3_8b_adopd_v2_sample_g1_b4096_bs256` |
| 8B 2k | `78db565f` `qwen3_8b_adopd_v2_sample_g1_b2048_bs256` |
| 14B 8k | `3c13e5bd` `qwen3_14b_adopd_v2_sample_g1_b8192_bs256` |

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
| 8B 4k 256x56 | 1–5 | 0.710 | 8236 | 3966 | 4268 | 0.89 | 0.096 | 0.097 | 213 |
| 8B 4k 256x56 | 52–56 | 0.869 | 5484 | 3650 | 1833 | 0.68 | 0.077 | 0.089 | 177 |
| 8B 2k 256x56 | 1–5 | 0.691 | 7587 | 2038 | 5548 | 0.98 | 0.104 | 0.100 | 206 |
| 8B 2k 256x56 | 52–56 | 0.814 | 4363 | 2000 | 2363 | 0.91 | 0.087 | 0.098 | 168 |
| 14B 8k 256x56 | 1–5 | 0.770 | 8553 | 6760 | 1792 | 0.53 | 0.073 | 0.083 | 278 |
| 14B 8k 256x56 | 52–56 | 0.922 | 6315 | 5258 | 1056 | 0.24 | 0.050 | 0.064 | 268 |
| 8B 8k 128x112（消融） | 1–5 | 0.709 | 9955 | 7130 | 2823 | 0.62 | 0.091 | 0.087 | 168 |
| 8B 8k 128x112（消融） | 108–112 | 0.833 | 4882 | 3758 | 1122 | 0.16 | 0.070 | 0.046 | 142 |
| 8B 4k 128x112（消融） | 1–5 | 0.692 | 8063 | 4006 | 4056 | 0.91 | 0.096 | 0.098 | 157 |
| 8B 4k 128x112（消融） | 108–112 | 0.864 | 4600 | 3442 | 1156 | 0.61 | 0.065 | 0.084 | 126 |

总耗时：
- 4B 每个 run 5.1–5.8 h（中途 eval 按旧的 16k avg@4 口径跑）。
- 8B 每个 run 4.3–4.6 h，含 3 次 32k/16k/forced eval（每次约 24 分钟）。
- 14B 8k 6.3 h，含 step 0（plain + budget）和 3 次 32k eval。

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

### Qwen3-8B eval 矩阵（acc / 平均长度）

step 0 = base model 加对应 run 的 budget prompt（8B base 合并 eval）。

base（step 0，plain，32k）：acc 0.706，长度 14,103

| run | eval | step 0 | step 20 | step 40 | step 56 | Δlen vs base 同 prompt | Δlen vs base plain(32k) |
|---|---|---|---|---|---|---|---|
| 8B 8k | 32k | 0.715 / 13.6k | 0.709 / 13.5k | 0.703 / 13.2k | 0.690 / 12.6k | -7.4% | -10.5% |
| 8B 8k | 16k | 0.581 / 10.8k | 0.588 / 10.7k | 0.593 / 10.6k | 0.602 / 10.3k | -4.8% | — |
| 8B 8k | forced | 0.602 / 9.6k | 0.614 / 9.4k | 0.616 / 9.2k | 0.621 / 8.9k | -6.8% | — |
| 8B 4k | 32k | 0.703 / 13.4k | 0.700 / 13.5k | 0.705 / 13.2k | 0.706 / 12.6k | -6.2% | -10.7% |
| 8B 4k | 16k | 0.584 / 10.7k | 0.581 / 10.7k | 0.589 / 10.5k | 0.613 / 10.1k | -5.2% | — |
| 8B 4k | forced | 0.589 / 8.1k | 0.578 / 7.8k | 0.581 / 7.5k | 0.596 / 7.2k | -10.8% | — |
| 8B 2k | 32k | 0.707 / 13.4k | 0.700 / 13.3k | 0.699 / 13.1k | 0.704 / 12.5k | -6.3% | -11.1% |
| 8B 2k | 16k | 0.594 / 10.6k | 0.579 / 10.6k | 0.600 / 10.4k | 0.611 / 10.1k | -4.9% | — |
| 8B 2k | forced | 0.561 / 7.4k | 0.558 / 7.2k | 0.547 / 6.4k | 0.547 / 5.7k | -22.6% | — |

8B 8k 按数据集（32k，step 56 vs base plain）：

| 数据集 | base acc | step 56 acc | 长度 base | 长度 step 56 | Δlen |
|---|---|---|---|---|---|
| aime25 | 0.658 | 0.671 | 17,929 | 16,130 | -10.0% |
| aime26 | 0.704 | 0.650 | 16,484 | 15,023 | -8.9% |
| hmmt26 | 0.405 | 0.443 | 20,152 | 18,318 | -9.1% |
| amc23 | 0.975 | 0.931 | 8,807 | 7,985 | -9.3% |
| gpqa_chemistry_general | 0.588 | 0.588 | 10,461 | 8,824 | -15.6% |
| gpqa_physics_general | 0.862 | 0.809 | 8,777 | 7,116 | -18.9% |

### Qwen3-14B 8k（32k eval）

| | acc | 长度 | Δlen vs base plain |
|---|---|---|---|
| base plain | 0.744 | 12,750 | — |
| base 8k prompt | 0.743 | 12,073 | -5.3% |
| step 20 | 0.743 | 12,262 | -3.8% |
| step 40 | 0.747 | 11,984 | -6.0% |
| step 56 | 0.738 | 11,713 | -8.1% |

按数据集（32k，step 56）：

| 数据集 | base plain | base 8k prompt | step 56 | Δacc vs plain | Δlen vs plain | Δlen vs 8k prompt |
|---|---|---|---|---|---|---|
| aime25 | 0.725 / 16,690 | 0.717 / 16,170 | 0.696 / 15,755 | -0.029 | -5.6% | -2.6% |
| aime26 | 0.738 / 14,939 | 0.708 / 14,747 | 0.713 / 14,263 | -0.025 | -4.5% | -3.3% |
| hmmt26 | 0.485 / 18,638 | 0.477 / 17,984 | 0.462 / 17,542 | -0.023 | -5.9% | -2.5% |
| amc23 | 0.991 / 7,789 | 0.988 / 7,067 | 0.994 / 6,718 | +0.003 | -13.8% | -5.0% |
| gpqa_chemistry_general | 0.675 / 8,982 | 0.731 / 7,652 | 0.713 / 7,505 | +0.037 | -16.4% | -1.9% |
| gpqa_physics_general | 0.789 / 7,255 | 0.803 / 6,310 | 0.809 / 6,127 | +0.020 | -15.5% | -2.9% |

### 跨模型对比（32k，step 56）

| model | base plain acc / 长度 | budget | step 56 acc | Δacc vs plain | Δlen vs plain | Δlen vs base 同 prompt |
|---|---|---|---|---|---|---|
| 4B | 0.699 / 13,692 | 8k | 0.665 | -0.034 | -16.6% | -8.6% |
| 4B | 0.699 / 13,692 | 4k | 0.672 | -0.027 | -15.4% | -1.8% |
| 4B | 0.699 / 13,692 | 2k | 0.674 | -0.025 | -12.2% | -0.6% |
| 8B | 0.706 / 14,103 | 8k | 0.690 | -0.016 | -10.5% | -7.4% |
| 8B | 0.706 / 14,103 | 4k | 0.706 | +0.001 | -10.7% | -6.2% |
| 8B | 0.706 / 14,103 | 2k | 0.704 | -0.001 | -11.1% | -6.3% |
| 14B | 0.744 / 12,750 | 8k | 0.738 | -0.007 | -8.1% | -3.0% |

4B/8B 的“同 prompt”基线来自各自 base 的合并 eval；8B base 的 8k prompt 两次 eval 分别为 13,497（128×112 run 的 step 0）和 13,620（合并 eval）。

### base model 强制 `</think>`（4B / 8B / 14B × 2k / 4k / 8k）

不训练，用对应 budget 的 prompt；同一条 rollout 若在 budget 内没结束思考，就截断、插入 `</think>`、只续写答案（总长 ≤ 16k）。avg@8，6 个数据集。

| model | base plain 32k | budget | forced acc | forced 长度 | 推理 | 答案 | 被强制比例 | 16k acc | 16k 长度 |
|---|---|---|---|---|---|---|---|---|---|
| 4B | 0.699 / 13,692 | 8k | 0.601 | 8,469 | 6,327 | 2,141 | 0.52 | 0.586 | 10,136 |
| 4B | | 4k | 0.570 | 6,361 | 3,693 | 2,667 | 0.76 | 0.601 | 9,891 |
| 4B | | 2k | 0.528 | 5,498 | 1,990 | 3,508 | 0.92 | 0.592 | 9,908 |
| 8B | 0.706 / 14,103 | 8k | 0.602 | 9,591 | 6,680 | 2,910 | 0.59 | 0.581 | 10,805 |
| 8B | | 4k | 0.589 | 8,078 | 3,804 | 4,273 | 0.81 | 0.584 | 10,690 |
| 8B | | 2k | 0.561 | 7,427 | 2,007 | 5,419 | 0.93 | 0.594 | 10,618 |
| 14B | 0.744 / 12,750 | 8k | 0.664 | 8,158 | 6,358 | 1,799 | 0.52 | 0.653 | 10,058 |
| 14B | | 4k | 0.605 | 6,243 | 3,731 | 2,511 | 0.76 | 0.649 | 9,957 |
| 14B | | 2k | 0.549 | 4,798 | 1,994 | 2,802 | 0.91 | 0.640 | 9,766 |

- budget 越紧，acc 越低（8k → 2k 掉 7–12 pts），被强制的比例从约 0.5 升到约 0.92。
- 被强制后，base 会在 `</think>` 之后继续推理：budget 越紧答案越长，8B 在 2k 时答案 5.4k，比推理本身还长。
- 在 8k 处强制 `</think>` 比直接截到 16k 更好（4B 0.601 vs 0.586，14B 0.664 vs 0.653）。

### 实例分析（详见 `analysis/case_studies.md`）

4B base vs 4B 8k（step 56），8B base vs 8B 4k（step 56）；AMC23 / AIME25 / GPQA 物理各 3 题 × 4 个样本，自由生成 + 强制 `</think>`。

- **缩短的不是解题路径，而是得出答案之后的冗余验证。** 在双方都答对的题上：
  - 答案出现之前的推理：4B +20%，8B +12%。
  - 答案出现之后的推理：4B −23%，8B −40%。
- **被截断后，接着已有进度作答，而不是推翻重来。**
  - 答案区的自我否定：4B 1.60 → 0.75，8B 2.28 → 0.93。
  - 答案长度 −26% 到 −44%，被强制时的 acc 不变。
- **（出乎预期）base 在答案区"失控"。** 被强制后写到长度上限都没给出答案：8B 0.22 → 0.11，4B 0.10 → 0.05。
- **推理中的反思密度和自由生成时的答案长度都不变。** 训练改变的是"何时停"和"如何收尾"，不是"怎么推理"。
- **局限**：
  - 难题（AIME）推理长度不变。
  - budget 太紧时，直接作答意味着补不回缺失的推理。

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

- **模型越大，缩短越少，acc 越稳**（8k，32k eval，相对 base plain）：4B −16.6% / −3.4 pts，8B −10.5% / −1.6 pts，14B −8.1% / −0.7 pts。
- **8B 的训练效果最“干净”**：相对“base + 同一 budget prompt”仍有 6–7% 的缩短，4k/2k 的 acc 与 base 持平；4B 只有 8k 有明显的训练增益，14B 只有 3%。
- **16k 预算下 acc 随训练上升**（8B：+2 到 +3 pts；4B 8k：+2.2 pts），说明在 16k 内能做完的题更多。
- **强制 `</think>` 下长度缩短最多**（4B 2k −26%，8B 2k −23%），但被强制的比例几乎不降（约 0.9）：模型学到的主要是更短的答案，而不是在 budget 内自己结束思考。
- **缩短集中在 GPQA 和 AMC23**（14B −14% 到 −16%），AIME/HMMT 只缩 3–6%，与 budget prompt 本身的效果分布一致。

## 已完成（工程）

- [x] eval variants：一条 32k rollout 同时得出 32k/16k/forced（`adopd_eval_variants`）
- [x] teacher cache 改为 driver 统一写到 head（节点本地磁盘）
- [x] 4B eval 矩阵合并成 10 个 job（原 32 个），结果写入 3 个 4B 训练 run
- [x] 8B 8k/4k/2k、14B 8k 训练与 eval；8B（32k/16k/forced）和 14B（32k）结果按 `eval_avg8_*` 写入各自训练 run
- [x] CRISP 实现（`opd_teacher_source: self`）：
  - teacher 是 actor worker 里 ref 模块持有的 student 拷贝，在"CRISP 简洁指令 + student 的 response"上前向，取 final-norm hidden state，配合上次刷新时收集的完整 lm_head，算精确的全词表 reverse KL。
  - 每 `opd_teacher_refresh_steps`（20）次更新，把 student 的参数分片拷进 teacher。
  - 配置为 `configs/qwen3_{4b,8b,14b}_crisp.yaml`。
  - 状态：单元测试已通过，GPU 冒烟测试待运行。
- [x] eval variant 支持 `prompt: plain`（不加 budget prompt，供 CRISP 用），所有 plain variant 共用同一条 rollout

## 注意事项

- 8B/14B 每个 rollout 副本占 2 张卡（TP 2），需要 `per_device_batch_size: 32` 才能和 4B 一样每步 256 prompt。
- 训练长度 16k、eval 长度 32k：verl 只把 eval response pad 到训练长度、不截断，所以 eval agent 返回截断后的 response，长度指标用完整 rollout 长度。
- wandb 里另有 5 个旧的 avg@4 eval-only run 和 1 个中止的 run，按要求保留未删。
