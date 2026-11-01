# TODO — ADOPD v2 efficient reasoning

更新时间：2026-11-01 16:50 UTC。两台机器，各为 2 × 8 H100（Ray head + workera），所有产物都在各自容器内：
1101（本机，原有实验）和 1102（11/01 起，环境与 1101 相同）。`todo.md` 由 1101 统一维护，1102 的结果发给 1101 后合并进来。

## 进行中

**1101**（`tmp/chain_crisp_8b_only.sh`，替换了 `tmp/chain_crisp_rest.sh`）：

- [x] CRISP 4B 训练（09:12–12:29 UTC）和 step 20/40/56 eval
- [x] 14B ADOPD 8k/4k/2k 在 step 20/40 的 16k/forced eval（6 个，13:50–15:54 UTC），已写入 wandb
- [ ] CRISP 8B：训练（15:55 UTC 开始），然后 step 20/40/56 eval（不加 budget prompt；一条 32k rollout 得出 32k、16k 截断和在 8k/4k/2k 处强制 `</think>`），然后删除 FSDP 分片。预计 21:30 UTC 左右结束，之后 1101 空闲，等待新任务。

**1102**（`tmp/chain_1102.sh`，用户决定把 CRISP 14B 放到 1102）：

- [ ] CRISP 14B：训练（wandb run `qwen3_14b_crisp`，从 1102 发出），step 20/40/56 eval，删除 FSDP 分片
- [ ] 14B base 在 4k/2k budget prompt 下的 32k eval（补 14B 矩阵 step 0 的两格空白；结果先存在 1102 本地）

`tmp/keep_hf.py` 在后台把每个 CRISP run 的 step 20/40 HF 权重硬链接到 `hf_keep/stepN` 并拷到 head（保险，见注意事项）。

## 待办

- [ ] CRISP 8B 的 eval 结果写入 wandb（`tmp/log_eval_crisp_to_wandb.py`），并补进本文；4B 已完成
- [ ] 合并 1102 的结果：CRISP 14B、14B base 在 4k/2k prompt 下的 32k eval
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
| 14B 4k | `ca52e83d` `qwen3_14b_adopd_v2_sample_g1_b4096_bs256` |
| 14B 2k | `cb163dc9` `qwen3_14b_adopd_v2_sample_g1_b2048_bs256` |
| CRISP 4B | `ff568aef` `qwen3_4b_crisp` |

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
| 14B 4k 256x56 | 1–5 | 0.719 | 6129 | 3920 | 2208 | 0.84 | 0.075 | 0.096 | 240 |
| 14B 4k 256x56 | 52–56 | 0.855 | 4704 | 3586 | 1117 | 0.64 | 0.050 | 0.088 | 202 |
| 14B 2k 256x56 | 1–5 | 0.656 | 4889 | 2032 | 2856 | 0.97 | 0.086 | 0.099 | 229 |
| 14B 2k 256x56 | 52–56 | 0.782 | 3112 | 1990 | 1121 | 0.90 | 0.053 | 0.097 | 201 |
| 8B 8k 128x112（消融） | 1–5 | 0.709 | 9955 | 7130 | 2823 | 0.62 | 0.091 | 0.087 | 168 |
| 8B 8k 128x112（消融） | 108–112 | 0.833 | 4882 | 3758 | 1122 | 0.16 | 0.070 | 0.046 | 142 |
| 8B 4k 128x112（消融） | 1–5 | 0.692 | 8063 | 4006 | 4056 | 0.91 | 0.096 | 0.098 | 157 |
| 8B 4k 128x112（消融） | 108–112 | 0.864 | 4600 | 3442 | 1156 | 0.61 | 0.065 | 0.084 | 126 |

总耗时：
- 4B 每个 run 5.1–5.8 h（中途 eval 按旧的 16k avg@4 口径跑）。
- 8B 每个 run 4.3–4.6 h，含 3 次 32k/16k/forced eval（每次约 24 分钟）。
- 14B 8k 6.3 h，含 step 0（plain + budget）和 3 次 32k eval；14B 4k 5.0 h、2k 4.7 h，含 3 次 32k eval。
- 14B 的 train acc 在第 25、29、45 步明显下跌（4k 和 2k 同步），是这几批题较难（同一 seed、同样的数据顺序），不是训练退化。

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

### Qwen3-14B eval 矩阵（acc / 平均长度）

- 32k：训练中的 eval。
- 16k / forced：合并 eval（`tmp/eval_merged14b/`）。之前写“14B 只保留最后一个 checkpoint，所以没有 step 20/40”是错的：verl 只删除 head 上的旧 checkpoint，worker 上 step 20/40 的 HF 权重都在。step 20/40 已补齐。
- step 0 = base model 加对应 run 的 budget prompt；4k/2k prompt 下 base 的 32k 还没跑，由 1102 补跑。

base（step 0，plain，32k）：acc 0.744，长度 12,750

| run | eval | step 0 | step 20 | step 40 | step 56 | Δlen vs base 同 prompt | Δlen vs base plain(32k) |
|---|---|---|---|---|---|---|---|
| 14B 8k | 32k | 0.743 / 12.1k | 0.743 / 12.3k | 0.747 / 12.0k | 0.738 / 11.7k | -3.0% | -8.1% |
| 14B 8k | 16k | 0.653 / 10.1k | 0.650 / 10.0k | 0.645 / 9.9k | 0.655 / 9.6k | -4.2% | — |
| 14B 8k | forced | 0.664 / 8.2k | 0.659 / 8.2k | 0.655 / 8.1k | 0.650 / 7.7k | -5.0% | — |
| 14B 4k | 32k | — | 0.749 / 11.8k | 0.743 / 11.8k | 0.743 / 11.7k | — | -8.1% |
| 14B 4k | 16k | 0.649 / 10.0k | 0.645 / 9.9k | 0.639 / 9.9k | 0.649 / 9.7k | -2.3% | — |
| 14B 4k | forced | 0.605 / 6.2k | 0.605 / 5.9k | 0.584 / 5.6k | 0.574 / 5.4k | -14.1% | — |
| 14B 2k | 32k | — | 0.738 / 11.7k | 0.746 / 11.6k | 0.746 / 11.4k | — | -10.8% |
| 14B 2k | 16k | 0.640 / 9.8k | 0.648 / 9.7k | 0.652 / 9.7k | 0.644 / 9.6k | -1.6% | — |
| 14B 2k | forced | 0.549 / 4.8k | 0.556 / 4.8k | 0.514 / 4.0k | 0.514 / 3.6k | -25.1% | — |

forced 的推理 / 答案长度（step 0 → 56）：

| run | 被强制比例 | 推理 | 答案 | acc |
|---|---|---|---|---|
| 14B 8k | 0.52 → 0.49 | 6,358 → 6,124 | 1,799 → 1,622 | 0.664 → 0.650 |
| 14B 4k | 0.76 → 0.74 | 3,731 → 3,694 | 2,511 → 1,666 | 0.605 → 0.574 |
| 14B 2k | 0.91 → 0.90 | 1,994 → 1,986 | 2,802 → 1,607 | 0.549 → 0.514 |

- 32k 和 16k 下 acc 与 base 持平（±0.7 pts），长度在 32k 下减 8–11%（相对 base plain），16k 下减 1.6–4.2%（相对 base 同 prompt）。16k 下整个训练过程中 acc 都在 base ±1.2 pts 以内。
- forced 下 acc 的下降集中在 step 20 → 40（4k −2.1 pts，2k −4.2 pts），step 40 → 56 基本不再掉，长度继续缩短；step 20 时三个 budget 都和 base 持平。
- forced 下长度减得最多（2k −25%），几乎全部来自 `</think>` 之后的答案变短（2,802 → 1,607），被强制的比例基本不变；acc 跟着降 1.4–3.5 pts。推测 base 在 `</think>` 之后的继续推理对 acc 有帮助，训练把它砍掉了。8B 在 8k/4k 的 forced 下 acc 反而上升（+1.9 / +0.7 pts），只有 2k 下降（−1.4 pts）。

各数据集，32k step 56 相对 base plain（Δacc / Δlen）：

| 数据集 | 14B 8k | 14B 4k | 14B 2k |
|---|---|---|---|
| aime25 | -0.029 / -5.6% | -0.037 / -3.3% | -0.004 / -6.0% |
| aime26 | -0.025 / -4.5% | +0.021 / -6.1% | -0.013 / -5.9% |
| hmmt26 | -0.023 / -5.9% | -0.011 / -5.9% | +0.011 / -10.0% |
| amc23 | +0.003 / -13.8% | -0.009 / -15.7% | +0.000 / -17.5% |
| gpqa_chemistry_general | +0.037 / -16.4% | +0.037 / -16.3% | +0.012 / -20.1% |
| gpqa_physics_general | +0.020 / -15.5% | +0.013 / -14.9% | +0.013 / -19.6% |

#### 14B 8k（32k eval）细节

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

### CRISP 基线（arXiv 2603.05433）

- 方法：自蒸馏 OPD。teacher 是 student 自身的拷贝，在 student 自己的 rollout 前加上 CRISP 的简洁指令（"Solve the following math problem concisely and correctly. Be direct — …"）后打分；loss 是全词表 reverse KL。teacher 每 20 次更新刷新为当前 student（第 21、41 次更新前）。
- 训练设置与 ADOPD 相同（256 × 56，16k，lr 1e-6，warmup 25）；student 不加 budget prompt，也不强制 `</think>`。
- eval：不加 budget prompt，一条 32k rollout 得出 32k、16k 截断和在 8k/4k/2k 处强制 `</think>`（只续写答案）。从 4B 起保存逐条生成，其它截断可以在 CPU 上重算。

训练指标（5 步平均）：

| run | steps | train acc | resp len | self-distill KL | s/step |
|---|---|---|---|---|---|
| CRISP 4B | 1–5 | 0.730 | 10984 | 0.0094 | 275 |
| CRISP 4B | 18–22 | 0.783 | 9425 | 0.0070 | 216 |
| CRISP 4B | 38–42 | 0.911 | 4571 | 0.0112 | 154 |
| CRISP 4B | 52–56 | 0.864 | 3185 | 0.0066 | 132 |

eval（acc / 平均长度）：

| run | eval | step 20 | step 40 | step 56 | step 56 被强制比例 | Δacc / Δlen step 56 vs base plain(32k) |
|---|---|---|---|---|---|---|
| CRISP 4B | 32k | 0.682 / 12.2k | 0.641 / 8.7k | 0.597 / 5.8k | — | -0.102 / -57.5% |
| CRISP 4B | 16k | 0.586 / 10.2k | 0.617 / 8.1k | 0.596 / 5.8k | — | — |
| CRISP 4B | forced_b8192 | 0.605 / 9.1k | 0.615 / 7.2k | 0.582 / 5.2k | 0.25 | — |
| CRISP 4B | forced_b4096 | 0.598 / 8.1k | 0.591 / 6.0k | 0.549 / 4.1k | 0.51 | — |
| CRISP 4B | forced_b2048 | 0.576 / 8.0k | 0.577 / 5.6k | 0.498 / 3.5k | 0.74 | — |

CRISP 4B 按数据集（32k，step 56 vs base plain）：

| 数据集 | base plain | CRISP step 56 | Δacc | Δlen |
|---|---|---|---|---|
| aime25 | 0.646 / 17,670 | 0.463 / 8,301 | -0.183 | -53.0% |
| aime26 | 0.683 / 15,926 | 0.567 / 7,133 | -0.117 | -55.2% |
| hmmt26 | 0.413 / 19,280 | 0.314 / 7,934 | -0.098 | -58.8% |
| amc23 | 0.969 / 8,619 | 0.919 / 3,181 | -0.050 | -63.1% |
| gpqa_chemistry_general | 0.569 / 10,299 | 0.562 / 3,938 | -0.006 | -61.8% |
| gpqa_physics_general | 0.875 / 8,433 | 0.704 / 3,678 | -0.171 | -56.4% |

- CRISP 压缩得远比 ADOPD 狠：4B 32k 下 −57.5% 长度、−10.2 pts acc（ADOPD 4B 为 −12% 到 −17%、−2.5 到 −3.4 pts）。缩短主要发生在第一次 teacher 刷新之后（step 20 → 40：训练 rollout 9.4k → 4.6k）。
- 到 step 56 几乎所有回答都在 16k 内结束（16k acc 0.596 ≈ 32k 0.597）。
- 和 ADOPD 不同，CRISP 学会了在 budget 内自己结束思考：8k 处被强制的比例 0.53 → 0.25（step 20 → 56）。ADOPD 下这个比例基本不变，缩短的是答案。
- acc 损失集中在难题和 GPQA 物理（AIME25 −18 pts、GPQA 物理 −17 pts）；GPQA 化学几乎不掉。
- forced 列不能直接和上面 base forced 表比：那里 base 用的是 budget prompt，这里是 plain prompt（被强制后答案更长）。

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
| 14B | 0.744 / 12,750 | 4k | 0.743 | -0.001 | -8.1% | — |
| 14B | 0.744 / 12,750 | 2k | 0.746 | +0.002 | -10.8% | — |

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

- **模型越大，缩短越少，acc 越稳**（8k，32k eval，相对 base plain）：4B −16.6% / −3.4 pts，8B −10.5% / −1.6 pts，14B −8.1% / −0.7 pts。14B 的 2k 缩得最多（−10.8%），acc 与 base 持平（+0.2 pts）。
- **14B 在 forced 下用 acc 换长度**：答案部分缩短 10–43%，acc 降 1.4–3.5 pts，budget 越紧降得越多（见 14B 矩阵）。
- **8B 的训练效果最“干净”**：相对“base + 同一 budget prompt”仍有 6–7% 的缩短，4k/2k 的 acc 与 base 持平；4B 只有 8k 有明显的训练增益，14B 只有 3%。
- **16k 预算下 acc 随训练上升**（8B：+2 到 +3 pts；4B 8k：+2.2 pts），说明在 16k 内能做完的题更多。
- **强制 `</think>` 下长度缩短最多**（4B 2k −26%，8B 2k −23%，14B 2k −25%），但被强制的比例几乎不降（约 0.9）：模型学到的主要是更短的答案，而不是在 budget 内自己结束思考。
- **缩短集中在 GPQA 和 AMC23**（14B −14% 到 −16%），AIME/HMMT 只缩 3–6%，与 budget prompt 本身的效果分布一致。
- **CRISP（4B）压缩更狠、代价更大**：−57.5% 长度、−10.2 pts acc；ADOPD 4B 在 −12% 到 −17% 长度下只掉 2.5–3.4 pts。CRISP 会在 budget 内自己结束思考，ADOPD 缩短的主要是答案。

## 已完成（工程）

- [x] eval variants：一条 32k rollout 同时得出 32k/16k/forced（`adopd_eval_variants`）
- [x] teacher cache 改为 driver 统一写到 head（节点本地磁盘）
- [x] 4B eval 矩阵合并成 10 个 job（原 32 个），结果写入 3 个 4B 训练 run
- [x] 8B 8k/4k/2k、14B 8k/4k/2k 训练与 eval；8B（32k/16k/forced）和 14B（32k；16k/forced 在 step 0/56）结果按 `eval_avg8_*` 写入各自训练 run
- [x] CRISP 实现（`opd_teacher_source: self`）：
  - teacher 是 actor worker 里 ref 模块持有的 student 拷贝，在"CRISP 简洁指令 + student 的 response"上前向，取 final-norm hidden state，配合上次刷新时收集的完整 lm_head，算精确的全词表 reverse KL。
  - 每 `opd_teacher_refresh_steps`（20）次更新，把 student 的参数分片拷进 teacher。
  - 配置为 `configs/qwen3_{4b,8b,14b}_crisp.yaml`。
  - 状态：GPU 上跑通（4B 冒烟 2 次更新 + 4B 正式 56 步）。修了两个只在 GPU 上暴露的问题：ref 模型缺少 `log_prob_micro_batch_size_per_gpu`（verl 配置校验），以及更新后清理 teacher 特征文件时 self-teacher 没有文件（`KeyError: 'path'`）。
  - verl 在 `use_kl_loss` 下每步额外做一次 ref log-prob 前向（4B 约 8 秒/步，KL 系数为 0，不影响 loss）。
- [x] eval variant 支持 `prompt: plain`（不加 budget prompt，供 CRISP 用），所有 plain variant 共用同一条 rollout
- [x] 16k 不需要 GPU：unforced variant 只是把同一条 rollout 截断后在 CPU 上打分；GPU 只用于 rollout 本身和 forced 的答案续写

## 注意事项

- 8B/14B 每个 rollout 副本占 2 张卡（TP 2），需要 `per_device_batch_size: 32` 才能和 4B 一样每步 256 prompt。
- 训练长度 16k、eval 长度 32k：verl 只把 eval response pad 到训练长度、不截断，所以 eval agent 返回截断后的 response，长度指标用完整 rollout 长度。
- wandb 里另有 5 个旧的 avg@4 eval-only run 和 1 个中止的 run，按要求保留未删。
- verl 的 `max_checkpoints_to_keep` 只删除 head 上的旧 checkpoint；worker（FSDP rank 0、HF 导出所在节点）保留每个 `global_step_N`，直到分片被清理。
- 停一个 run 时要连 `python3 -m src.train` 一起停：只停 `run_baseline.py` 的话驱动进程继续运行，Ray placement group 一直占着 GPU，后面等 16 张卡空闲的脚本会卡住。
