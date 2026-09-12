# GRPO, OPD, TRD, OPSD, and EOPD training baselines

Portable Qwen3-4B-Base recipes with prepared train/test data, automatic Hugging
Face model preparation, and per-benchmark `pass@8` and `avg@8` evaluation.
Copy this repository, including `data/train/` and `data/eval/`, to the training
platform. Dataset files are regular files with repository-relative paths.
Preparation code and the prepared Parquet files under `data/train/` and
`data/eval/` are tracked under `data/`. Raw downloads and caches are ignored.
No other local checkout is needed.

## Quick start on the training platform

Use Linux, Python 3.10+, and CUDA GPUs. Training and evaluation use one node
with eight GPUs; the OPD teacher shares those GPUs with the student and needs
sufficient additional VRAM. The per-device batch sizes are 128 for OPD and 16
for GRPO, giving 1,024 training responses per step for either method.

```bash
python -m venv .venv
source .venv/bin/activate

# Resolve vLLM/PyTorch before installing the FlashAttention extension.
pip install packaging ninja wheel
pip install 'vllm==0.12.0' 'transformers>=4.51.0,<5'
pip install --no-build-isolation 'flash-attn>=2.6.0'
pip install -r requirements.txt
pip install -e third_party/verl
pip check

# Set wandb_api_key in the selected YAML, or authenticate with:
wandb login

# Choose one training objective:
python train.py configs/qwen3_4b_opd.yaml
python train.py configs/qwen3_4b_grpo.yaml
python train.py configs/qwen3_4b_trd.yaml
python train.py configs/qwen3_4b_opsd.yaml
python train.py configs/qwen3_4b_eopd.yaml
```

The setup above selects vLLM 0.12.0 and its required PyTorch 2.9.0 before
installing FlashAttention. Use a Linux CUDA environment and NVIDIA driver
compatible with those wheels. See the
[vLLM version's CUDA dependencies](https://github.com/vllm-project/vllm/blob/v0.12.0/requirements/cuda.txt)
and [FlashAttention installation instructions](https://github.com/Dao-AILab/flash-attention#installation-and-features).

The Python launcher switches to the repository root and on first
use downloads the pinned `Qwen/Qwen3-4B-Base` checkpoint from HF. It prepares its
ChatML stop/pad settings under `models/Qwen3-4B-Base-chatml/` and reuses it later.
OPD loads `Qwen/Qwen3-30B-A3B-Instruct-2507` as the teacher through HF.
The generated model directory links to the target platform's HF cache; let each
platform prepare its own models. Models and caches are excluded from the bundle.

OPD uses reverse KL in policy-gradient mode, with no task-reward term. Its
`distillation_topk` switch chooses vanilla sampled `k1` (`null`, also the default
when omitted) or a conditional top-k `k1`/PPO expectation (a positive integer).
The packaged OPD and TRD configs set `null`. EOPD sets `16` and adds
an entropy-gated reverse KL auxiliary term. OPSD uses pointwise-clipped reverse
KL with top-k `64`. All distillation recipes use reverse KL; GRPO uses verifier
reward.

```bash
# Print the command without downloading models, starting W&B, or claiming GPUs.
python train.py configs/qwen3_4b_opd.yaml --dry-run
python train.py configs/qwen3_4b_grpo.yaml --dry-run

# Existing local model directories or alternative HF IDs:
MODEL_PATH=./models/my-student TEACHER_MODEL_PATH=Qwen/Qwen3-14B \
  python train.py configs/qwen3_4b_opd.yaml

# Extra native Hydra overrides:
python train.py configs/qwen3_4b_grpo.yaml -- \
  trainer.total_training_steps=100
```

Non-default students must already have suitable tokenizer, padding, and stopping
settings. Edit the YAMLs to change batch size, GPU count, or rollout parameters.

## Code layout

| Location | Responsibility |
| --- | --- |
| `train.py` | Shared CLI, recipe dispatch, model preparation, W&B environment, and process launch. |
| `src/opd.py`, `src/grpo.py`, `src/trd.py`, `src/opsd.py`, `src/eopd.py` | Algorithm parameters, TRD trajectory refinement, OPSD pointwise-clipped reverse KL, and entropy-aware reverse KL. |
| `src/config.py`, `src/runtime.py` | Shared configuration helpers and Hydra/Ray training integration. |
| `eval/` | Benchmark metrics, validation output, grading shared with training rewards, and `eval.sh`. |
| `data/` | Dataset preparation, HF model preparation, and shared prompt/input adapters. |

The launch flow is `train.py` -> the selected `src` algorithm module for configuration,
then `python -m src.runtime` -> native verl training with `eval.trainer` for
validation metrics. Vanilla OPD and GRPO use native verl losses. Top-k OPD,
OPSD, and EOPD install local losses through the existing worker interface. TRD shares the
selected OPD loss. Optimizer updates remain in the unmodified `third_party/verl/`
code.

`data/prepare_qwen3_base.py` prepares model files under `models/`, while
`data/prompt_modes.py` and `data/plaint_runtime.py` handle the shared training
and evaluation inputs. Run commands from the repository root; direct entrypoints
also locate the repository when invoked by absolute path from another directory.

## OPD k1 with optional top-k reverse KL

```yaml
distillation_loss_mode: k1
distillation_topk: null  # Original sampled-token k1 and PPO.
```

Set `distillation_topk: 64` to enable top-k reverse KL. The launcher resolves
this combination to the local runtime loss `reverse_kl_topk`; the recipe stays
`method: opd` (or `trd`) with `distillation_loss_mode: k1`.

At each response position, let `S` be the teacher's top-k vocabulary candidates.
Normalize teacher and student probabilities separately over the same set `S`:

```text
q_teacher(v) = p_teacher(v) / sum_{u in S} p_teacher(u)
q_student(v) = p_student(v) / sum_{u in S} p_student(u)

reverse KL on S = sum_{v in S} q_student(v) * (log q_student(v) - log q_teacher(v))
advantage(v) = stop_gradient(clamp(log q_teacher(v) - log q_student(v), -10, 10))
ratio(v) = q_student_current(v) / q_student_old(v)
```

The loss uses the existing PPO clipping and dual-clip formula for each candidate,
weighted by its fixed `q_student_old(v)`, then sums candidates and averages over
valid response tokens. Without clipping, its gradient equals the conditional
reverse KL gradient. The original FSDP old-logprob forward now also caches the
top-k reference probabilities, which stay fixed across all minibatches/epochs
for that rollout batch. No extra model scoring forward is added.

This is a reverse KL objective over the selected support. Candidates outside
`S` receive no distillation-logit gradient at that position; `k=1` is degenerate
because both conditional distributions contain only one candidate. Top-k does
not change rollout sampling or answer length. Setting `null` restores the exact
native sampled-token path and does not request or cache top-k probabilities.
`distillation_log_prob_min_clamp` remains unused for both k1 variants;
`distillation_loss_max_clamp` controls the symmetric per-candidate signal clamp.

Top-k k1 requires the FSDP/FSDP2 engine workers, matching student/teacher token
IDs, and the regular old-logprob pass (no rollout bypass). The existing
`distillation_use_policy_gradient: true` setting is retained. W&B records
`actor/distillation/topk_reverse_kl`, `topk_pg_clipfrac`, `topk_pg_clipfrac_lower`,
`topk_ppo_kl`, and `topk_teacher_mass` under the same
`actor/distillation/` prefix. The teacher mass reports coverage before
renormalization. The entropy-gated auxiliary term is enabled separately by
`method: eopd`.

## EOPD with reverse KL throughout

```bash
python train.py configs/qwen3_4b_eopd.yaml
```

```yaml
method: eopd
distillation_loss_mode: k1
distillation_topk: 16
distillation_use_policy_gradient: true
eopd_entropy_threshold: 0.8
eopd_aux_loss_coef: 1.0
```

This recipe combines the top-k k1/PPO loss above with a direct reverse KL term
at positions where teacher entropy exceeds the threshold:

```text
H_teacher = -sum_v q_teacher(v) * log q_teacher(v)
L = L_topk_k1_PPO + alpha * I[H_teacher > threshold] * KL(q_student || q_teacher)
```

Both `q` distributions are normalized on the same teacher top-16 support.
Entropy is computed from this conditional teacher distribution, so it
approximates full-vocabulary teacher entropy. The teacher probabilities and
entropy gate are detached. The auxiliary term differentiates directly through
student probabilities, with no PPO ratio or pointwise clipping, and is averaged
over all valid response tokens, including zero contributions from positions
below the threshold. The main k1/PPO term retains its signal clamp and ratio
clipping. Setting `eopd_aux_loss_coef: 0` leaves the top-k k1/PPO objective.

This is the requested **reverse-KL variant** of
[Entropy-Aware On-Policy Distillation of Language Models](https://arxiv.org/html/2603.07079v3).
The paper combines sampled OPD with an entropy-gated forward KL term; this
implementation uses top-k k1/PPO and a reverse KL auxiliary term. It shares
OPD's rollout, teacher scoring, cached old probabilities, and optimizer loop,
without adding a model scoring forward. Training and evaluation settings match
the OPD recipe. The experiment is `qwen3_4b_eopd`, with outputs under
`outputs/qwen3_4b/eopd/<run>/`.

W&B records the top-k OPD metrics plus `actor/eopd/aux_reverse_kl`,
`actor/eopd/teacher_topk_entropy`, and `actor/eopd/high_entropy_fraction`.
The auxiliary loss metric is reported before multiplying by its coefficient.

The EOPD recipe uses an actor microbatch budget of 18432 tokens per GPU. Its
main and auxiliary losses share one top-k selection, whose backward saves
indices instead of retaining the full vocabulary logits. The FSDP engine
detaches returned predictions after constructing the loss so unused probability
graphs do not retain logits across microbatches. These changes preserve the
objective, top-16 support, and 16384-token response budget. The LM head still
produces full-vocabulary logits and one dense gradient, so peak GPU memory must
be checked on the training hardware.

## OPSD reverse KL with pointwise clipping

```bash
python train.py configs/qwen3_4b_opsd.yaml --dry-run
python train.py configs/qwen3_4b_opsd.yaml
```

`method: opsd` keeps the OPD student rollout, external frozen teacher, data,
batch size, evaluation schedule, and optimizer settings. It changes the loss
to reverse KL with the pointwise upper-clipping mechanism in
[Self-Distilled Reasoner, Section 3.2](https://arxiv.org/html/2601.18734v3#S3.SS2).
The KL direction is changed to reverse KL for this local recipe.
The experiment is `qwen3_4b_opsd`, with checkpoints and evaluation outputs
under `outputs/qwen3_4b/opsd/<run>/`.

```yaml
method: opsd
distillation_loss_mode: reverse_kl_topk
distillation_topk: 64
distillation_pointwise_clip: 0.05
distillation_use_policy_gradient: false
distillation_use_task_rewards: false
distillation_loss_max_clamp: null
distillation_log_prob_min_clamp: null
```

For each response position, the local loss computes
`sum_v min(p_student[v] * (logp_student[v] - logp_teacher[v]), 0.05)`
over the teacher's top-k vocabulary entries, then averages over valid response
tokens across the training batch. Teacher probabilities are detached. Gradients
flow through both the student probability weight and student log-probability;
there is no PPO ratio in this
loss. Clipping applies before the vocabulary sum, only from above. Negative
contributions and negative summed losses are preserved. Set
`distillation_pointwise_clip: null` to disable this clipping.

This is an **external-teacher, top-k approximation**, not a full reproduction
of the paper's self-teacher or full-vocabulary experiment. Retained probabilities
keep their full-vocabulary normalization; omitted tail contributions are zero,
and top-k probabilities are not renormalized. Student and teacher must share
token-to-ID mappings. Both scoring temperatures are 1.0. The existing FSDP
`old_log_probs` scoring pass remains in the shared training loop, although the
OPSD loss does not consume it. Evaluation continues to sample the student.

W&B additionally receives `actor/distillation/loss`,
`actor/opsd/unclipped_loss`, `actor/opsd/clip_fraction`,
`actor/opsd/teacher_topk_mass`, and `actor/opsd/student_topk_mass` each training
step. Top-k mass measures how much probability the approximation retains.
The local logit calculation uses chunks and activation checkpointing to limit
extra FP32 workspace; GPU training memory and throughput still need validation
on the target platform.

## TRD trajectories with the OPD loss

`configs/qwen3_4b_trd.yaml` uses the OPD sampled `k1` loss with top-k disabled
(`null`) and `mode: thinking` for teacher chat templating. Its actor micro-batch token
budget remains 40,960 per GPU; the other packaged recipes use 18,432 to reduce
training memory peaks while keeping their full response length and global batch
size. Its experiment
name is `qwen3_4b_trd`, with outputs under
`outputs/qwen3_4b/trd/`. On the same eight GPUs, each training step runs:

1. Student vLLM generates `y_o`, then sleeps.
2. Teacher vLLM reads the question, `y_o`, and the math rewrite instructions in
   `data/refine.py`, generates `y_r`, then sleeps.
3. The existing OPD teacher scores the original student prompt plus `y_r`, then
   sleeps. The rewrite prompt is used only for generation.
4. Student FSDP computes fixed `old_log_probs` on `y_r` (including the conditional
   top-k reference when enabled), then performs its usual
   training forward, backward, and optimizer update. Updated weights are sent
   to the student rollout engine.

The loss is the same selected `k1` policy-gradient loss as OPD, controlled by
`distillation_topk` (`null` in both packaged recipes for vanilla k1).
It includes the `[-10, 10]` distillation-signal clamp, PPO ratio clipping at
`[0.8, 1.2]`, and dual clipping. Training rewards and masks are
recomputed for `y_r`; cached probabilities from generating `y_o` are discarded.
Evaluation still samples and grades the student directly, without refinement.

`max_completion_length: 8192` caps student `y_o` generation. The recipe omits
`val_max_completion_length` and `refine_max_new_tokens`, so validation responses
and teacher `y_r` generation inherit this budget. The student engine's context
is 12,288 tokens (4,096 validation prompt tokens + 8,192 response tokens), while
training questions remain limited to 2,048 tokens.

The rewrite input limit is calculated from `max_prompt_length` plus
`max_completion_length` plus `refine_prompt_token_reserve: 2048`, giving
12,288 tokens (2,048 + 8,192 + 2,048). An explicit `refine_max_prompt_length`
overrides that calculation. The reserve defaults to 1,024 if omitted; it is
an allowance for instructions, chat templating, and retokenization, not a
guaranteed bound on teacher input length. The teacher engine's total context
is 20,480 tokens, the larger of rewrite input plus output (12,288 + 8,192) and
native scoring (2,048 + 8,192 + 1). The runtime
recomputes this requirement after Hydra overrides and increases the teacher
context if needed before creating the engine. It checks the initialized teacher
capacity before initial validation. The rewrite output budget comes from the
configured `refine_max_new_tokens` (or the rollout response length if omitted),
independently of padded tensor width or the teacher scoring service's one-token
output default. Rewrite prompts that exceed their budget raise an error instead
of silently cutting the question or `y_o`.
Increase `refine_max_prompt_length` and restart if a future input exceeds the
limit; it controls teacher input space, not the answer length. For example,
`trd.max_prompt_length=24576` explicitly reserves a larger input budget and
raises teacher context to 32,768 with an 8,192-token output budget.
Use `--dry-run` to check effective values when a remote job reports an old
limit. Teacher generation uses the selected prompt `mode` and
the same temperature, top-p, and top-k as student training rollout.

```bash
python train.py configs/qwen3_4b_trd.yaml --dry-run
python train.py configs/qwen3_4b_trd.yaml
python train.py configs/qwen3_4b_trd.yaml --eval-only
```

## Prompt mode

Set `mode` in either recipe YAML. It applies to both training and evaluation:

| Mode | Model input |
| --- | --- |
| `thinking` | Qwen chat template with `enable_thinking=true`. |
| `non-thinking` | Qwen chat template with `enable_thinking=false`, including the empty thinking block. |
| `plaint` | Direct text encoding without chat-template or role tokens. |

The OPD config uses `mode: plaint`; the original GRPO config uses
`mode: non-thinking`. The `mode` setting replaces `student_enable_thinking` and
`val_enable_thinking`. To use plain text:

```yaml
mode: plaint
```

The plain prompt for math is:

```text
{question}
Please reason step by step, and put your final answer within \boxed{}.
```

The existing standard answer hint is moved to its own line without duplication.
AMO's official question and answer-format instructions are retained before this
hint. `plaint` adds no user/assistant delimiters or `<think>` block. Prompt-length
filtering and rollout use the same direct tokenization, without changing the
prepared Parquet files. The mode is included in the printed launch summary and
the W&B configuration as `data.prompt_mode`.
MBPP+ retains its Python code-block instruction, and GPQA-Diamond retains its
boxed answer-letter instruction, in all three modes.

GRPO also has a separate config for each mode:

| Config | Mode | W&B experiment name |
| --- | --- | --- |
| `configs/qwen3_4b_grpo_mode_thinking.yaml` | `thinking` | `qwen3_4b_grpo_mode_thinking` |
| `configs/qwen3_4b_grpo_mode_no_thinking.yaml` | `non-thinking` | `qwen3_4b_grpo_mode_no_thinking` |
| `configs/qwen3_4b_grpo_mode_plaint.yaml` | `plaint` | `qwen3_4b_grpo_mode_plaint` |

Run any of these with `python train.py <config>`. They inherit the
original GRPO training settings and W&B project, with separate output directories
under `outputs/qwen3_4b/grpo/mode_<mode>/`.

## Prepared data and reproduction

| Dataset | Repository-relative path | Questions |
| --- | --- | ---: |
| DAPO-Math-17k | `data/train/dapo_math_17k.parquet` | 17,916 |
| AIME25 | `data/eval/aime25.parquet` | 30 |
| AIME26 | `data/eval/aime26.parquet` | 30 |
| HMMT February 2026 | `data/eval/hmmt26.parquet` | 33 |
| AMO-Bench-P | `data/eval/amobench.parquet` | 39 |
| MBPP+ v0.2.0 | `data/eval/mbppplus.parquet` | 378 |
| GPQA-Diamond | `data/eval/gpqa_diamond.parquet` | 198 |

The pinned DAPO source has 17,917 rows. One question exactly matches an AIME26
question after case folding and whitespace removal, so it is excluded from
training during data preparation. This filtering does not detect paraphrases.
AMO-Bench-P retains the official number, set, and variable answer types; the 11
description questions from the 50-question full benchmark are excluded.

Data preparation needs only CPU-side dependencies:

```bash
pip install -r requirements-data.txt
python data/prepare_data.py            # Prepare missing files; reuse existing files.
python data/prepare_data.py --rebuild  # Regenerate using pinned sources/cache.
```

`data/prepare_data.py` contains test-data conversion for all four benchmarks
and calls `data/build_dapo_math_17k.py` for training data after the test files
are prepared. The latter still works on its own and removes overlaps against
evaluation files already present. The unified script also calls
`data/prepare_code_science.py`. Use it to prepare all seven
Parquet files. Raw caches live under `data/raw/` and are ignored by Git.
Training and evaluation run offline
with respect to datasets. HF model downloads still need network access.

Source revisions are pinned directly in the preparation scripts. Data manifests
and startup dataset checks are not used. Sources:
[DAPO](https://github.com/thunlp/OPD/tree/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/datasets),
[AIME25](https://huggingface.co/datasets/math-ai/aime25),
[AIME26](https://huggingface.co/datasets/math-ai/aime26),
[HMMT26](https://huggingface.co/datasets/MathArena/hmmt_feb_2026), and
[AMO-Bench](https://huggingface.co/datasets/meituan-longcat/AMO-Bench).
Dataset rights remain with their sources. Local source notices under
`data/notices/` are ignored by Git and are not required by the launchers.

### Code and science data

```bash
# Prepare only the two additional evaluation datasets, without CUDA or EvalPlus.
pip install -r requirements-data.txt
python data/prepare_code_science.py
python data/prepare_code_science.py --rebuild
```

The preparation script pins the official
[MBPP+ release](https://github.com/evalplus/mbppplus_release/releases/tag/v0.2.0)
and [GPQA repository revision](https://github.com/idavidrein/gpqa/tree/56686c06f5e19865c153de0fdb11be3890014df7),
verifies download SHA256 checksums, and stores source metadata in each Parquet.
This MBPP+ artifact contains 378 tasks. GPQA-Diamond contains all 198 questions;
option indices are shuffled with seed 0 and a stable question ID. The source's
repeated incorrect options are preserved. GPQA data is by Irving David Rein,
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

MBPP+ includes the reference implementations and original/expanded test inputs
in the grader's `ground_truth` field, so evaluation needs no dataset download.
They are excluded from generation prompts. GPQA prompts contain only the
question, shuffled choices, and answer instructions. These are evaluation-only
datasets; training data remains DAPO-Math-17k.

## Evaluation

All packaged recipes evaluate before training and every 25 steps, including the last
step. Every evaluation generates 8 answers per question across all six
benchmarks, for 5,664 answers total. Defaults are temperature 0.7, top-p 0.8,
top-k 20, and a 16,384-token completion limit.

- `pass@8`: fraction of questions with at least one correct answer among 8.
- `avg@8`: mean correct-answer fraction across the 8 samples per question.

Values are in `[0, 1]`. Exact observed successes are used, without bootstrap
resampling. AIME and HMMT use boxed-answer Math-Verify grading. AMO uses its
parser protocol, including set comparison and the official variable probes.
Grading runs in separate CPU processes so parser timeouts work under verl.
MBPP+ uses pinned EvalPlus 0.3.1 input deserialization, sanitization, special
oracles, and subprocess execution. A solution passes only if both the base and
expanded tests pass. `eval/mbppplus/pass@1` is the sample-average pass rate,
equal to `avg@8` with the default 8 samples. GPQA-Diamond uses final-letter
exact matching and additionally records `eval/gpqa_diamond/accuracy`, also equal
to `avg@8`. These are sampled scores, not greedy single-generation results.
Thinking blocks and ambiguous answers receive no final-answer credit.

EvalPlus executes generated Python in temporary subprocesses with its default
time/memory limits; code sanitization has a 10-second limit. Use an isolated
Linux evaluation environment since EvalPlus's reliability guard is not a full
security sandbox. A known-correct execution probe fails the run if the execution
environment is broken. The loader also rejects MBPP+/GPQA evaluations that lose
questions to prompt filtering or sample limits; increase `val_max_prompt_length`
if the error reports filtered questions. Packaged recipes use 4,096 tokens for
validation prompts to retain the longest GPQA question, while training prompt
filtering remains at 2,048. The shared rollout engine reserves the larger context.

MBPP+ rewards run in a persistent standalone Python worker, so EvalPlus's child
processes do not reload the Ray/training entrypoint or inherit model memory.
The worker uses one BLAS/OpenMP thread and retains EvalPlus's time/memory limits.
Recipes that validate on MBPP+ check this same reward path before training starts,
including a correct solution and a solution that fails the plus tests. To run
the check separately in the Linux training container:

```bash
python -m eval.reward_async --check
```

To evaluate without training or loading an OPD teacher:

```bash
python train.py configs/qwen3_4b_grpo.yaml --eval-only
MODEL_PATH=./models/my-exported-hf-checkpoint \
  python train.py configs/qwen3_4b_grpo.yaml --eval-only
python train.py configs/qwen3_4b_grpo.yaml --eval-only --dry-run

# Equivalent shortcut using the default GRPO config; forwards extra arguments.
bash eval/eval.sh --dry-run
```

Use an HF-format checkpoint for `MODEL_PATH`. The eval-only path initializes
verl's student workers and still requires the configured CUDA GPUs.

To evaluate an existing native verl checkpoint, use its original recipe and the
same GPU count used when saving the FSDP shards. No HF export is required:

```bash
pip install 'evalplus==0.3.1' 'tree-sitter==0.24.0' 'tree-sitter-python==0.23.6'

# Evaluate just MBPP+ and GPQA-Diamond at step 25. Replace RUN with your run directory.
python train.py configs/qwen3_4b_opd.yaml --eval-only \
  --checkpoint outputs/qwen3_4b/opd/RUN/global_step_25 \
  --benchmarks mbppplus gpqa_diamond

# Inspect the command without loading weights or claiming GPUs.
python train.py configs/qwen3_4b_opd.yaml --eval-only --dry-run \
  --checkpoint outputs/qwen3_4b/opd/RUN/global_step_25 \
  --benchmarks mbppplus gpqa_diamond

# Evaluate every saved checkpoint in a run, in numeric step order.
for step in $(find outputs/qwen3_4b/opd/RUN -maxdepth 1 -type d \
  -name 'global_step_*' | sed 's/.*global_step_//' | sort -n); do
  python train.py configs/qwen3_4b_opd.yaml --eval-only \
    --checkpoint "outputs/qwen3_4b/opd/RUN/global_step_${step}" \
    --benchmarks mbppplus gpqa_diamond || break
done
```

`--checkpoint` restores actor weights and the checkpoint step, disables the
teacher and training updates, and leaves the source checkpoint intact. Results
go into a new evaluation run directory, with `eval_metrics/25.json` and
`validation/25.jsonl` for step 25. Omit `--benchmarks` to evaluate all configured
benchmarks. The same selection flag can restrict periodic validation during
training. Newly started training runs automatically include both datasets at
steps 25, 50, 75, etc.; a process already running uses its original configuration.

Metrics are written to `outputs/qwen3_4b/<method>/<run>/eval_metrics/<step>.json`.
Sample outputs are saved under the run's `validation/` directory. Checkpoints,
the effective recipe, launch command, and Hydra output stay under the same
relative `outputs/` tree. Training and evaluation share the selected prompt mode
and completion limit; the launcher rejects mismatched completion lengths.
