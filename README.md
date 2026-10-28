# GRPO and OPD training baselines

Portable Qwen3-4B-Base recipes with prepared train/test data, automatic Hugging
Face model preparation, and per-benchmark `pass@k` and `avg@k` evaluation.
Copy this repository, including `data/train/` and `data/eval/`, to the training
platform. The launcher requires absolute paths for train/evaluation files.
The main GRPO and OPD configs read the same train/evaluation files under
`/scratch/l/luli/data/lightning_opd/`.
Other recipe YAMLs have not been migrated; set their data paths to absolute paths
before using them.
Only the prepared Parquet files under `data/train/` and `data/eval/` are tracked
under `data/`. No other local checkout is needed.

## Qwen3-4B V2 answer-only distillation (ADOPD)

`configs/qwen3_4b_adopd_v2.yaml` implements the V2 reference-teacher method in
`scripts/suhail_method.pdf` with selectable **sampled-token or full-vocabulary
reverse KL on answer positions**, using a frozen
copy of the initial local `Qwen3-4B`. Teacher reference traces are sampled once
offline and reused throughout training and checkpoint recovery. Both teacher
and student have a 16,384-token response limit. The student switches to the
answer after a natural `</think>`, or receives a forced `</think>` after 8,192
response tokens. The injected token counts toward the response limit and has
zero policy loss. The answer uses the remaining response budget.

Prepare the shared teacher cache once, then launch the training jobs.
Cache preparation is resumable and skips cached prompts. It uses every GPU
specified by `h100_nnodes` and `n_gpus_per_node`: Ray schedules local teacher
replicas of `teacher_tp_size` GPUs on every node and distributes distinct
prompt batches across them. The cache is fingerprinted by the frozen weights,
tokenizer, generation settings, and seed. Preparation runs in a separate process that
exits before training starts, releasing its vLLM workers and CUDA state.
Run one cache-preparation job first, then start the training jobs with the
completed cache. Cache preparation has no global lock.

```bash
# First: prepare the shared cache once, using all configured nodes/GPUs.
bash scripts/sbatch/qwen3_4b_adopd_v2.sbatch --prepare-teacher-cache
# After it completes: train with one or eight rollouts/prompt.
bash scripts/sbatch/qwen3_4b_adopd_v2.sbatch
bash scripts/sbatch/qwen3_4b_adopd_v2.sbatch -- adopd_use_group_rollouts=true
# The same automatic workflow from an existing GPU allocation:
python scripts/run_baseline.py configs/qwen3_4b_adopd_v2.yaml --auto-prepare-teacher-cache
```

The `bash` submission entry reads the YAML and passes its node/GPU counts to
`sbatch`. Set `h100_nnodes: 4` and `n_gpus_per_node: 4` to request and use 16 GPUs
for cache preparation and training. The same script supports other positive
node counts; `teacher_tp_size` must divide the GPUs per node. Direct `sbatch`
submission must request matching resources explicitly; mismatches fail before
generation instead of leaving allocated nodes unused.

Student training and evaluation prepend the following instruction to the user
question before applying the thinking chat template:

```text
<budget:8192>
Use at most 8192 tokens for reasoning. Finish early when possible.
```

Both occurrences of 8192 follow `adopd_reasoning_budget`. Prompt-length filtering
includes this instruction. The same budget is enforced by the two-phase rollout
and the injected `</think>` token. The V2 teacher always uses the original
question without this instruction, including cache lookup and answer scoring;
previously generated teacher traces remain reusable.

Select the answer loss with one hyperparameter:

```yaml
opd_loss_token_mode: sample # sample | full_vocab
opd_distribution_chunk_size: 32
```

`sample` keeps the sampled-token k1 estimator and its existing logprob floor
and cost clipping. `full_vocab` computes exact reverse KL over the complete
vocabulary at each sampled answer prefix, using FP32 log-softmax and reductions
without those floor/clipping approximations. Both modes supervise only answer
positions. In full-vocabulary mode, teacher answer hidden states are captured
during the existing scoring pass and reconstructed into logits in chunks of
`opd_distribution_chunk_size` answer positions. Temporary answer features are
removed after the actor update. The frozen teacher-trace cache is shared across
loss modes; changing this hyperparameter does not regenerate reference traces.

The native `distillation_loss_mode: k1` registration remains fixed; the ADOPD
loss callback implements the choice made by `opd_loss_token_mode`. To override
the choice for a run:

```bash
bash scripts/sbatch/qwen3_4b_adopd_v2.sbatch -- opd_loss_token_mode=full_vocab
# It can also be combined with group rollouts:
bash scripts/sbatch/qwen3_4b_adopd_v2.sbatch -- opd_loss_token_mode=full_vocab adopd_use_group_rollouts=true
```

Run-directory names, experiment names, and logged ADOPD settings include the
chosen loss mode.

The fixed YAML switch is `adopd_use_group_rollouts: false`. Without groups,
each prompt has one rollout and the objective is answer KL plus
`adopd_length_penalty * trace_length / 8192`. Its length gradient uses
single-rollout REINFORCE: detach that length cost and multiply it by the SUM
of policy-chosen trace-token log probabilities, including a natural halt and
excluding an injected halt or padding. No group advantage or baseline is
computed. The sum is essential: averaging trace log probabilities would divide
out the length signal. The actor uses `cost + (pg - pg.detach())`, so the
reported loss retains the value `D_g + cost` while backpropagating the length
policy gradient. Setting `adopd_length_penalty: 0` recovers answer KL alone.

With groups, the
detached advantage uses `D_g + 0.1 * trace_length / 8192`, normalized by the
mean and sample standard deviation of all eight rollouts for that prompt.
Only policy-chosen trace tokens receive its REINFORCE term; the additional
single-rollout length term is not applied in group mode. Both paths average
KL over answer tokens within each trajectory, then average trajectories. No
task-verifier reward enters the objective. The answer loss and the group cost
both use the selected KL mode. Full-vocabulary group costs are evaluated during
the old-policy forward before computing the group advantage.
Training logs separately report `answer_kl`, `length_cost`,
`regularized_distortion` (the Eq. (5) cost), and `trace_reinforce` under
`train/adopd/`. In single-rollout mode, the last metric is the value-preserving
length surrogate; in group mode, it is the Eq. (6) trace term.

`D_g` uses student-generated answer prefixes, with the teacher conditioned on
its cached trace and the **student's** answer prefix. In `sample` mode,
both log probabilities are floored at -10 before subtraction and the signed
difference is clipped to [-10, 10]. Sampled costs can still be negative. The
answer surrogate is `r * clipped_cost.detach()`, where
`r = exp(log p_student - log p_old_student)` uses unfloored student log
probabilities. The group advantage uses the same clipped answer cost.
Setting both distillation clamp options to null restores the unclipped k1
estimator with its unit control variate. These sampled-token clamps do not
apply in `full_vocab` mode. Each rollout batch receives
one optimizer update; multiple PPO epochs or split optimizer minibatches are
rejected. Invalid empty-answer trajectories are masked, and a completely
invalid batch skips the optimizer and scheduler.

Both modes generate 16 trajectories per GPU per update: with eight GPUs this
is 128 prompts times one response, or 16 prompts times eight. With 16 GPUs it
is 256 prompts times one response, or 32 prompts times eight. The optimizer
batch automatically follows the full rollout batch when `ppo_mini_batch_size`
is null. Each uses one native training epoch over English
DAPO Math (with the existing dataloader's incomplete-final-batch dropping), so
group training has approximately eight times as many updates. Training uses
`/scratch/l/luli/data/lightning_opd/train/dapo_math_17k_en.parquet`; validation
uses only AIME25, AIME26, and AMC23 from the adjacent `eval/` directory.
Validation retains the 8k reasoning cutoff and 16k response cap, grades only
the answer after `</think>`, and reports `avg@4`, `pass@4`, response/reasoning/
answer lengths, and the forced-halt rate. W&B and run directories distinguish
the group settings. The default sbatch workflow uses one allocation for both
stages, sharing its 12-hour time limit. A failed cache-preparation stage stops
the job before training. On resubmission, completed cache entries are reused;
use `--resume` to also recover a saved training checkpoint.

```bash
# These dry runs require no GPUs and do not create a training run.
python scripts/run_baseline.py configs/qwen3_4b_adopd_v2.yaml --dry-run
python scripts/run_baseline.py configs/qwen3_4b_adopd_v2.yaml --dry-run -- adopd_use_group_rollouts=true
python scripts/run_baseline.py configs/qwen3_4b_adopd_v2.yaml --prepare-teacher-cache --dry-run

# MODEL_PATH may point to an exported student checkpoint. Evaluation needs no teacher cache.
MODEL_PATH=/absolute/path/to/student/hf_model \
  python scripts/run_baseline.py configs/qwen3_4b_adopd_v2.yaml --eval-only
# Recovery preserves the frozen teacher and reuses the same cached references.
python scripts/run_baseline.py configs/qwen3_4b_adopd_v2.yaml --resume /absolute/path/to/run
```

## Quick start on the training platform

On this cluster, use the existing `verl` environment (`conda activate verl`).

Use Linux, Python 3.10+, and CUDA GPUs. Training and evaluation use one node
with eight GPUs; the OPD teacher shares those GPUs with the student and needs
sufficient additional VRAM. The current per-device prompt batch sizes are 32
for OPD and 4 for GRPO. Across eight GPUs, OPD generates 256 prompts × 1
response per update; GRPO generates 32 prompts × 8 responses. Both methods
generate 256 trajectories per update.

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

# Set WANDB_API_KEY in the Git-ignored .env file (see .env.example).
# SLURM training records W&B data offline; upload manually after the job finishes.

# Choose one training objective:
python scripts/run_baseline.py configs/qwen3_4b_base_opd.yaml
python scripts/run_baseline.py configs/qwen3_4b_base_grpo.yaml
```

The setup above selects vLLM 0.12.0 and its required PyTorch 2.9.0 before
installing FlashAttention. Use a Linux CUDA environment and NVIDIA driver
compatible with those wheels. See the
[vLLM version's CUDA dependencies](https://github.com/vllm-project/vllm/blob/v0.12.0/requirements/cuda.txt)
and [FlashAttention installation instructions](https://github.com/Dao-AILab/flash-attention#installation-and-features).

The Python launcher switches to the repository root and prepares the pinned
`Qwen/Qwen3-4B-Base` or `Qwen/Qwen3-1.7B-Base` checkpoint from HF. The OPD recipe
uses pre-downloaded local snapshots (`local_models_only: true`) and makes no HF
requests during training. It prepares ChatML stop/pad settings under
`models/<source-name>-chatml/` and reuses the prepared model later.
OPD loads the cached `Qwen/Qwen3-8B` teacher snapshot.
The generated model directory links to the target platform's HF cache; let each
platform prepare its own models. Models and caches are excluded from the bundle.

The packaged OPD objective remains native verl sampled-token `k1` in policy
gradient mode, with no task-reward term. GRPO uses verifier reward.

All launchers use FP32 sharded actor parameters and AdamW states, BF16
forward/backward compute, and FP32 gradient reduction. Keeping optimizer
parameters in FP32 lets 1e-6-scale updates accumulate across steps, matching
Lightning-OPD's FP32 master weights. Actor losses use `seq-mean-token-mean`:
average selected tokens within each response, then average responses over the
global optimizer batch. Distillation refreshes the global normalization counts
before its first microbatch as well as subsequent ones.

Sampled OPD matches Lightning-OPD's stability settings: floor both student and
teacher log probabilities at -10, subtract them, and clip the signed cost to
[-10, 10]. The policy ratio uses the original student log probabilities, so
the floor does not suppress policy gradients for low-probability tokens.
These defaults also apply to ADOPD's answer loss. The vocabulary distribution
losses and diagnostic/selection scores do not use the logprob floor.

```bash
# Print the command without downloading models, starting W&B, or claiming GPUs.
python scripts/run_baseline.py configs/qwen3_4b_base_opd.yaml --dry-run
python scripts/run_baseline.py configs/qwen3_4b_base_grpo.yaml --dry-run

# Existing local model directories or alternative HF IDs:
MODEL_PATH=./models/my-student TEACHER_MODEL_PATH=Qwen/Qwen3-14B \
  python scripts/run_baseline.py configs/qwen3_4b_base_opd.yaml

# Extra native Hydra overrides:
python scripts/run_baseline.py configs/qwen3_4b_base_grpo.yaml -- \
  trainer.total_training_steps=100
```

Students other than these two Qwen3 Base models must already have suitable
tokenizer, padding, and stopping settings. Edit the YAMLs to change batch size,
GPU count, or rollout parameters.

## Offline W&B on SLURM

The launcher reads the repository-root `.env` automatically. Use:

```dotenv
WANDB_API_KEY=<your-api-key>
WANDB_ENTITY=efficiency-reasoning-with-opd
WANDB_PROJECT=efficient-reasoning
WANDB_MODE=offline
```

Exported variables take precedence over `.env`. The OPD YAML sets
`wandb_mode: offline` and `wandb_entity: efficiency-reasoning-with-opd`; the launcher enforces
offline training even when `WANDB_MODE=online` was inherited. API keys are not
written into the saved training recipe or launch command. There is no automatic
upload when training exits.

After the SLURM job finishes, return to a login node with internet access and run:

```bash
python scripts/sync_wandb.py --dry-run
python scripts/sync_wandb.py
# Or select one timestamped training run:
python scripts/sync_wandb.py /scratch/l/luli/results/onestep/<run>
```

The sync script loads `.env`, uploads through `wandb sync`, and skips runs that
have already been synced or lack the launcher's `training_finished.json` marker.
The repository-root `.env` must exist and define nonempty `WANDB_ENTITY` and
`WANDB_PROJECT` values, otherwise syncing (including `--dry-run`) reports an error.
Sync destinations come from `.env` even when exported variables or offline logs
specify another destination. Use `--entity` or `--project` to explicitly override
these settings.
The marker records the exit code when the training process returns, including
failed training runs. Actual uploads are refused inside a SLURM job;
`--dry-run` only prints the commands and never connects to W&B.

## Prompt mode

Set `mode` in either recipe YAML. It applies to both training and evaluation:

| Mode | Model input |
| --- | --- |
| `thinking` | Qwen chat template with `enable_thinking=true`. |
| `non-thinking` | Qwen chat template with `enable_thinking=false`, including the empty thinking block. |
| `plaint` | Direct text encoding without chat-template or role tokens. |

The main Base OPD and GRPO configs use `mode: plaint`; the post-trained OPD
variants use `mode: non-thinking`. The `mode` setting replaces `student_enable_thinking` and
`val_enable_thinking`. To use plain text:

```yaml
mode: plaint
```

The plain prompt is:

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

Base-model recipes and their launchers include `_base` in their filenames.
Their OPD W&B names also include `base` and `teacher_8b`; post-trained
non-thinking runs include the student, mode, and `teacher_4b_instruct_2507`.
The launcher accepts old Base config paths for compatibility with queued jobs.

GRPO also has a separate config for each mode:

| Config | Mode | W&B experiment name |
| --- | --- | --- |
| `configs/qwen3_4b_base_grpo_mode_thinking.yaml` | `thinking` | `qwen3_4b_base_grpo_mode_thinking` |
| `configs/qwen3_4b_base_grpo_mode_no_thinking.yaml` | `non-thinking` | `qwen3_4b_base_grpo_mode_no_thinking` |
| `configs/qwen3_4b_base_grpo_mode_plaint.yaml` | `plaint` | `qwen3_4b_base_grpo_mode_plaint` |

Run any of these with `python scripts/run_baseline.py <config>`. They inherit the
original GRPO training settings and W&B project, with separate output directories
under `outputs/qwen3_4b/grpo/mode_<mode>/`.

## Thinking student with a concise external teacher

Three efficient-reasoning recipes use `Qwen/Qwen3-1.7B` in thinking mode with
a fixed `Qwen/Qwen3-4B-Instruct-2507` teacher. They adapt the teacher-only concise
conditioning idea from [CRISP](https://arxiv.org/abs/2603.05433). Teacher weights
stay fixed throughout training. The teacher prompt preserves necessary reasoning
on difficult questions and the dataset's existing boxed-answer instructions.
Student and evaluation prompts keep their original question/instructions.
The [teacher model](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) supports
non-thinking mode; its own chat template is used when constructing teacher inputs.

| OPD loss token mode | YAML | sbatch |
| --- | --- | --- |
| `sample` | [config](configs/qwen3_1_7b_thinking_opd_efficient_reasoning_sample.yaml) | [script](scripts/sbatch/qwen3_1_7b_thinking_opd_efficient_reasoning_sample.sbatch) |
| `full_vocab` | [config](configs/qwen3_1_7b_thinking_opd_efficient_reasoning_full_vocab.yaml) | [script](scripts/sbatch/qwen3_1_7b_thinking_opd_efficient_reasoning_full_vocab.sbatch) |
| `top_k_diff` | [config](configs/qwen3_1_7b_thinking_opd_efficient_reasoning_top_k_diff.yaml) | [script](scripts/sbatch/qwen3_1_7b_thinking_opd_efficient_reasoning_top_k_diff.sbatch) |

All three retain the existing OPD data, 256 prompts × 1 response, one epoch,
two nodes with four H100 GPUs each, vocabulary metrics, and `opd_token_top_k: 32`.
Training student prompts allow 2,048 tokens; teacher prompts allow 4,096 tokens.
Data loading first filters student prompts, then filters teacher prompts using
the teacher's tokenizer, concise instruction, and chat template. A training
sample is kept only when both limits are satisfied; prompts are not truncated.
The same filtering runs when training data is reloaded for checkpoint recovery.
Validation uses only its student prompt limit and is not filtered by the teacher.
Training and evaluation responses allow 16,384 tokens, with `val_n: 1` and the
existing 4,096-token validation prompt limit. Teacher scoring consumes the full
student response under the teacher prompt, so its context capacity is 20,481
tokens including verl's one extra scoring token. There is no separate teacher
response rollout. Response predictions and cached features are aligned by their
own prompt boundaries before computing the inherited OPD loss.

Submit any desired variant, or all three:

```bash
for mode in sample full_vocab top_k_diff; do
  sbatch "scripts/sbatch/qwen3_1_7b_thinking_opd_efficient_reasoning_${mode}.sbatch"
done
```

Use `teacher_prompt`, `teacher_max_prompt_length`, and
`teacher_enable_thinking` to configure the teacher-only context. These settings
are propagated to the Ray workers, support launcher overrides, and require
the shared teacher resource pool (`teacher_enable_resource_pool: false`).
Each variant has a distinct W&B experiment name and output-directory suffix.

Two additional sample variants train only the last 100 valid response tokens:

| Config suffix | Training trajectories |
| --- | --- |
| `sample_last100` | All generated trajectories |
| `sample_last100_verified` | Only trajectories whose answer verifier returns `acc == 1` |

Both inherit `qwen3_1_7b_thinking_opd_efficient_reasoning_sample.yaml`. Earlier
response tokens remain visible to student and teacher as causal context, with
zero token loss. Responses shorter than 100 tokens use all their valid tokens;
padding is excluded and the termination token is included when valid. The
verified variant uses the existing boxed-answer verifier on the complete
trajectory, without adding ground truth to the teacher prompt. Rejected
trajectories have zero loss, and batches with no selected tokens skip the
optimizer and LR scheduler update. Filtering does not refill the rollout batch.
Validation still scores full responses with the baseline settings.

The YAML options are `opd_loss_last_n_tokens: 100` and
`opd_require_verified: true`/`false`; their defaults preserve the original loss.
W&B records selected/filtered trajectory counts, loss-token counts, and skipped
updates under `train/opd/`. The new configurations and output directories have
distinct names. Submit either variant with the same two-node, four-H100-per-node
layout as the sample baseline:

```bash
sbatch scripts/sbatch/qwen3_1_7b_thinking_opd_efficient_reasoning_sample_last100_verified.sbatch
sbatch scripts/sbatch/qwen3_1_7b_thinking_opd_efficient_reasoning_sample_last100.sbatch
```

### Top-1 divergence sample OPD on eight H100 GPUs

`configs/qwen3_1_7b_thinking_opd_efficient_reasoning_sample_divergence_top1_h100_8gpu_train.yaml`
keeps the full-vocabulary eight-H100 recipe's data, teacher prompt, global batch
of 128, response lengths, and evaluation settings, and changes the objective to
native sampled-token `k1` OPD with `opd_loss_top_k_divergence: 1`.
For each complete response, rank valid sampled tokens by
`abs(log p_teacher(y_t) - log p_old_student(y_t))` and keep one position for
the actor loss. Scores use unclipped log probabilities before the actor update;
ties select the earliest position, padding is excluded, and an empty response
contributes no loss. The native sample loss keeps its own clipping and
policy-gradient behavior, averaging selected tokens within each response and
then responses over the global batch. Full causal context
and evaluation responses are retained.

This adapts the absolute log-likelihood difference in
[HDL, Equation (2)](https://arxiv.org/html/2609.36864v1#S3.SS2)
to the fixed OPD teacher and rollout student. It uses the existing teacher prompt
and does not generate hindsight reflections or branched rollouts.
W&B records selected-token counts and selected divergence scores under
`train/opd/`, in project `efficient-reasoning`.

The two-node, four-H100-per-node job has a ten-hour walltime limit:

```bash
sbatch scripts/sbatch/qwen3_1_7b_thinking_opd_efficient_reasoning_sample_divergence_top1_h100_8gpu_train.sbatch
```

`scripts/run_h100_8gpu_opd.py` uses the shared launcher with explicit
`h100_nnodes` and student/teacher `h100_runtime` settings, including eager mode,
chunked prefill, and the H100 reporting/cache-release hooks. The YAML can also
be validated with this launcher's `--dry-run` option.

## Prepared data and reproduction

| Dataset | Repository-relative path | Questions |
| --- | --- | ---: |
| DAPO-Math-17k | `data/train/dapo_math_17k.parquet` | 17,916 |
| AIME25 | `data/eval/aime25.parquet` | 30 |
| AIME26 | `data/eval/aime26.parquet` | 30 |
| HMMT February 2026 | `data/eval/hmmt26.parquet` | 33 |
| AMO-Bench-P | `data/eval/amobench.parquet` | 39 |

The pinned DAPO source has 17,917 rows. One question exactly matches an AIME26
question after case folding and whitespace removal, so it is excluded from
training during data preparation. This filtering does not detect paraphrases.
AMO-Bench-P retains the official number, set, and variable answer types; the 11
description questions from the 50-question full benchmark are excluded.

Data preparation needs only CPU-side dependencies:

```bash
pip install -r requirements-data.txt
python scripts/prepare_data.py            # Prepare missing files; reuse existing files.
python scripts/prepare_data.py --rebuild  # Regenerate using pinned sources/cache.
```

`scripts/prepare_data.py` contains test-data conversion for all four benchmarks
and calls `scripts/build_dapo_math_17k.py` for training data after the test files
are prepared. The latter still works on its own and removes overlaps against
evaluation files already present. Use the unified script to prepare all five
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

## Evaluation

The main OPD configs train for one prompt epoch. GRPO stops after 69 updates
to match that trajectory budget, using 32 prompts × 8 responses per update.
Both methods evaluate at steps 0, 15, 30, 45, 60, and 69, for six evaluations.
They use the same seven evaluation sets, excluding GSM8K and AIME25, with one
answer per question (`val_n: 1`), temperature 0.7, top-p 0.8, top-k 20, and a
16,384-token completion limit.

- `pass@k`: fraction of questions with at least one correct answer among k.
- `avg@k`: mean correct-answer fraction across the k samples per question.

Each evaluation point now has the same cumulative number of training
trajectories: 0, 3,840, 7,680, 11,520, 15,360, and 17,664. The methods visit
different numbers of prompts: OPD visits 17,664; GRPO visits 2,208 and samples
eight responses for each. GRPO's `num_epochs: 1` is an upper limit, while
`total_training_steps: 69` stops it before a complete prompt epoch. The 69-step
cap follows the current OPD data and batch; update it if either changes.

Values are in `[0, 1]`. Exact observed successes are used, without bootstrap
resampling. AIME and HMMT use boxed-answer Math-Verify grading. AMO uses its
parser protocol, including set comparison and the official variable probes.
Grading runs in separate CPU processes so parser timeouts work under verl.

To evaluate without training or loading an OPD teacher:

```bash
python scripts/run_baseline.py configs/qwen3_4b_base_grpo.yaml --eval-only
MODEL_PATH=./models/my-exported-hf-checkpoint \
  python scripts/run_baseline.py configs/qwen3_4b_base_grpo.yaml --eval-only
python scripts/run_baseline.py configs/qwen3_4b_base_grpo.yaml --eval-only --dry-run
```

Use an HF-format checkpoint for `MODEL_PATH`. The eval-only path initializes
verl's student workers and still requires the configured CUDA GPUs.

OPD metrics are written to `/scratch/l/luli/results/onestep/<run>/eval_metrics/<step>.json`.
Evaluation continues every 15 steps without saving generated answers as Parquet
or JSONL (`save_eval_generations: false`). Evaluation resumes training directly
from the model and optimizer state in memory; it does not require a disk checkpoint.

OPD saves every 25 steps (`save_steps: 25`) and keeps the latest completed
checkpoint (`max_checkpoints_to_keep: 1`). A checkpoint contains model and optimizer
shards, scheduler/RNG state, the dataloader position, and an HF-format export at
`/scratch/l/luli/results/onestep/<run>/global_step_<step>/actor/huggingface/`.
The launcher deletes older `global_step_*` directories only after verl writes
the new completion tracker. In-progress saves are left alone, and cleanup also
works after resuming. At step 300, only the step-300 checkpoint remains.

OPD run names are `<student>-<teacher>-opd-<UTC timestamp>`, for example
`Qwen3-4B-Base-Qwen3-8B-opd-20261008.120000`. GRPO uses
`<student>-grpo-<UTC timestamp>`. Model names follow the effective
`MODEL_PATH` and `TEACHER_MODEL_PATH` overrides when provided. Other recipe YAMLs
retain their configured save frequency, export contents, and output directory.

To recover an interrupted training job, use the same environment and recipe:

```bash
conda activate verl
python scripts/run_baseline.py configs/qwen3_4b_base_opd.yaml --resume \
  /scratch/l/luli/results/onestep/<student>-<teacher>-opd-<timestamp>
```

`--resume` accepts an absolute run directory (selecting its latest completed
checkpoint) or a specific `global_step_*` directory. It reuses the original run
directory and restores training through verl's native checkpoint loader.
Use the training checkpoint directory for recovery; `actor/huggingface/` alone
does not include optimizer or dataloader state. Keep the original model, teacher,
GPU count, and training settings when resuming. Recipe, launch command, Hydra
output, and offline W&B logs stay under the same run directory.
Training and evaluation share the selected prompt mode and completion limit;
the launcher rejects mismatched completion lengths.


### Aligned standard OPD recipe

`configs/config/qwen3_4b_base_opd.yaml` is the standard OPD recipe aligned with
Lightning-OPD's `configs/config/qwen3_4b_opd.yaml` for the prepared datasets and
W&B project/metric names. It retains the 150-update training budget and
25-update evaluation interval. The sbatch script uses
`configs/qwen3_4b_base_opd.yaml`, which trains for one epoch with `num_epochs: 1`,
`total_training_steps: null`, and evaluation every 15 updates.
Native verl computes the step count from the
filtered training dataloader. This preparation does not submit a Slurm job.

The 1.7B student variant, `configs/qwen3_1_7b_base_opd.yaml`, changes only the
student to `Qwen/Qwen3-1.7B-Base` and the experiment name. Its teacher, batches,
training steps, evaluation schedule and other parameters match the 4B recipe.
The launcher prepares its cached, pinned weights under
`models/Qwen3-1.7B-Base-chatml/` with the same ChatML stop/pad settings.
Both OPD sbatch scripts request eight H100 GPUs (two nodes × four GPUs) for
eight hours:

```bash
HF_HOME=/scratch/l/luli/hf python scripts/prepare_qwen3_base.py \
  --source Qwen/Qwen3-1.7B-Base --local-files-only
sbatch scripts/sbatch/qwen3_4b_base_opd.sbatch
sbatch scripts/sbatch/qwen3_1_7b_base_opd.sbatch
```

GRPO uses the same two-node allocation and the aligned 69-update trajectory
budget. Its 1.7B and 4B configs differ only in the student and experiment name:

```bash
sbatch scripts/sbatch/qwen3_1_7b_base_grpo.sbatch
sbatch scripts/sbatch/qwen3_4b_base_grpo.sbatch
```

Post-trained 0.6B and 1.7B students also have separate OPD recipes with a
`Qwen/Qwen3-4B-Instruct-2507` teacher. They use `mode: non-thinking`: the
student's original chat template with `enable_thinking=false` for training and
evaluation. Their models are loaded directly from the offline HF cache, with
no Base-model ChatML preparation. Other batch, epoch, data, and evaluation
settings match the main OPD configs. W&B experiment names include the student,
non-thinking mode, and teacher:

```bash
sbatch scripts/sbatch/qwen3_0_6b_non_thinking_opd.sbatch
sbatch scripts/sbatch/qwen3_1_7b_non_thinking_opd.sbatch
```

The explicitly selected batch is **256 prompts × 1 response**, with one update
per batch for one epoch and evaluation every 15 updates, before training, and
at the final update. With both students' tokenizers, the current training file
retains all 17,916 rows after prompt-length filtering and yields 69 full batches.
The native dataloader drops the final 252 rows. Evaluation runs at steps
0, 15, 30, 45, 60, and 69: six times in total.
This differs from the source standard OPD recipe's 64 × 4 while preserving its
256-sample update size. The seven eval sets exclude GSM8K and AIME25. Train and
eval parquet files point directly to `/scratch/l/luli/data/lightning_opd/`;
their native verl prompt, reward model and question IDs are preserved.

All other training/model settings retain the baseline recipe: Qwen3-4B-Base
student, Qwen3-8B teacher, lr 1e-6, warmup 25, weight decay 0.01, plain prompts,
rollout temperature 1.0 and response limit 16,384. The teacher prompt-length
reference is explicitly fixed to the training limit 2,048, instead of inheriting
the evaluation limit 4,096. Teacher capacity remains 18,433. Evaluation still
uses prompt limit 4,096 and response limit 16,384. The current Lightning-OPD
checkout is untouched by this migration.

Vocabulary options (K is independent of rollout-block K and sampling top_k):

```yaml
opd_loss_token_mode: sample
opd_token_top_k: 32
opd_enable_metrics: true
opd_distribution_chunk_size: 32
```

- `sample`: native k1 policy-gradient loss with Lightning-OPD floors, clipping,
  and per-response token means followed by a global response mean.
- `top_k_student`: select K largest next-token student probabilities.
- `top_k_teacher`: select K largest next-token teacher probabilities.
- `top_k_diff`: select K largest signed full-vocabulary differences p_teacher − p_student.
- `full_vocab`: use the full vocabulary.

For each top-k mode, select one shared support and independently renormalize
student and teacher over that support, then directly differentiate KL(student ||
teacher). This is a distribution loss, rather than the native sampled-token PPO
loss. Teacher probabilities are detached. Distribution losses use the native
`distillation_loss_max_clamp` (10); the native logprob floor (-10) applies to
sample loss only, so it does not distort the selected distributions or metrics.
All vocabulary modes use the shared `seq-mean-token-mean` loss reduction.

The three response-token overlap metrics use T=topK(teacher), S=topK(student),
D=topK(p_teacher − p_student). Each is 100 × intersection size / K, averaged
over valid response tokens, excluding prompt/padding:

- `train/opd/teacher_student_topk_overlap_pct`
- `train/opd/teacher_difference_topk_overlap_pct`
- `train/opd/student_difference_topk_overlap_pct`

W&B keeps native verl metrics and adds comparable `train/loss`, `train/pg_loss`,
`train/pg_clipfrac`, `train/ppo_kl`, `train/grad_norm`, `train/lr-pg_0`, and
`eval_metric/<dataset>-avg@1` names. Training steps remain zero-based, matching
Lightning-OPD. Both GRPO and OPD log evaluation steps as completed update counts,
so `eval/step` is 0, 15, 30, 45, 60, or 69; original verl steps remain available.
The run config records `training_framework: verl` and OPD-only vocabulary
options when applicable. Both methods record prompt visits and response samples
cumulatively; `eval/cumulative_response_samples` gives the shared trajectory
budget at each evaluation. Runs remain offline in project `opd`, entity
`efficiency-reasoning-with-opd`; preparation does not upload W&B runs.

Exact difference metrics require both full next-token distributions. An opt-in
Ray worker hook captures the teacher's final response hidden states during its
existing vLLM prompt-logprob pass and reconstructs logits in bounded chunks on
the actor. No second teacher forward is added. Native vLLM continuous batching,
TP and teacher sleep/wake scheduling remain in place. Cached response features
are deleted after each successful complete actor update; the fixed teacher
output head is retained. Long responses add shared-disk traffic and actor work:
for Qwen3-8B, the hidden cache is about 8 KiB per response token (up to 32 GiB for
256 maximum-length responses). These diagnostics therefore affect GPU-time
comparisons. Disable `opd_enable_metrics` with `sample` to use the original
teacher-scoring/actor path without feature capture.

The extension targets the installed vLLM 0.12, unquantized Qwen3 output heads,
text-only scoring and the existing FSDP2 remove-padding path with sequence
parallel size 1. CPU checks cover selected-support loss/gradients, signed
probability difference, chunked-prefill causal alignment, variable-length
packed batches, native sample delegation and Hydra/Ray configuration. A real
multi-GPU run is still required to validate the vLLM hook on H100s.

For fair comparisons, additionally track generated/trained response tokens and
GPU-hours; equal sample counts do not imply equal token budgets. Prompt format,
rollout temperature, reply limit, optimizer and initialization also need to be
reported because this recipe intentionally keeps baseline settings. verl/FSDP2
and slime/Megatron training checkpoints are not interchangeable; export HF
weights for weight-only transfer and retain each framework's native checkpoints
for optimizer/RNG resume.

```bash
python scripts/run_baseline.py configs/config/qwen3_4b_base_opd.yaml --dry-run
python -m pytest -q tests/test_opd_extensions.py tests/test_opd_recipe.py
# Submit only when a training run is desired:
sbatch scripts/sbatch/qwen3_4b_base_opd.sbatch
```
