# GRPO and OPD training baselines

Portable Qwen3-4B-Base recipes with prepared train/test data, automatic Hugging
Face model preparation, and per-benchmark `pass@12` and `avg@12` evaluation.
Copy this repository, including `data/train/` and `data/eval/`, to the training
platform. Dataset files are regular files with repository-relative paths.
Only the prepared Parquet files under `data/train/` and `data/eval/` are tracked
under `data/`. No other local checkout is needed.

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
python scripts/run_baseline.py configs/qwen3_4b_opd.yaml
python scripts/run_baseline.py configs/qwen3_4b_grpo.yaml
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

The packaged OPD objective remains native verl sampled-token `k1` in policy
gradient mode, with no task-reward term. GRPO uses verifier reward.

```bash
# Print the command without downloading models, starting W&B, or claiming GPUs.
python scripts/run_baseline.py configs/qwen3_4b_opd.yaml --dry-run
python scripts/run_baseline.py configs/qwen3_4b_grpo.yaml --dry-run

# Existing local model directories or alternative HF IDs:
MODEL_PATH=./models/my-student TEACHER_MODEL_PATH=Qwen/Qwen3-14B \
  python scripts/run_baseline.py configs/qwen3_4b_opd.yaml

# Extra native Hydra overrides:
python scripts/run_baseline.py configs/qwen3_4b_grpo.yaml -- \
  trainer.total_training_steps=100
```

Non-default students must already have suitable tokenizer, padding, and stopping
settings. Edit the YAMLs to change batch size, GPU count, or rollout parameters.

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

GRPO also has a separate config for each mode:

| Config | Mode | W&B experiment name |
| --- | --- | --- |
| `configs/qwen3_4b_grpo_mode_thinking.yaml` | `thinking` | `qwen3_4b_grpo_mode_thinking` |
| `configs/qwen3_4b_grpo_mode_no_thinking.yaml` | `non-thinking` | `qwen3_4b_grpo_mode_no_thinking` |
| `configs/qwen3_4b_grpo_mode_plaint.yaml` | `plaint` | `qwen3_4b_grpo_mode_plaint` |

Run any of these with `python scripts/run_baseline.py <config>`. They inherit the
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

Both recipes evaluate before training and every 25 steps, including the last
step. Every evaluation generates 12 answers per question across all four
benchmarks, for 1,584 answers total. Defaults are temperature 0.7, top-p 0.8,
top-k 20, and a 16,384-token completion limit.

- `pass@12`: fraction of questions with at least one correct answer among 12.
- `avg@12`: mean correct-answer fraction across the 12 samples per question.

Values are in `[0, 1]`. Exact observed successes are used, without bootstrap
resampling. AIME and HMMT use boxed-answer Math-Verify grading. AMO uses its
parser protocol, including set comparison and the official variable probes.
Grading runs in separate CPU processes so parser timeouts work under verl.

To evaluate without training or loading an OPD teacher:

```bash
python scripts/run_baseline.py configs/qwen3_4b_grpo.yaml --eval-only
MODEL_PATH=./models/my-exported-hf-checkpoint \
  python scripts/run_baseline.py configs/qwen3_4b_grpo.yaml --eval-only
python scripts/run_baseline.py configs/qwen3_4b_grpo.yaml --eval-only --dry-run
```

Use an HF-format checkpoint for `MODEL_PATH`. The eval-only path initializes
verl's student workers and still requires the configured CUDA GPUs.

Metrics are written to `outputs/qwen3_4b/<method>/<run>/eval_metrics/<step>.json`.
Sample outputs are saved under the run's `validation/` directory. Checkpoints,
the effective recipe, launch command, and Hydra output stay under the same
relative `outputs/` tree. Training and evaluation share the selected prompt mode
and completion limit; the launcher rejects mismatched completion lengths.
