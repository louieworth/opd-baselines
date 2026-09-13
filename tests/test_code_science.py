"""Offline data, grading, and saved-checkpoint regression tests (no GPUs)."""

import asyncio
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pyarrow.parquet as pq
import pytest

from data.prepare_code_science import SOURCES, convert_gpqa, convert_mbpp
from data.prompt_modes import CHOICE_ANSWER_INSTRUCTION, CODE_ANSWER_INSTRUCTION, plaint_prompt
from eval.baseline_reward import compute_score
from eval.code_science_reward import extract_choice
from train import load_config, main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def mac_reference_execution(monkeypatch):
    # EvalPlus's Linux RLIMIT_AS/DATA settings are not supported on macOS.
    # Only these fixed reference/synthetic tests omit the memory guard there.
    if sys.platform == "darwin":
        monkeypatch.setenv("EVALPLUS_MAX_MEMORY_BYTES", "-1")


@pytest.mark.parametrize("name", list(SOURCES))
def test_prepared_data_is_complete_and_has_benchmark_prompt(name):
    table = pq.read_table(ROOT / f"data/eval/{name}.parquet")
    rows = table.to_pylist()
    assert len(rows) == SOURCES[name]["count"]
    assert len({r["extra_info"]["index"] for r in rows}) == len(rows)
    assert table.schema.metadata[b"source_sha256"].decode() == SOURCES[name]["sha256"]
    for row in rows:
        assert row["data_source"] == name
        prompt = plaint_prompt(row["prompt"])
        assert prompt == row["prompt"][0]["content"]
        assert prompt.endswith(CODE_ANSWER_INSTRUCTION if name == "mbppplus" else CHOICE_ANSWER_INSTRUCTION)
    if name == "gpqa_diamond":
        assert set(r["reward_model"]["ground_truth"] for r in rows) == set("ABCD")


def test_gpqa_shuffle_is_reproducible_and_maps_the_correct_answer():
    raw = [{"Record ID": str(i), "Question": "Synthetic science question?", "Correct Answer": "correct",
            "Incorrect Answer 1": "wrong-one", "Incorrect Answer 2": "wrong-two", "Incorrect Answer 3": "wrong-three"}
           for i in range(20)]
    rows = convert_gpqa(raw)
    assert rows == convert_gpqa(raw)
    assert rows != convert_gpqa(raw, seed=1)
    assert len({r["reward_model"]["ground_truth"] for r in rows}) == 4
    for row in rows:
        truth = row["reward_model"]["ground_truth"]
        assert f"{truth}. correct\n" in row["prompt"][0]["content"]
        assert compute_score("gpqa_diamond", f"\\boxed{{{truth}}}", truth)["acc"] == 1


@pytest.mark.parametrize("text,expected", [
    (r"Reasoning about A. Final: \boxed{C}", "C"),
    (r"\boxed{A} followed by \boxed{D}", "D"),
    (r"\boxed{\text{B}}", "B"),
    ("The final answer is (B).", "B"),
    ("Reasoning\nAnswer: D", "D"),
    ("C", "C"),
    ("A seems plausible but B might work.", None),
    ("Answer: A\nActually I cannot determine the answer.", None),
    (r"\boxed{A or B}", None),
    (r"\boxed{A}\n\boxed{", None),
    (r"<think>\boxed{A}</think>I don't know", None),
    (r"<think>\boxed{A}", None),
    (r"<think>\boxed{B}</think>\boxed{D}", "D"),
    ("", None),
])
def test_gpqa_final_answer_extraction(text, expected):
    assert extract_choice(text) == expected


@pytest.fixture
def identity_task():
    return {
        "task_id": "Mbpp/9999", "prompt": '"""Return the input integer unchanged."""\n',
        "entry_point": "identity", "canonical_solution": "def identity(x):\n    return x\n",
        "base_input": [[1]], "plus_input": [[-1]], "atol": 0.0,
    }


def test_mbpp_hidden_data_stays_out_of_prompt(identity_task):
    row = convert_mbpp([identity_task])[0]
    assert json.loads(row["reward_model"]["ground_truth"]) == identity_task
    prompt = row["prompt"][0]["content"]
    assert "def identity" not in prompt
    assert "plus_input" not in prompt


@pytest.mark.parametrize("code,expected", [
    ("def identity(x):\n    return x", 1),
    ("def identity(x):\n    return abs(x)", 0),  # passes base, fails plus
    ("def identity(x):\n    return 0", 0),
    ("def unrelated(x):\n    return x", 0),
    ("def identity(:", 0),
    ("def identity(x):\n    raise RuntimeError('wrong')", 0),
    ("def identity(x):\n    while True:\n        pass\n    return x", 0),
])
def test_mbpp_execution_and_timeouts(identity_task, code, expected):
    pytest.importorskip("evalplus")
    result = compute_score("mbppplus", f"```python\n{code}\n```", json.dumps(identity_task))
    assert result["acc"] == expected


def test_mbpp_does_not_grade_code_from_thinking(identity_task):
    pytest.importorskip("evalplus")
    response = f"<think>```python\n{identity_task['canonical_solution']}\n```</think>I cannot solve this."
    assert compute_score("mbppplus", response, json.dumps(identity_task))["acc"] == 0


@pytest.mark.parametrize("task_id", ["Mbpp/2", "Mbpp/124", "Mbpp/252", "Mbpp/793"])
def test_mbpp_official_reference_solutions_and_special_input_types(task_id):
    pytest.importorskip("evalplus")
    rows = pq.read_table(ROOT / "data/eval/mbppplus.parquet").to_pylist()
    row = next(row for row in rows if row["extra_info"]["index"] == task_id)
    truth = row["reward_model"]["ground_truth"]
    task = json.loads(truth)
    assert compute_score("mbppplus", task["prompt"] + task["canonical_solution"], truth)["acc"] == 1


def test_async_grader_can_spawn_evalplus_execution(identity_task):
    pytest.importorskip("evalplus")
    from eval.reward_async import compute_score as score_async

    result = asyncio.run(score_async("mbppplus", identity_task["canonical_solution"], json.dumps(identity_task)))
    assert result["acc"] == 1
    assert asyncio.run(score_async("gpqa_diamond", r"\boxed{B}", "B"))["acc"] == 1


def test_mbpp_worker_does_not_reimport_slow_training_entrypoint(tmp_path, identity_task):
    pytest.importorskip("evalplus")
    script = tmp_path / "slow_training_entrypoint.py"
    marker = tmp_path / "entrypoint_reimported"
    script.write_text(textwrap.dedent(f"""\
        import asyncio
        import json
        from pathlib import Path
        import sys
        import time

        sys.path.insert(0, {str(ROOT)!r})
        if __name__ == '__mp_main__':
            Path({str(marker)!r}).touch()
            time.sleep(4)  # Longer than EvalPlus's three-second probe budget.

        from eval.reward_async import compute_score
        from eval import mbpp_worker

        async def run():
            truth = {json.dumps(identity_task)!r}
            codes = ['def identity(x):\\n    return x',
                     'def identity(x):\\n    return abs(x)',
                     'def identity(x):\\n    while True: pass',
                     'def identity(x):\\n    return ' + ' + '.join(['x'] * 1500),
                     'def identity(x):\\n    return x']
            results = await asyncio.gather(*(compute_score('mbppplus', code, truth) for code in codes))
            assert [result['acc'] for result in results] == [1, 0, 0, 0, 1], results
            assert results[3] == {{'score': 0.0, 'acc': 0.0, 'formatted': 0.0}}, results[3]
            pid = mbpp_worker._process.pid
            try:
                await compute_score('mbppplus', codes[0], '{{}}')
            except RuntimeError as exc:
                assert 'entry_point' in str(exc)
            else:
                raise AssertionError('worker errors must propagate')
            assert (await compute_score('mbppplus', codes[0], truth))['acc'] == 1
            assert mbpp_worker._process.pid == pid  # Reuse the worker and oracle cache.
            mbpp_worker._process.terminate()
            mbpp_worker._process.wait(timeout=5)
            try:
                await compute_score('mbppplus', codes[0], truth)
            except RuntimeError:
                pass
            else:
                raise AssertionError('worker exit must fail, not become a zero reward')

        if __name__ == '__main__':
            asyncio.run(run())
    """))
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=os.environ.copy(),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Exception ignored in atexit callback" not in result.stderr
    assert not marker.exists(), "EvalPlus must not import the training entrypoint"


def test_mbpp_reward_preflight_cli():
    pytest.importorskip("evalplus")
    result = subprocess.run([sys.executable, "-m", "eval.reward_async", "--check"], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EvalPlus reward self-check passed" in result.stdout


def test_mbpp_preflight_failure_stops_before_training_imports(monkeypatch):
    import builtins
    import train

    command = [sys.executable, "-m", "eval.reward_async", "--check"]
    run = Mock(side_effect=subprocess.CalledProcessError(1, command))
    monkeypatch.setattr(train.subprocess, "run", run)
    imports = Mock(side_effect=AssertionError("must stop before importing training dependencies"))
    monkeypatch.setattr(builtins, "__import__", imports)
    with pytest.raises(subprocess.CalledProcessError):
        train.preflight({"model_path": "Qwen/Qwen3-4B", "val_files": ["data/eval/mbppplus.parquet"]})
    run.assert_called_once_with(command, cwd=ROOT, check=True)
    imports.assert_not_called()


def test_checkpoint_cli_restores_step_without_teacher(capsys):
    config = str(ROOT / "configs/qwen3_4b_opd.yaml")
    assert main([config, "--eval-only", "--checkpoint", "outputs/test/global_step_25", "--dry-run",
                 "--benchmarks", "mbppplus", "gpqa_diamond"]) == 0
    output = capsys.readouterr().out
    assert "trainer.resume_mode=resume_path" in output
    assert "global_step_25" in output
    assert 'actor_rollout_ref.actor.checkpoint.load_contents=' in output
    assert "trainer.val_only=true" in output
    assert "distillation.enabled=false" in output
    assert "distillation.teacher_model.model_path=" not in output
    assert 'data.val_files=' in output
    assert "data/eval/aime25.parquet" not in output


def test_checkpoint_requires_eval_only():
    with pytest.raises(SystemExit):
        main(["configs/qwen3_4b_grpo.yaml", "--checkpoint", "outputs/test/global_step_25", "--dry-run"])


def test_checkpoint_flags_compose_in_native_hydra(capsys):
    from hydra import compose, initialize_config_dir
    import shlex

    main(["configs/qwen3_4b_opd.yaml", "--eval-only", "--checkpoint", "outputs/test/global_step_25", "--dry-run"])
    output = capsys.readouterr().out
    _, end = json.JSONDecoder().raw_decode(output)
    command = shlex.split(output[end:].strip())
    with initialize_config_dir(config_dir=str(ROOT / "third_party/verl/verl/trainer/config"), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=command[3:])
    assert cfg.actor_rollout_ref.actor.checkpoint.load_contents == ["model"]
    assert cfg.trainer.resume_from_path.endswith("global_step_25")
    assert cfg.trainer.val_only is True


def test_all_recipes_evaluate_both_benchmarks_every_checkpoint():
    for path in (ROOT / "configs").glob("qwen3_4b_*.yaml"):
        cfg = load_config(path)
        assert cfg["eval_steps"] == cfg["save_steps"] == 25
        assert cfg["val_n"] == 8
        assert "data/eval/mbppplus.parquet" in cfg["val_files"]
        assert "data/eval/gpqa_diamond.parquet" in cfg["val_files"]
        assert cfg["max_prompt_length"] == 2048
        assert cfg["val_max_prompt_length"] == 4096


def test_validation_prompt_config_does_not_mutate_training():
    from omegaconf import OmegaConf
    from data.prompt_modes import dataset_prompt_config
    config = OmegaConf.create({"max_prompt_length": 2048, "val_max_prompt_length": 4096})
    assert dataset_prompt_config(config, is_train=True) is config
    validation = dataset_prompt_config(config, is_train=False)
    assert validation.max_prompt_length == 4096
    assert config.max_prompt_length == 2048


def test_rollout_reserves_validation_context_without_changing_training_budget():
    from train import build_command
    config = load_config(ROOT / "configs/qwen3_4b_opd.yaml")
    command = build_command(config, Path("outputs/TEST"), [])
    overrides = dict(item.split("=", 1) for item in command[3:])
    assert overrides["data.max_prompt_length"] == "2048"
    assert overrides["+data.val_max_prompt_length"] == "4096"
    assert overrides["actor_rollout_ref.rollout.prompt_length"] == "4096"
    assert overrides["actor_rollout_ref.rollout.max_model_len"] == str(4096 + 16384)


def test_execution_environment_failure_is_not_reported_as_model_failure(monkeypatch):
    pytest.importorskip("evalplus")
    import evalplus.eval
    from eval.code_science_reward import check_execution_runtime
    monkeypatch.setattr(evalplus.eval, "untrusted_check", lambda *args, **kwargs: ("timeout", []))
    with pytest.raises(RuntimeError, match="known-correct execution probe.*status='timeout'.*start_method="):
        check_execution_runtime.__wrapped__()


def test_sanitizer_timeout_is_bounded(monkeypatch, identity_task):
    pytest.importorskip("evalplus")
    import eval.code_science_reward as grading
    from evalplus.eval.utils import TimeoutException
    def timeout(*args):
        raise TimeoutException("synthetic sanitizer timeout")
    monkeypatch.setattr(grading, "check_execution_runtime", lambda: None)
    monkeypatch.setattr(grading, "extract_code", timeout)
    assert grading.score_mbpp("malformed output", json.dumps(identity_task))["acc"] == 0


@pytest.mark.parametrize("stage", ["sanitize", "parse"])
def test_mbpp_recursion_failure_scores_zero_silently(monkeypatch, identity_task, stage, capsys, caplog):
    pytest.importorskip("evalplus")
    import eval.code_science_reward as grading
    import evalplus.sanitize as sanitizer

    monkeypatch.setattr(grading, "check_execution_runtime", lambda: None)
    oracle = Mock(side_effect=AssertionError("unprocessable code must not reach execution"))
    monkeypatch.setattr(grading, "mbpp_oracle", oracle)
    failure = Mock(side_effect=RecursionError("synthetic deep syntax tree"))
    with monkeypatch.context() as patch:
        if stage == "sanitize":
            patch.setattr(sanitizer, "sanitize", failure)
        else:
            patch.setattr(grading, "extract_code", lambda *args: identity_task["canonical_solution"])
            patch.setattr(grading.ast, "parse", failure)
        result = grading.score_mbpp(identity_task["canonical_solution"], json.dumps(identity_task))

    assert result == {"score": 0.0, "acc": 0.0, "formatted": 0.0}
    failure.assert_called_once()
    oracle.assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert not caplog.records


@pytest.fixture
def trainer_class(monkeypatch):
    module = ModuleType("verl.trainer.ppo.ray_trainer")
    class NativeTrainer:
        def _create_dataloader(self, *args):
            pass
        def _load_checkpoint(self):
            raise AssertionError("evaluation must not restore training state")
    module.RayPPOTrainer = NativeTrainer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = importlib.util.spec_from_file_location("code_science_trainer_test", ROOT / "eval/trainer.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded.BaselineTrainer


def test_eval_checkpoint_restores_weights_and_preserves_step(trainer_class, tmp_path):
    checkpoint = tmp_path / "global_step_25"
    (checkpoint / "actor").mkdir(parents=True)
    class Settings(dict):
        __getattr__ = dict.__getitem__
    trainer = trainer_class()
    trainer.config = SimpleNamespace(trainer=Settings(
        val_only=True, resume_mode="resume_path", resume_from_path=str(checkpoint)))
    trainer.actor_rollout_wg = SimpleNamespace(load_checkpoint=Mock())
    trainer._load_checkpoint()
    assert trainer.global_steps == 25
    trainer.actor_rollout_wg.load_checkpoint.assert_called_once_with(
        str(checkpoint / "actor"), del_local_after_load=False)


def test_filtered_gpqa_is_rejected_before_reporting_scores(trainer_class):
    trainer = trainer_class()
    trainer.config = SimpleNamespace(data=SimpleNamespace(val_files=[str(ROOT / "data/eval/gpqa_diamond.parquet")]))
    trainer.val_dataset = SimpleNamespace(dataframe={"data_source": ["gpqa_diamond"] * 197})
    with pytest.raises(ValueError, match="197/198"):
        trainer._create_dataloader(None, None, None, None)


def test_benchmark_metrics_use_full_test_correctness_and_sample_accuracy():
    from eval.eval_metrics import compute_eval_metrics
    metrics = compute_eval_metrics(["mbppplus"] * 2 + ["gpqa_diamond"] * 2,
                                   ["code"] * 2 + ["science"] * 2, [1, 0, 0, 1], k=2)
    assert metrics["eval/mbppplus/pass@1"] == 0.5
    assert metrics["eval/mbppplus/pass@2"] == 1.0
    assert metrics["eval/gpqa_diamond/accuracy"] == 0.5
    assert metrics["eval/gpqa_diamond/pass@2"] == 1.0
