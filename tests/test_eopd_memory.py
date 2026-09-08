"""EOPD gradient equivalence and graph lifetime, runnable without Ray or CUDA."""

import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import weakref

from omegaconf import OmegaConf
import pytest
import torch

from src.eopd import entropy_reverse_kl_terms, eopd_loss
from src.opd import topk_k1_terms
from src.topk import _GatherLastDim, selected_log_probs


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def loss_data_helpers(monkeypatch):
    """Supply only the data access/SP interfaces used by the real callbacks."""
    utils = ModuleType("verl.utils")
    utils.tensordict_utils = SimpleNamespace(
        get_non_tensor_data=lambda data, key, default=False: data.get(key, default),
    )
    ulysses = ModuleType("verl.utils.ulysses")
    ulysses.get_ulysses_sequence_parallel_world_size = lambda: 1
    ulysses.slice_input_tensor = lambda values, dim: values
    for name, module in (("verl", ModuleType("verl")), ("verl.utils", utils),
                         ("verl.utils.ulysses", ulysses)):
        monkeypatch.setitem(sys.modules, name, module)


def _nested(values):
    return SimpleNamespace(values=lambda: values)


def _loss_config():
    return SimpleNamespace(distillation_loss=OmegaConf.create({
        "loss_max_clamp": 0.4, "clip_ratio": 0.2, "clip_ratio_low": 0.1,
        "clip_ratio_high": 0.3, "clip_ratio_c": 3.0,
    }))


def _base_terms(logps, teacher, old):
    return topk_k1_terms(logps, teacher, old, signal_clip=0.4, clip_low=0.1, clip_high=0.3)


def _graph_nodes(output):
    seen, queue = set(), [output.grad_fn]
    while queue:
        node = queue.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        queue.extend(parent for parent, _ in node.next_functions)
    return [type(node).__name__ for node in seen]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("topk,threshold,coefficient", [(1, 0.0, 1.0), (16, 0.8, 1.0),
                                                      (64, 100.0, 1.0), (16, 0.0, 0.0)])
def test_eopd_shared_selection_matches_previous_loss_and_gradient(
    loss_data_helpers, dtype, topk, threshold, coefficient,
):
    generator = torch.Generator().manual_seed(2026)
    logits = torch.randn(1, 9, 67, generator=generator, dtype=dtype).requires_grad_()
    teacher, ids = torch.randn(9, 67, generator=generator).log_softmax(-1).topk(topk, dim=-1)
    teacher.requires_grad_()
    old = torch.randn(9, topk, generator=generator).log_softmax(-1).requires_grad_()
    data = {"teacher_ids": _nested(ids.int()), "teacher_logprobs": _nested(teacher),
            "opd_old_topk_log_probs": _nested(old)}
    actual = eopd_loss(config=None, distillation_config=_loss_config(), data=data,
                       student_logits=logits, entropy_threshold=threshold, aux_loss_coef=coefficient)

    # Reference is the pre-fix graph: one ordinary gather for each loss.
    reference = logits.detach().clone().requires_grad_()
    selected = reference.gather(-1, ids.unsqueeze(0)).float().log_softmax(-1)
    expected = _base_terms(selected, teacher.unsqueeze(0), old.unsqueeze(0))
    auxiliary = reference.gather(-1, ids.unsqueeze(0)).float().log_softmax(-1)
    expected.update(entropy_reverse_kl_terms(auxiliary, teacher.unsqueeze(0), threshold))
    expected["topk_teacher_mass"] = teacher.detach().exp().sum(-1).unsqueeze(0)
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key])

    mask = torch.tensor([[0, 1, 1, 0, 1, 1, 1, 0, 1]])
    reduce = lambda values: ((values["distillation_losses"] + coefficient * values["eopd_aux_losses"])
                             * mask).sum() / mask.sum()
    loss, ref_loss = reduce(actual), reduce(expected)
    assert _graph_nodes(loss).count("_GatherLastDimBackward") == 1
    assert "GatherBackward0" not in _graph_nodes(loss)
    loss.backward()
    ref_loss.backward()
    tolerance = {"rtol": 0.02, "atol": 2e-5} if dtype == torch.bfloat16 else {"rtol": 1e-5, "atol": 1e-7}
    torch.testing.assert_close(logits.grad, reference.grad, **tolerance)
    assert teacher.grad is None and old.grad is None
    assert torch.count_nonzero(logits.grad[:, mask[0] == 0]) == 0


def test_scoring_returns_only_frozen_candidate_probabilities(loss_data_helpers):
    logits = torch.randn(1, 5, 19)
    teacher, ids = torch.randn(5, 19).log_softmax(-1).topk(3, dim=-1)
    data = {"teacher_ids": _nested(ids), "teacher_logprobs": _nested(teacher), "opd_topk_scoring": True}
    with torch.no_grad():
        output = eopd_loss(config=None, distillation_config=_loss_config(), data=data,
                           student_logits=logits, entropy_threshold=0.8, aux_loss_coef=1.0)
    assert set(output) == {f"opd_old_topk_{i}" for i in range(3)}
    expected = logits.gather(-1, ids.unsqueeze(0)).float().log_softmax(-1)
    for i in range(3):
        value = output[f"opd_old_topk_{i}"]
        torch.testing.assert_close(value, expected[..., i])
        assert value.is_contiguous() and not value.requires_grad


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_selection_backward_handles_duplicate_indices_and_views(noncontiguous):
    logits = torch.randn(2, 4, 7, dtype=torch.float64)
    if noncontiguous:
        logits = logits.transpose(0, 1)
    logits.requires_grad_()
    indices = torch.ones(*logits.shape[:-1], 3, dtype=torch.long)
    indices[..., 0] = 2
    assert torch.autograd.gradcheck(lambda x: _GatherLastDim.apply(x, indices), (logits,))
    assert torch.autograd.gradgradcheck(lambda x: _GatherLastDim.apply(x, indices), (logits,))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_selected_probabilities_do_not_save_full_vocabulary_logits(dtype):
    logits = torch.randn(1, 32, 2048, dtype=dtype, requires_grad=True)
    ids = torch.randint(2048, (1, 32, 16))
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda value: (saved.append(value) or value), lambda value: value):
        output = selected_log_probs(logits, ids)
    assert all(value.numel() <= ids.numel() for value in saved)
    output.square().mean().backward()
    assert logits.grad.shape == logits.shape


def _load_forward_step():
    """Execute the actual engine method without importing its Ray/CUDA stack."""
    path = ROOT / "third_party/verl/verl/workers/engine/fsdp/transformer_impl.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FSDPEngineWithLMHead")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward_step")
    namespace = {"torch": torch, "TensorDict": torch.Tensor,
                 "get_device_name": lambda: "cpu", "get_device_id": lambda: "cpu"}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["forward_step"]


@pytest.mark.parametrize("sampled_loss", [False, True])
def test_engine_releases_unused_logits_and_preserves_training_gradients(sampled_loss):
    forward_step = _load_forward_step()
    module = torch.nn.Linear(7, 31)
    reference = torch.nn.Linear(7, 31)
    reference.load_state_dict(module.state_dict())
    ids = torch.tensor([2, 4, 9]).expand(5, 3)
    logits_refs, metadata = [], []

    def prepare_outputs(*, output, **kwargs):
        logits_refs.append(weakref.ref(output))
        return {"log_probs": output.float().log_softmax(-1)[:, 0].clone(),
                "distillation_losses": selected_log_probs(output, ids).square().mean(-1)}

    def loss_fn(*, model_output, **kwargs):
        loss = model_output["distillation_losses"].mean()
        if sampled_loss:
            loss = loss - model_output["log_probs"].mean()
        return loss, {}

    engine = SimpleNamespace(module=lambda input, use_cache: module(input),
                             prepare_model_inputs=lambda micro_batch: ({"input": micro_batch}, {}),
                             prepare_model_outputs=prepare_outputs, get_data_parallel_group=lambda: None)
    for _ in range(3):
        data = torch.randn(5, 7)
        loss, output = forward_step(engine, data, loss_fn, False)
        metadata.append(output)
        assert loss.requires_grad
        assert all(not value.requires_grad for value in output["model_output"].values())
        if not sampled_loss:
            # The top-k backward needs no logits values. A reporting-only
            # sampled-probability graph must not keep this allocation alive.
            assert logits_refs[-1]() is None
        loss.backward()

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            ref_logits = reference(data)
            ref_loss = ref_logits.gather(-1, ids).float().log_softmax(-1).square().mean()
            if sampled_loss:
                ref_loss = ref_loss - ref_logits.float().log_softmax(-1)[:, 0].mean()
        torch.testing.assert_close(loss, ref_loss)
        ref_loss.backward()
    for actual, expected in zip(module.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0.02, atol=2e-3)
    assert all(ref() is None for ref in logits_refs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA memory comparison requires a GPU")
def test_cuda_topk_backward_peak_is_lower():
    """Compare the old and new allocation patterns, including retained reports."""
    import gc

    device = "cuda"
    tokens, vocab, topk = 2048, 8192, 16
    hidden = torch.randn(tokens, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
    head = torch.randn(32, vocab, device=device, dtype=torch.bfloat16, requires_grad=True)
    ids = torch.randint(vocab, (tokens, topk), device=device)

    def run(optimized):
        hidden.grad = head.grad = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        logits = hidden @ head
        reports = {"log_probs": logits.log_softmax(-1)[:, 0].clone()}
        if optimized:
            reports = {key: value.detach() for key, value in reports.items()}
            first = second = selected_log_probs(logits, ids)
        else:
            first = logits.gather(-1, ids).float().log_softmax(-1)
            second = logits.gather(-1, ids).float().log_softmax(-1)
        loss = first.square().mean() + second.exp().square().mean()
        del logits
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        return peak

    old_peak, new_peak = run(False), run(True)
    assert new_peak < old_peak, (old_peak, new_peak)
