"""Top-k selection without saving full-vocabulary logits for backward."""

import torch


class _GatherLastDim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, indices):
        # The derivative needs indices and the input shape, not logits values.
        # torch.gather's default backward also saves the full input tensor.
        ctx.input_shape = logits.shape
        ctx.save_for_backward(indices)
        return logits.gather(-1, indices)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        # The LM head still needs one dense gradient. Accumulate candidate
        # gradients here after all losses on the selected logits have combined.
        grad_logits = grad_output.new_zeros(ctx.input_shape)
        grad_logits.scatter_add_(-1, indices, grad_output)
        return grad_logits, None


def selected_log_probs(logits, indices):
    """Normalize on the selected support, retaining only small tensors."""
    selected = _GatherLastDim.apply(logits, indices.long())
    return selected.float().log_softmax(-1)
