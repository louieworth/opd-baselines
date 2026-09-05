"""Native verl extension classes for the plaint prompt mode."""

from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.dataset.rl_dataset import RLHFDataset

from src.prompt_modes import PlaintAgentMixin, PlaintDatasetMixin


class PlaintDataset(PlaintDatasetMixin, RLHFDataset):
    """Use plain prompt lengths for both training and validation datasets."""


class PlaintAgentLoop(PlaintAgentMixin, SingleTurnAgentLoop):
    """Generate from plain token IDs using the native single-turn rollout loop."""
