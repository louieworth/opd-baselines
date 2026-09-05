"""Native verl training with exact benchmark evaluation metrics."""

import os

import hydra
import ray

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer import main_ppo
from verl.utils.device import auto_set_device

from eval.trainer import BaselineTrainer
from src.config import ROOT
from src.eopd import EOPDMixin
from src.opd import OPDTopKMixin
from src.opsd import OPSDLossMixin
from src.trd import TRDTrajectoryMixin


class OPDTrainer(OPDTopKMixin, BaselineTrainer):
    """Use optional top-k k1 with native student rollout and evaluation."""


class TRDTrainer(TRDTrajectoryMixin, OPDTrainer):
    """Use teacher-refined trajectories with native OPD training and evaluation."""


class OPSDTrainer(OPSDLossMixin, BaselineTrainer):
    """Use pointwise-clipped reverse KL with native rollout and evaluation."""


class EOPDTrainer(EOPDMixin, BaselineTrainer):
    """Use top-k reverse KL with an entropy-gated reverse-KL auxiliary loss."""


class BaselineTaskRunner(main_ppo.TaskRunner):
    def run(self, config):
        # Install the trainer only within this recipe's Ray driver process.
        os.chdir(ROOT)
        if config.get("opsd", {}).get("enabled", False):
            main_ppo.RayPPOTrainer = OPSDTrainer
        elif config.get("eopd", {}).get("enabled", False):
            main_ppo.RayPPOTrainer = EOPDTrainer
        elif config.get("trd", {}).get("enabled", False):
            main_ppo.RayPPOTrainer = TRDTrainer
        elif config.get("opd", {}).get("topk_enabled", False):
            main_ppo.RayPPOTrainer = OPDTrainer
        else:
            main_ppo.RayPPOTrainer = BaselineTrainer
        return super().run(config)


@hydra.main(config_path="../third_party/verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    main_ppo.run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(BaselineTaskRunner))


if __name__ == "__main__":
    main()
