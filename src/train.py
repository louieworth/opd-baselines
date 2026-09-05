"""Native verl training with exact benchmark evaluation metrics."""

import json
import os
from pathlib import Path

import hydra
import ray

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer import main_ppo
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.device import auto_set_device

from src.eval_metrics import compute_eval_metrics

ROOT = Path(__file__).resolve().parents[1]


class BaselineTrainer(RayPPOTrainer):
    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        metrics = compute_eval_metrics(
            data_sources, sample_uids, reward_extra_infos_dict["acc"],
            k=self.config.actor_rollout_ref.rollout.val_kwargs.n,
        )
        target = Path(self.config.trainer.default_local_dir) / "eval_metrics"
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{self.global_steps}.json").write_text(json.dumps(metrics, indent=2) + "\n")
        return metrics


class BaselineTaskRunner(main_ppo.TaskRunner):
    def run(self, config):
        # Install the trainer only within this recipe's Ray driver process.
        os.chdir(ROOT)
        main_ppo.RayPPOTrainer = BaselineTrainer
        return super().run(config)


@hydra.main(config_path="../third_party/verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    main_ppo.run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(BaselineTaskRunner))


if __name__ == "__main__":
    main()
