"""Attach exact benchmark metrics and per-step JSON output to native validation."""

import json
from pathlib import Path

from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from eval.eval_metrics import compute_eval_metrics


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
