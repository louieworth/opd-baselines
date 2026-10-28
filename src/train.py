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

from src.eval_data import with_eval_prompt_limit
from src.eval_metrics import compute_eval_metrics
from src.opd_options import needs_features, runtime_options, selects_response_tokens
from src.opd_response_mask import select_actor_loss
from src.teacher_prompt import has_teacher_prompt
from src.adopd_options import runtime_options as adopd_runtime_options
from src.adopd_trainer import AdopdTrainerMixin

ROOT = Path(__file__).resolve().parents[1]


class BaselineTrainer(RayPPOTrainer):
    def init_workers(self):
        if (self.config.distillation.enabled and has_teacher_prompt(runtime_options())
                and self.config.distillation.teacher_model.enable_resource_pool):
            raise ValueError("Teacher-only prompts require colocated, batched teacher scoring")
        super().init_workers()
        if self.config.distillation.enabled and has_teacher_prompt(runtime_options()):
            manager = self.teacher_model_manager
            manager.server_manager._opd_tokenizer = manager.rollout_replicas[0].model_config.tokenizer

    def _update_actor(self, batch):
        opts = runtime_options()
        if not self.config.distillation.enabled or not selects_response_tokens(opts):
            return self._update_actor_with_features(batch)
        if self.use_legacy_worker_impl != "disable":
            raise ValueError("OPD response selection requires the native FSDP2 engine worker.")
        actor_batch, selection_metrics = select_actor_loss(batch, opts)
        if selection_metrics["actor/opd/loss_tokens"] == 0:
            # AdamW/momentum and the LR scheduler must not advance without data.
            from verl.protocol import DataProto
            self._clear_teacher_features(batch.non_tensor_batch.get("opd_teacher_features"))
            selection_metrics["actor/distillation/loss"] = 0.0
            return DataProto.from_single_dict(data={}, meta_info={"metrics": selection_metrics})
        output = self._update_actor_with_features(actor_batch)
        output.meta_info["metrics"].update(selection_metrics)
        return output

    def _update_actor_with_features(self, batch):
        references = batch.non_tensor_batch.get("opd_teacher_features")
        if references is None:
            return super()._update_actor(batch)
        if self.use_legacy_worker_impl != "disable":
            raise ValueError("OPD vocabulary extensions require the native FSDP2 engine worker.")
        from verl.protocol import DataProto
        from verl.utils import tensordict_utils as tu
        from verl.utils.py_functional import rename_dict
        from verl.workers.utils.padding import left_right_2_no_padding

        # Same native batch, epochs, shuffle, temperature and aggregation. Only
        # enable its logits-processor callback for our requested diagnostics/loss.
        rollout = self.config.actor_rollout_ref.rollout
        actor = self.config.actor_rollout_ref.actor
        batch.meta_info["multi_turn"] = rollout.multi_turn.enable
        batch.meta_info["temperature"] = rollout.temperature
        batch_td = left_right_2_no_padding(batch.to_tensordict())
        mini_batch = actor.ppo_mini_batch_size * rollout.n
        tu.assign_non_tensor(batch_td, calculate_entropy=actor.entropy_coeff != 0.0,
                             distillation_use_topk=True, global_batch_size=mini_batch,
                             mini_batch_size=mini_batch, epochs=actor.ppo_epochs,
                             seed=actor.data_loader_seed, dataloader_kwargs={"shuffle": actor.shuffle},
                             compute_loss=True)
        try:
            output = tu.get(self.actor_rollout_wg.update_actor(batch_td), "metrics")
        finally:
            self._clear_teacher_features(references)
        output = rename_dict(output, "actor/")
        output["perf/mfu/actor"] = output.pop("actor/mfu")
        return DataProto.from_single_dict(data={}, meta_info={"metrics": output})

    @staticmethod
    def _clear_teacher_features(references):
        # All DP workers and microbatches have completed. Keep only the fixed
        # output head between steps; response features otherwise grow every step.
        if references is None:
            return
        from src.opd_runtime import cache_root
        root = cache_root().resolve()
        for reference in references:
            if reference is None:
                continue
            path = Path(reference["path"]).resolve()
            if path.is_relative_to(root) and path != root / "output_head.pt":
                path.unlink(missing_ok=True)
            else:
                raise ValueError("Unexpected teacher feature cache path.")

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        metrics = compute_eval_metrics(
            data_sources, sample_uids, reward_extra_infos_dict["acc"],
            k=self.config.actor_rollout_ref.rollout.val_kwargs.n,
            response_lengths=reward_extra_infos_dict.get("response_length"),
        )
        target = Path(self.config.trainer.default_local_dir) / "eval_metrics"
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{self.global_steps}.json").write_text(json.dumps(metrics, indent=2) + "\n")
        return metrics


class BaselineTaskRunner(main_ppo.TaskRunner):
    def run(self, config):
        # Install recipe extensions only within its Ray driver process.
        os.chdir(ROOT)
        from src.opd_logging import install_tracking
        install_tracking()
        opts = runtime_options()
        if needs_features(opts) or has_teacher_prompt(opts) or selects_response_tokens(opts):
            from src.opd_runtime import install
            install()
        adopd = adopd_runtime_options()
        if adopd.get('enabled', False):
            from src.adopd_runtime import install as install_adopd
            install_adopd()
        main_ppo.RayPPOTrainer = AdopdTrainer if adopd.get('enabled') or adopd.get('eval_only') else BaselineTrainer
        main_ppo.create_rl_dataset = with_eval_prompt_limit(main_ppo.create_rl_dataset)
        return super().run(config)


class AdopdTrainer(AdopdTrainerMixin, BaselineTrainer):
    """Answer-only self-distillation with an optional group-relative trace objective."""


@hydra.main(config_path="../third_party/verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    main_ppo.run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(BaselineTaskRunner))


if __name__ == "__main__":
    main()
