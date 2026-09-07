"""Attach exact benchmark metrics and per-step JSON output to native validation."""

import json
from collections import Counter
from pathlib import Path
import re

from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from eval.eval_metrics import compute_eval_metrics


class BaselineTrainer(RayPPOTrainer):
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        super()._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        # Native verl filters overlong prompts. Refuse partial benchmark scores.
        import pyarrow.parquet as pq

        expected = Counter()
        files = self.config.data.val_files
        if isinstance(files, str):
            files = [files]
        for filename in files:
            expected.update(pq.read_table(filename, columns=["data_source"])["data_source"].to_pylist())
        actual = Counter(self.val_dataset.dataframe["data_source"])
        for name in ("mbppplus", "gpqa_diamond"):
            if actual[name] != expected[name]:
                raise ValueError(
                    f"{name}: retained {actual[name]}/{expected[name]} evaluation questions. "
                    "Increase val_max_prompt_length or remove data.val_max_samples; partial scores are disabled."
                )

    def _load_checkpoint(self):
        if self.config.trainer.get("val_only", False) and self.config.trainer.resume_mode == "resume_path":
            path = Path(self.config.trainer.resume_from_path)
            match = re.fullmatch(r"global_step_(\d+)", path.name)
            if match is None or not (path / "actor").is_dir():
                raise ValueError(f"Invalid evaluation checkpoint: {path}")
            # Evaluation restores actor weights without restoring training data progress.
            self.actor_rollout_wg.load_checkpoint(str(path / "actor"), del_local_after_load=False)
            self.global_steps = int(match.group(1))
            return
        return super()._load_checkpoint()

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        metrics = compute_eval_metrics(
            data_sources, sample_uids, reward_extra_infos_dict["acc"],
            k=self.config.actor_rollout_ref.rollout.val_kwargs.n,
        )
        target = Path(self.config.trainer.default_local_dir) / "eval_metrics"
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{self.global_steps}.json").write_text(json.dumps(metrics, indent=2) + "\n")
        return metrics
