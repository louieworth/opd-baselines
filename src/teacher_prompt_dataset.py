"""Filter training data against both student and formatted teacher prompt limits."""

from transformers import AutoTokenizer
from verl.utils.dataset.rl_dataset import RLHFDataset

from src.prompt_modes import PlaintDatasetMixin
from src.teacher_prompt import TeacherPromptTooLong, teacher_prompt_ids


class TeacherPromptDatasetMixin:
    def maybe_filter_out_long_prompts(self, dataframe):
        dataframe = super().maybe_filter_out_long_prompts(dataframe)
        settings = self.config.get('teacher_prompt_filter')
        if not settings or not self.filter_overlong_prompts:
            return dataframe
        if self.processor is not None:
            raise ValueError('Teacher-only prompt filtering supports text-only datasets')

        settings = dict(settings)
        tokenizer = AutoTokenizer.from_pretrained(settings['model_path'],
                                                  local_files_only=settings['local_files_only'])
        prompt_key = self.prompt_key

        def within_limit(row):
            try:
                teacher_prompt_ids(tokenizer, row[prompt_key], settings)
            except TeacherPromptTooLong:
                return False
            return True

        before = len(dataframe)
        dataframe = dataframe.filter(
            within_limit, num_proc=self.num_workers,
            desc=f'Filtering teacher prompts longer than {settings["teacher_max_prompt_length"]} tokens',
        )
        print(f'teacher prompt filter: {before} -> {len(dataframe)} '
              f'(removed {before - len(dataframe)})')
        return dataframe


class TeacherPromptDataset(TeacherPromptDatasetMixin, RLHFDataset):
    """Use student chat-template filtering followed by teacher-template filtering."""


class TeacherPromptPlaintDataset(TeacherPromptDatasetMixin, PlaintDatasetMixin, RLHFDataset):
    """Preserve plain student encoding when teacher-only prompting is requested."""
