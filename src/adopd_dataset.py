"""Filter ADOPD prompts using the same budget instruction as student rollouts."""

from verl.utils.dataset.rl_dataset import RLHFDataset

from src.adopd_prompt import prompt_pair_ids


class AdopdDataset(RLHFDataset):
    def maybe_filter_out_long_prompts(self, dataframe):
        if self.processor is not None:
            raise ValueError('ADOPD budget prompts support text-only datasets')
        if not self.filter_overlong_prompts:
            return dataframe

        tokenizer, key = self.tokenizer, self.prompt_key
        budget, limit = self.config['adopd_reasoning_budget'], self.max_prompt_length

        def within_limit(row):
            teacher, student = prompt_pair_ids(tokenizer, row[key], budget)
            return max(len(teacher), len(student)) <= limit

        dataframe = dataframe.filter(within_limit, num_proc=self.num_workers,
            desc=f'Filtering budget-conditioned prompts longer than {limit} tokens')
        print(f'filter dataset len: {len(dataframe)}')
        return dataframe

    def __getitem__(self, item):
        row = super().__getitem__(item)
        if self.config.get('agent_name'):
            # verl runs each row with its own agent loop; validation selects the unforced evaluator.
            row['agent_name'] = self.config['agent_name']
        return row
