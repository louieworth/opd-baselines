"""Two-stage budget forcing. All generation stays on-policy except the marked halt."""

from uuid import uuid4

from src.adopd_prompt import student_messages


async def budget_rollout(generate, prompt_ids, sampling_params, halt_token_id,
                         reasoning_budget, response_budget, blocked_eos=()):
    """Return (tokens, policy mask); phase two shares phase one's sticky session."""
    session = uuid4().hex
    trace_params = dict(sampling_params, max_tokens=reasoning_budget,
                        stop_token_ids=[halt_token_id], ignore_eos=True)
    if blocked_eos:
        trace_params['bad_words'] = list(blocked_eos)
    trace = await generate(request_id=session, prompt_ids=list(prompt_ids), sampling_params=trace_params)
    tokens = list(trace.token_ids)
    if halt_token_id in tokens:
        tokens = tokens[:tokens.index(halt_token_id) + 1]
        chosen = [1] * len(tokens)
    else:
        tokens = tokens[:reasoning_budget]
        chosen = [1] * len(tokens)
        tokens.append(halt_token_id)
        chosen.append(0)
    remaining = response_budget - len(tokens)
    if remaining <= 0:
        raise ValueError('Reasoning exhausted the response budget before the answer')
    answer = await generate(request_id=session, prompt_ids=list(prompt_ids) + tokens,
                            sampling_params=dict(sampling_params, max_tokens=remaining))
    answer_tokens = list(answer.token_ids)[:remaining]
    return tokens + answer_tokens, chosen + [1] * len(answer_tokens)


# Loaded by the native verl agent registry only inside training/evaluation workers.
def agent_class():
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
    from verl.utils.profiler import simple_timer
    from src.adopd_cache import halt_id
    from src.adopd_options import runtime_options

    class AdopdAgentLoop(SingleTurnAgentLoop):
        async def run(self, sampling_params, **kwargs):
            opts = runtime_options()
            messages = list(kwargs['raw_prompt'])
            teacher_prompt = await self.apply_chat_template(messages)
            prompt = await self.apply_chat_template(
                student_messages(messages, opts['adopd_reasoning_budget']))
            halt = halt_id(self.tokenizer)
            # Qwen's end-of-text token is also its BOS and a secondary EOS.
            eos_ids = {self.tokenizer.eos_token_id, self.tokenizer.bos_token_id} - {None}
            blocked = [self.tokenizer.decode([token]) for token in sorted(eos_ids)]
            metrics = {}
            with simple_timer('generate_sequences', metrics):
                tokens, chosen = await budget_rollout(self.server_manager.generate, prompt,
                    sampling_params, halt, opts['adopd_reasoning_budget'], self.response_length, blocked)
            stop = tokens.index(halt)
            # Grade ONLY the final answer; a boxed value inside the trace is not an answer.
            from src.reward_async import compute_score
            score = await compute_score(kwargs['data_source'],
                        self.tokenizer.decode(tokens[stop + 1:], skip_special_tokens=True),
                        kwargs['reward_model']['ground_truth'])
            score.update(response_length=len(tokens), reasoning_length=stop,
                         answer_length=len(tokens) - stop - 1, forced_halt=float(not chosen[stop]))
            return AgentLoopOutput(prompt_ids=prompt, response_ids=tokens, response_mask=chosen,
                reward_score=score['acc'], num_turns=2, metrics=metrics,
                extra_fields={'reward_extra_info': score, 'turn_scores': [], 'tool_rewards': [],
                              'adopd_teacher_prompt_ids': teacher_prompt})

    return AdopdAgentLoop


def __getattr__(name):
    if name == 'AdopdAgentLoop':
        cls = agent_class()
        globals()[name] = cls
        return cls
    raise AttributeError(name)
