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


async def free_rollout(generate, prompt_ids, sampling_params, response_budget):
    """Unforced evaluation rollout: the policy alone decides when, or whether, to close its trace."""
    output = await generate(request_id=uuid4().hex, prompt_ids=list(prompt_ids),
                            sampling_params=dict(sampling_params, max_tokens=response_budget))
    return list(output.token_ids)[:response_budget]


async def answer_score(tokenizer, tokens, halt_token_id, reasoning_budget, data_source, ground_truth):
    """Grade ONLY the final answer after the halt; a boxed value inside the trace is not an answer."""
    from src.reward_async import compute_score
    finished = halt_token_id in tokens
    stop = tokens.index(halt_token_id) if finished else len(tokens)
    answer = tokens[stop + 1:]
    score = await compute_score(data_source, tokenizer.decode(answer, skip_special_tokens=True), ground_truth)
    # No response_length: verl records the scored response's length itself during validation.
    score.update(reasoning_length=stop, answer_length=len(answer),
                 finished=float(finished), over_budget=float(stop > reasoning_budget))
    return score


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
            score = await answer_score(self.tokenizer, tokens, halt, opts['adopd_reasoning_budget'],
                                       kwargs['data_source'], kwargs['reward_model']['ground_truth'])
            score.update(forced_halt=float(not chosen[stop]))
            return AgentLoopOutput(prompt_ids=prompt, response_ids=tokens, response_mask=chosen,
                reward_score=score['acc'], num_turns=2, metrics=metrics,
                extra_fields={'reward_extra_info': score, 'turn_scores': [], 'tool_rewards': [],
                              'adopd_teacher_prompt_ids': teacher_prompt})

    return AdopdAgentLoop


def eval_agent_class():
    import asyncio
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
    from verl.utils.profiler import simple_timer
    from src.adopd_cache import halt_id
    from src.adopd_options import runtime_options

    class AdopdEvalAgentLoop(SingleTurnAgentLoop):
        """Evaluation never forces a halt: one rollout with the budget instruction, optionally one without."""

        async def run(self, sampling_params, **kwargs):
            options = runtime_options()
            budget = options['adopd_reasoning_budget']
            messages = list(kwargs['raw_prompt'])
            plain = await self.apply_chat_template(messages)
            prompt = await self.apply_chat_template(student_messages(messages, budget))
            # The trainer decides per evaluation step whether the plain prompt runs too.
            prompts = (prompt, plain) if kwargs['adopd_eval_plain'] else (prompt,)
            halt = halt_id(self.tokenizer)
            metrics = {}
            with simple_timer('generate_sequences', metrics):
                rollouts = await asyncio.gather(*(
                    free_rollout(self.server_manager.generate, ids, sampling_params, options['eval_max_completion_length'])
                    for ids in prompts))
            score, *plain_score = await asyncio.gather(*(
                answer_score(self.tokenizer, ids, halt, budget, kwargs['data_source'],
                             kwargs['reward_model']['ground_truth'])
                for ids in rollouts))
            # verl pads eval responses to the training length but never truncates them; the scores above
            # and these lengths cover the full rollout.
            tokens = rollouts[0][:self.response_length]
            score.update(full_response_length=len(rollouts[0]))
            if plain_score:
                score.update({f'nobudget_{key}': value for key, value in plain_score[0].items()},
                             nobudget_response_length=len(rollouts[1]))
            return AgentLoopOutput(prompt_ids=prompt, response_ids=tokens, response_mask=[1] * len(tokens),
                reward_score=score['acc'], num_turns=1, metrics=metrics,
                extra_fields={'reward_extra_info': score, 'turn_scores': [], 'tool_rewards': [],
                              'adopd_teacher_prompt_ids': plain})

    return AdopdEvalAgentLoop


def __getattr__(name):
    factories = {'AdopdAgentLoop': agent_class, 'AdopdEvalAgentLoop': eval_agent_class}
    if name in factories:
        cls = factories[name]()
        globals()[name] = cls
        return cls
    raise AttributeError(name)
