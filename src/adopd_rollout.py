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


def blocked_eos(tokenizer):
    """Qwen's end-of-text token is also its BOS and a secondary EOS; neither may end a reasoning trace."""
    eos_ids = {tokenizer.eos_token_id, tokenizer.bos_token_id} - {None}
    return [tokenizer.decode([token]) for token in sorted(eos_ids)]


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
            blocked = blocked_eos(self.tokenizer)
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

    from src.adopd_options import eval_variants

    class AdopdEvalAgentLoop(SingleTurnAgentLoop):
        """Every eval variant for each question (plus the plain prompt when asked), all from one long rollout.

        A rollout with a shorter limit is the long rollout cut at that limit. A forced variant keeps the
        rollout if it closed its trace within the budget; otherwise it keeps the first budget reasoning
        tokens, inserts </think>, and generates only the answer within its limit.
        """

        async def run(self, sampling_params, **kwargs):
            options = runtime_options()
            # Without explicit variants: one unforced evaluation at the configured limit, reported under eval/.
            variants = eval_variants(options) or [dict(
                name='', budget=options['adopd_reasoning_budget'],
                limit=options['eval_max_completion_length'], forced=False)]
            messages = list(kwargs['raw_prompt'])
            plain = await self.apply_chat_template(messages)
            # One rollout per distinct prompt: the budget instruction for each budget, or the plain question.
            def asking(variant):
                return 'plain' if variant.get('prompt', 'budget') == 'plain' else variant['budget']
            prompts = {key: plain if key == 'plain' else
                       await self.apply_chat_template(student_messages(messages, key))
                       for key in {asking(variant) for variant in variants}}
            halt = halt_id(self.tokenizer)
            eos = {self.tokenizer.eos_token_id, self.tokenizer.bos_token_id} - {None}
            longest = max(variant['limit'] for variant in variants)

            async def scored(ids, rollout, variant):
                budget, limit = variant['budget'], variant['limit']
                extra = {}
                if not variant['forced'] or halt in rollout[:budget]:
                    tokens = rollout[:limit]
                else:
                    trace = rollout[:budget]
                    while trace and trace[-1] in eos:  # an early end-of-text is not an answer
                        trace.pop()
                    trace.append(halt)
                    answer = await self.server_manager.generate(request_id=uuid4().hex, prompt_ids=list(ids) + trace,
                        sampling_params=dict(sampling_params, max_tokens=limit - len(trace)))
                    tokens = trace + list(answer.token_ids)[:limit - len(trace)]
                if variant['forced']:
                    extra['forced_halt'] = float(halt not in rollout[:budget])
                score = await answer_score(self.tokenizer, tokens, halt, budget, kwargs['data_source'],
                                           kwargs['reward_model']['ground_truth'])
                # verl pads eval responses to the training length but never truncates them; scores and this
                # length cover the full rollout.
                return tokens, dict(score, full_response_length=len(tokens), **extra)

            asked = dict(prompts)
            # The trainer decides per evaluation step whether the plain prompt runs too (as the first variant).
            if kwargs['adopd_eval_plain']:
                asked['plain'] = plain
            metrics = {}
            with simple_timer('generate_sequences', metrics):
                rollouts = dict(zip(asked, await asyncio.gather(*(
                    free_rollout(self.server_manager.generate, ids, sampling_params, longest) for ids in asked.values()))))
                jobs = [scored(prompts[asking(v)], rollouts[asking(v)], v) for v in variants]
                if kwargs['adopd_eval_plain']:
                    jobs.append(scored(plain, rollouts['plain'], variants[0]))
                results = await asyncio.gather(*jobs)
            # The first variant is the main result verl reports; every named variant also gets its own prefix.
            first_tokens, score = results[0]
            score = dict(score)
            for variant, (_, result) in zip(variants, results):
                if variant['name']:
                    score.update({f"{variant['name']}_{key}": value for key, value in result.items()})
            if kwargs['adopd_eval_plain']:
                plain_tokens, plain_score = results[-1]
                score.update({f'nobudget_{key}': value for key, value in plain_score.items()},
                             nobudget_response_length=len(plain_tokens))
            tokens = first_tokens[:self.response_length]
            prompt = prompts[asking(variants[0])]
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
