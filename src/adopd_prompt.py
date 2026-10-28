"""Budget-conditioned student prompts and unbudgeted V2 teacher prompts."""

from copy import deepcopy


def student_messages(messages, budget):
    """Prepend the budget instruction to the final user message without mutation."""
    if type(budget) is not int or budget < 1:
        raise ValueError('ADOPD reasoning budget must be a positive integer')
    messages = deepcopy(list(messages))
    for message in reversed(messages):
        if message.get('role') == 'user':
            content = message.get('content')
            if not isinstance(content, str):
                raise ValueError('ADOPD budget prompts require a text user message')
            message['content'] = (f'<budget:{budget}>\n'
                f'Use at most {budget} tokens for reasoning. Finish early when possible.\n\n'
                + content)
            return messages
    raise ValueError('ADOPD budget prompts require a user message')


def prompt_pair_ids(tokenizer, messages, budget):
    """Return (original teacher IDs, budget-conditioned student IDs)."""
    kwargs = dict(tokenize=True, add_generation_prompt=True, enable_thinking=True)
    teacher = list(tokenizer.apply_chat_template(list(messages), **kwargs))
    student = list(tokenizer.apply_chat_template(student_messages(messages, budget), **kwargs))
    return teacher, student
