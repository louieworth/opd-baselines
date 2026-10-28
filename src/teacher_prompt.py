"""Teacher-only concise context for fixed-teacher on-policy distillation."""

from copy import deepcopy


KEYS = ('teacher_prompt', 'teacher_max_prompt_length', 'teacher_enable_thinking')


class TeacherPromptTooLong(ValueError):
    """A teacher-formatted prompt exceeds the configured token limit."""


def options(config):
    prompt = config.get('teacher_prompt', '')
    if not isinstance(prompt, str):
        raise ValueError('teacher_prompt must be a string')
    limit = config.get('teacher_max_prompt_length', config.get('max_prompt_length', 2048))
    if type(limit) is not int or limit < 1:
        raise ValueError('teacher_max_prompt_length must be a positive integer')
    thinking = config.get('teacher_enable_thinking', False)
    if type(thinking) is not bool:
        raise ValueError('teacher_enable_thinking must be a boolean')
    return dict(teacher_prompt=prompt, teacher_max_prompt_length=limit,
                teacher_enable_thinking=thinking)


def has_teacher_prompt(config):
    return bool(config.get('teacher_prompt', '').strip())


def teacher_prompt_ids(tokenizer, messages, config):
    """Preserve the question/answer format; add concise context only for the teacher."""
    messages = deepcopy(list(messages))
    if len(messages) != 1 or messages[0].get('role') != 'user':
        raise ValueError('Teacher-only prompting currently requires a single user message')
    if not isinstance(messages[0].get('content'), str):
        raise ValueError('Teacher-only prompting currently supports text-only prompts')
    messages[0]['content'] = config['teacher_prompt'].strip() + '\n\n' + messages[0]['content']
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=config['teacher_enable_thinking'])
    if len(ids) > config['teacher_max_prompt_length']:
        raise TeacherPromptTooLong(f'Teacher prompt has {len(ids)} tokens; limit is {config["teacher_max_prompt_length"]}')
    return list(ids)
