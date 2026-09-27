"""Binary completion-token length reward; independent of answer correctness."""
from functools import partial
from numbers import Integral
from src.rewards.reference_reward import completion_text


def length_reward(prompts, completions, *, max_length, completion_ids=None,
                  tokenizer=None, **kwargs):
    """Return 1 for <= max_length tokens, else 0 (prompt tokens excluded).

    Prefer unpadded completion_ids from TRL; these count all generated tokens,
    including EOS when present. Text-only callers must supply a tokenizer;
    that fallback encodes visible completion text without extra special tokens.
    Empty completions receive 1 by the length rule alone.
    """
    if isinstance(max_length, bool) or not isinstance(max_length, Integral) or max_length < 0:
        raise ValueError("max_length must be a non-negative integer")
    if completion_ids is not None:
        if len(completion_ids) != len(completions):
            raise ValueError("Completion/token-ID count mismatch")
        lengths = [len(ids) for ids in completion_ids]
    else:
        if tokenizer is None:
            raise ValueError("Provide completion_ids or tokenizer for token length")
        lengths = [len(tokenizer.encode(completion_text(c), add_special_tokens=False))
                   for c in completions]
    return [float(n <= max_length) for n in lengths]


def make_length_reward(max_length, tokenizer=None):
    """Bind the threshold for use in GRPOTrainer(reward_funcs=[...])."""
    if isinstance(max_length, bool) or not isinstance(max_length, Integral) or max_length < 0:
        raise ValueError("max_length must be a non-negative integer")
    reward = partial(length_reward, max_length=max_length, tokenizer=tokenizer)
    reward.__name__ = "length_reward"
    return reward
