import pytest
from src.rewards.length_reward import length_reward, make_length_reward


def test_token_boundary_and_empty():
    ids = [[], [1, 2], [1, 2, 3], [1, 2, 3, 4]]
    assert length_reward([], [""] * 4, max_length=3, completion_ids=ids) == [1, 1, 1, 0]


def test_token_ids_take_priority_over_visible_text():
    assert length_reward([], ["a very long visible answer"], max_length=1,
                         completion_ids=[[42]]) == [1]


def test_tokenizer_and_chat_completion():
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return text.split()
    reward = make_length_reward(2, tokenizer=Tokenizer())
    assert reward([], ["one two", [{"role": "assistant", "content": "one two three"}]]) == [1, 0]
    assert reward.__name__ == "length_reward"


def test_missing_token_counter():
    with pytest.raises(ValueError):
        length_reward([], ["answer"], max_length=2)


def test_mismatched_count():
    with pytest.raises(ValueError):
        length_reward([], ["answer"], max_length=2, completion_ids=[])


@pytest.mark.parametrize("limit", [-1, 1.5, True, "512"])
def test_invalid_threshold(limit):
    with pytest.raises(ValueError):
        make_length_reward(limit)
    with pytest.raises(ValueError):
        length_reward([], [], max_length=limit, completion_ids=[])
