import asyncio
from types import SimpleNamespace
from src.rewards.cache import RewardCache
from src.rewards.tool_jev_no_answer_reward import (
    ToolJevNoAnswerReward, format_score_no_answer,
)

CALL = '<think></think><tool_call>{"name":"lookup","parameters":{"id":1}}</tool_call>'
NO_CALL = '<think></think><response>Please provide the missing ID.</response>'


class Client:
    def __init__(self):
        self.states = []

    async def system_one(self, **kwargs):
        self.states.append(kwargs["state"])
        return SimpleNamespace(
            nouls={"correct": SimpleNamespace(noul=0.8)},
            usage=SimpleNamespace(model_dump=lambda: {}),
            model="test",
            model_dump=lambda **kwargs: {"test": True},
        )


def test_reference_free_api_state_and_no_call_scoring(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("JEV_BUDGET_ID", raising=False)

    async def go():
        client = Client()
        engine = ToolJevNoAnswerReward(
            cache=RewardCache(tmp_path / "db"), client=client
        )
        values = await engine.score(
            ['[{"role":"user","content":"Find ID 1"}]'] * 3,
            [CALL, CALL, NO_CALL],
        )
        assert values == [0.8, 0.8, 0.8]
        assert len(client.states) == 2
        for state in client.states:
            assert set(state) == {"dialogue_and_tools", "candidate_response"}
            assert "reference" not in repr(state).lower()
        await engine.score(
            ['[{"role":"user","content":"Find ID 1"}]'], [CALL]
        )
        assert len(client.states) == 2
        engine.cache.close()

    asyncio.run(go())


def test_reference_free_format():
    assert format_score_no_answer(CALL) == 1
    assert format_score_no_answer(NO_CALL) == 1
    assert format_score_no_answer('<think></think><tool_call>bad</tool_call>') == 0
    assert format_score_no_answer('<think></think><tool_call>{}</tool_call>') == 0
    assert format_score_no_answer('<think></think><tool_call>{}</tool_call><response>x</response>') == 0
