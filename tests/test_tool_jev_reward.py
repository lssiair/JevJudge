import asyncio
from types import SimpleNamespace
import pytest
from src.rewards.tool_jev_reward import ToolJevReward, combine_tool_rewards
from src.rewards.cache import RewardCache

GT='<think>x</think><tool_call>{"name":"lookup","parameters":{"id":1}}</tool_call>'
NO='<think>x</think><response>please clarify</response>'

class Client:
    def __init__(self): self.states=[]
    async def system_one(self,**kw):
        self.states.append(kw["state"])
        return SimpleNamespace(nouls={"correct":SimpleNamespace(noul=.9)},
            usage=SimpleNamespace(model_dump=lambda: {}),
            model="test",model_dump=lambda **kwargs: {"test":True})

def test_state_cache_and_no_calls(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("JEV_BUDGET_ID",raising=False)
    async def go():
        client=Client()
        e=ToolJevReward(cache=RewardCache(tmp_path/"db"),client=client)
        values=await e.score(["context"]*3,[GT,GT,NO],[GT,GT,NO])
        assert values==[.9,.9,None]
        assert len(client.states)==1
        assert client.states[0]["reference_tool_calls"][0]["name"]=="lookup"
        assert "reference_final_answer" not in client.states[0]
        await e.score(["context"],[GT],[GT])
        assert len(client.states)==1
        e.cache.close()
    asyncio.run(go())

def test_combination_preserves_format_and_scale():
    assert combine_tool_rewards([{"format_reward":1}]*3,[0,1,None])==[-2,4,1]

def test_failure_not_zero(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("JEV_BUDGET_ID",raising=False)
    class Failing:
        async def system_one(self,**kwargs):raise ConnectionError()
    async def go():
        e=ToolJevReward(cache=RewardCache(tmp_path/"db"),client=Failing())
        e.retries=0
        with pytest.raises(RuntimeError):await e.score(["p"],[GT],[GT])
        e.cache.close()
    asyncio.run(go())
