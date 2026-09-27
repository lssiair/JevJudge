import asyncio
from types import SimpleNamespace
import pytest
from src.rewards.jev_reward import JevReward,normalized,parse_response,retry_delay
from src.rewards.cache import RewardCache
@pytest.fixture(autouse=True)
def isolated_test_io(tmp_path,monkeypatch):
 monkeypatch.chdir(tmp_path)
 monkeypatch.delenv("JEV_BUDGET_ID",raising=False)
class Response:
 def __init__(self):
  self.nouls={"correct":SimpleNamespace(noul=.8)}
  self.model="test";self.usage=SimpleNamespace(model_dump=lambda:dict(input_tokens=10,output_tokens=1))
 def model_dump(self,**kwargs):return {"model":"test","answers":{"correct":{"type":"noul","noul":.8}}}
class Client:
 def __init__(self,fail=0):self.calls=0;self.fail=fail
 async def system_one(self,**kw):
  self.calls+=1
  if self.calls<=self.fail:raise ConnectionError("test")
  return Response()
def test_normalization():
 assert normalized(3,4)==.75
 for v in [float("nan"),float("inf"),-1,2]:
  with pytest.raises(ValueError):normalized(v)
def test_parse():
 assert parse_response(Response())["reward"]==.8
def test_retry_header():
 assert retry_delay(SimpleNamespace(headers={"retry-after":"2"}),0)==2
def test_cache_and_dedup(tmp_path):
 async def go():
  c=Client();e=JevReward(cache=RewardCache(tmp_path/"c"),client=c)
  assert await e.score(["p","p"],["c","c"],["r","r"])==[.8,.8]
  assert c.calls==1
  await e.score(["p"],["c"],["r"]);assert c.calls==1
 asyncio.run(go())
def test_failure_never_zero(tmp_path):
 async def go():
  c=Client(20);e=JevReward(cache=RewardCache(tmp_path/"c"),client=c);e.retries=0
  with pytest.raises(RuntimeError):await e.score(["p"],["c"],["r"])
  assert c.calls==1
 asyncio.run(go())
def test_retry_recovers(tmp_path):
 async def go():
  c=Client(1);e=JevReward(cache=RewardCache(tmp_path/"c"),client=c)
  assert await e.score(["p"],["c"],["r"])==[.8]
  assert c.calls==2
 asyncio.run(go())

def test_group_context_separates_cache():
 c=RewardCache.key
 assert c("m","p","r",["a","b"],"grouped")!=c("m","p","r",["a","c"],"grouped")
 assert c("m","p","r","a","strict")!=c("m","p","r","a","grouped")

def test_rate_limit_retry(tmp_path):
 class RateLimit(Exception):
  status=429
  headers={"retry-after":"0"}
 class Limited(Client):
  async def system_one(self,**kw):
   self.calls+=1
   if self.calls==1:raise RateLimit()
   return Response()
 async def go():
  c=Limited();e=JevReward(cache=RewardCache(tmp_path/"rate"),client=c)
  assert await e.score(["p"],["c"],["r"])==[.8]
  assert c.calls==2
 asyncio.run(go())
