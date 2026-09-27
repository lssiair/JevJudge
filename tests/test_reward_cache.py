import pytest
from src.rewards.cache import RewardCache,BudgetExceeded
def test_hit_and_key(tmp_path):
 c=RewardCache(tmp_path/"cache.sqlite")
 key=c.key("m","p","r","c","v")
 assert c.get(key) is None
 c.put(key,{"reward":.5})
 assert c.get(key)=={"reward":.5}
 for vals in [("m2","p","r","c","v"),("m","p","r2","c","v"),("m","p","r","c","v2")]:
  assert c.key(*vals)!=key
 c.close()
def test_cross_process_budget(tmp_path):
 p=tmp_path/"c.sqlite";a=RewardCache(p);b=RewardCache(p)
 a.reserve("run",2);b.reserve("run",2)
 with pytest.raises(BudgetExceeded):a.reserve("run",2)
 a.close();b.close()

def test_budget_shared_across_runs(tmp_path,monkeypatch):
 monkeypatch.setenv("JEV_BUDGET_ID","suite")
 c=RewardCache(tmp_path/"scope")
 c.reserve("a",1)
 with pytest.raises(BudgetExceeded):c.reserve("b",1)
 c.close()
