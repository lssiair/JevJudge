"""Fail smoke tests if optimization did not run or reward groups had no variation."""
import argparse,json,math
from pathlib import Path
import numpy as np
from src.common import read_jsonl,dump_json
def check(path):
    root=Path(path);done=json.loads((root/"completed.json").read_text())
    rows=[]
    for p in root.glob("rollouts-rank*.jsonl"):rows.extend(read_jsonl(p))
    rewards=np.array([r["reward"] for r in rows])
    groups={}
    for r in rows:groups.setdefault((r["step"],r["prompt"]),[]).append(r["reward"])
    stats=dict(run=path,optimizer_steps=done["global_step"],samples=len(rows),
               finite_rewards=bool(np.isfinite(rewards).all()),reward_variance=float(rewards.var()),
               variable_groups=int(sum(np.ptp(v)>0 for v in groups.values())),
               max_parameter_change=done["max_parameter_change"],
               checkpoint_exists=(root/"final"/"config.json").exists())
    dump_json(root/"smoke_check.json",stats)
    if not (stats["finite_rewards"] and stats["checkpoint_exists"] and stats["optimizer_steps"]>0
            and stats["max_parameter_change"]>0 and stats["variable_groups"]>0):
        raise RuntimeError(f"Smoke verification failed: {stats}")
    print(stats)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("run");check(p.parse_args().run)
