"""Post-hoc Jev scoring; never changes the optimized rewards."""
import argparse,asyncio
from pathlib import Path
from src.common import read_jsonl,dump_json
from src.rewards.jev_reward import JevReward
async def main(run,limit):
    rows=[]
    for path in sorted(Path(run).glob("rollouts-rank*.jsonl")):
        rows.extend(read_jsonl(path))
    rows=sorted(rows,key=lambda r:r["step"])
    # Fixed evenly spaced sample over the entire trajectory.
    if len(rows)>limit:
        import numpy as np
        rows=[rows[i] for i in np.linspace(0,len(rows)-1,limit,dtype=int)]
    print(f"Offline reward audit <= {len(rows)} strict requests before retries")
    e=JevReward(run_id=Path(run).name+"-audit")
    for i in range(0,len(rows),32):
        chunk=rows[i:i+32]
        scores=await e.score([r["prompt"] for r in chunk],[r["completion"] for r in chunk],[r["reference"] for r in chunk])
        for row,score in zip(chunk,scores):row["jev_reward"]=score
    dump_json(Path(run)/"offline_reward_audit.json",dict(scope="posthoc_sample",rows=rows))
    e.cache.close()
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--limit",type=int,default=200)
    a=p.parse_args();asyncio.run(main(a.run,a.limit))
