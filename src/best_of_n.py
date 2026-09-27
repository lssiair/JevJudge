"""Optional E4 control, evaluated by the independent final-answer evaluator."""
import argparse,asyncio
from collections import defaultdict
from pathlib import Path
from src.common import read_jsonl,append_jsonl
from src.rewards.jev_reward import JevReward
async def main(source,output):
    if Path(output).exists():raise FileExistsError(output)
    groups=defaultdict(list)
    for row in read_jsonl(source):groups[row["example_id"]].append(row)
    e=JevReward(run_id="best-of-n")
    for rows in groups.values():
        scores=await e.score([r["prompt"] for r in rows],[r["completion"] for r in rows],[r["reference"] for r in rows])
        selected=max(range(len(rows)),key=lambda i:scores[i])
        append_jsonl(output,dict(rows[selected],selector_reward=scores[selected],n_candidates=len(rows)))
    e.cache.close()
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();asyncio.run(main(a.input,a.output))
