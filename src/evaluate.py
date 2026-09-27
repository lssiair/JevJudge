import argparse,asyncio
from pathlib import Path
import numpy as np
from src.common import read_jsonl,dump_json,append_jsonl,bootstrap,correlation
from src.rewards.reference_reward import reference_reward,extract_answer
from src.rewards.jev_reward import JevReward

async def evaluate(path,name,jev=True):
    rows=read_jsonl(path)
    refs=[r["reference"] for r in rows];cs=[r["completion"] for r in rows];ps=[r["prompt"] for r in rows]
    objective=reference_reward(ps,cs,refs)
    scores=None
    if jev:
        print(f"Evaluation Jev audit: at most {len(rows)} requests before retries",flush=True)
        e=JevReward(run_id="evaluation-"+name)
        scores=[]
        for i in range(0,len(rows),32):
            scores.extend(await e.score(ps[i:i+32],cs[i:i+32],refs[i:i+32]))
        e.cache.close()
    for i,r in enumerate(rows):
        r["objective_reward"]=objective[i]
        r["jev_reward"]=scores[i] if scores is not None else None
        r["invalid_answer"]=extract_answer(r["completion"]) is None
    output=Path("results")/(name+".evaluated.jsonl")
    output.write_text("")
    for r in rows:append_jsonl(output,r)
    lengths=[r["completion_tokens"] for r in rows]
    result=dict(name=name,n=len(rows),accuracy=bootstrap(objective),
        mean_completion_length=float(np.mean(lengths)),median_completion_length=float(np.median(lengths)),
        invalid_answer_rate=float(np.mean([r["invalid_answer"] for r in rows])),
        reference_reward=float(np.mean(objective)),jev_reward=None if scores is None else float(np.mean(scores)),
        reward_correlation=None if scores is None else correlation(scores,objective))
    dump_json("results/"+name+".metrics.json",result)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--name",required=True)
    p.add_argument("--no-jev",action="store_true");a=p.parse_args()
    print(asyncio.run(evaluate(a.input,a.name,not a.no_jev)))
