import argparse, asyncio, itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
from src.common import correlation,dump_json,bootstrap
from src.data.helpsteer2 import load_helpsteer2,DIMENSIONS
from src.rewards.jev_reward import JevReward

async def validate(limit=None):
    rows=load_helpsteer2(limit)
    print(f"HelpSteer2: {len(rows)} responses; strict requests <= {len(rows)} before retries",flush=True)
    engine=JevReward(run_id="reward-validation")
    # Batches limit task memory while retaining checkpointed cache on interruption.
    scored=[]
    for start in range(0,len(rows),32):
        scored.extend(await engine.semantic(rows[start:start+32]))
        print(f"Validated {len(scored)}/{len(rows)}; cache hits={engine.hits}",flush=True)
    records=[]
    for row,out in zip(rows,scored):
        record=dict(prompt=row["prompt"],response=row["response"])
        record.update({f"human_{d}":row[d]/4 for d in DIMENSIONS})
        record.update({f"jev_{d}":v for d,v in out["dimensions"].items()})
        records.append(record)
    df=pd.DataFrame(records)
    summary={}
    for d in DIMENSIONS:
        x,y=df[f"human_{d}"],df[f"jev_{d}"]
        pairs=[]
        for _,group in df.groupby("prompt",sort=False):
            for (_,a),(_,b) in itertools.combinations(group.iterrows(),2):
                h=a[f"human_{d}"]-b[f"human_{d}"]
                if h==0: continue
                j=a[f"jev_{d}"]-b[f"jev_{d}"]
                pairs.append(.5 if j==0 else float(h*j>0))
        summary[d]=dict(**correlation(x,y),mae=float(np.mean(abs(x-y))),
                        pairwise=bootstrap(pairs),human_ties_excluded=True,jev_tie_credit=.5)
    output=dict(status="completed",scope="full_validation" if limit is None else "smoke_subset",n=len(rows),prompts=df.prompt.nunique(),
                metrics=summary,cache_hits=engine.hits,cache_misses=engine.misses,
                rubric_version=scored[0]["rubric_version"],models=sorted({r["raw"]["model"] for r in scored}))
    dump_json("results/reward_validation.json",output)
    df.to_csv("results/reward_validation.csv",index=False)
    Path("results/reward_validation_summary.md").write_text(
        "# Jev reward validation\n\nHuman labels were excluded from API state. Scores normalized to [0,1].\n\n"+
        pd.DataFrame({d:{k:v for k,v in m.items() if k in ("pearson","spearman","mae")} for d,m in summary.items()}).T.to_markdown()+
        "\n\nPairwise details and sample counts: reward_validation.json\n")
    engine.cache.close()
    from src.reward_validation_plots import plot_validation
    output["plots"] = plot_validation()
    dump_json("results/reward_validation.json", output)
    return output

async def grouped_check(path):
    from src.common import read_jsonl
    rows=read_jsonl(path)
    if len(rows)<100: raise ValueError("Grouped calibration requires at least 100 real examples")
    strict=JevReward(run_id="grouped-calibration",mode="strict")
    grouped=JevReward(run_id="grouped-calibration",mode="grouped")
    p=[r["prompt"] for r in rows]; c=[r["completion"] for r in rows]; a=[r["reference"] for r in rows]
    x=await strict.score(p,c,a); y=await grouped.score(p,c,a)
    stats=dict(n=len(x),**correlation(x,y),mae=float(np.mean(np.abs(np.asarray(x)-y))))
    stats["approved"]=bool(stats["pearson"] is not None and stats["pearson"]>=.95 and
                           stats["spearman"]>=.9 and stats["mae"]<=.05)
    dump_json("results/grouped_agreement.json",stats)
    return stats

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--limit-prompts",type=int); p.add_argument("--grouped-check")
    p.add_argument("--plot-only", action="store_true", help="Plot saved validation JSON/CSV without calling Jev")
    a=p.parse_args()
    if a.plot_only:
        if a.grouped_check or a.limit_prompts is not None:
            p.error("--plot-only cannot be combined with --grouped-check or --limit-prompts")
        from src.reward_validation_plots import plot_validation
        plot_validation()
    else:
        print(asyncio.run(grouped_check(a.grouped_check) if a.grouped_check else validate(a.limit_prompts)))
