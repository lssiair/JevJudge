"""Aggregate measured results; never fill missing experiments with fabricated metrics."""
import json, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.common import read_jsonl,dump_json,append_jsonl,bootstrap,correlation

def main():
    root=Path("results");root.mkdir(exist_ok=True)
    plots=root/"plots";plots.mkdir(exist_ok=True)
    metrics=[json.loads(p.read_text()) for p in sorted(root.glob("*.metrics.json"))]
    table=[dict(arm=m["name"],accuracy=m["accuracy"]["mean"],ci95=m["accuracy"]["ci95"],
                mean_length=m["mean_completion_length"],invalid_rate=m["invalid_answer_rate"],
                jev_reward=m["jev_reward"]) for m in metrics]
    pd.DataFrame(table).to_csv(root/"metrics.csv",index=False)
    evaluated={p.name.replace(".evaluated.jsonl",""):read_jsonl(p) for p in root.glob("*.evaluated.jsonl")}
    paired={}
    seed_statistics={}
    for arm in ("base","reference","jev"):
        subset=[m for m in metrics if m["name"].startswith(arm+"-seed")]
        if subset:
            seed_statistics[arm]=bootstrap([m["accuracy"]["mean"] for m in subset])
            seed_statistics[arm]["unit"]="independent training seed"
    seed_pairs={}
    for name in evaluated:
        if name.startswith("reference-seed") and name.replace("reference-","jev-",1) in evaluated:
            left={r["example_id"]:r for r in evaluated[name]}
            right={r["example_id"]:r for r in evaluated[name.replace("reference-","jev-",1)]}
            if set(left)!=set(right):raise ValueError("Seed evaluation sets differ")
            seed_pairs[name.split("-seed")[1]]=bootstrap([right[k]["objective_reward"]-left[k]["objective_reward"] for k in left])
    left_name,right_name=("reference","jev") if "reference" in evaluated else ("smoke-reference","smoke-jev")
    if left_name in evaluated and right_name in evaluated:
        a={r["example_id"]:r for r in evaluated[left_name]}
        b={r["example_id"]:r for r in evaluated[right_name]}
        if set(a)!=set(b):raise ValueError("Evaluation sets differ")
        paired=bootstrap([b[k]["objective_reward"]-a[k]["objective_reward"] for k in a])
        paired["comparison"]=right_name+" minus "+left_name
    disagreements=[]
    for arm,rows in evaluated.items():
        for r in rows:
            if r["jev_reward"] is not None and ((r["jev_reward"]>=.5)!=(r["objective_reward"]==1)):
                disagreements.append(dict(arm=arm,**r))
    disagreements.sort(key=lambda r:abs(r["jev_reward"]-r["objective_reward"]),reverse=True)
    for filename,rows in (("disagreement_cases.jsonl",disagreements),
                          ("reward_hacking_cases.jsonl",[r for r in disagreements if "jev" in r["arm"].split("-") and r["objective_reward"]==0])):
        (root/filename).write_text("")
        for r in rows:append_jsonl(root/filename,r)
    curves={}
    for run in sorted(Path("outputs").glob("*/completed.json")):
        directory=run.parent;rs=[]
        for p in directory.glob("rollouts-rank*.jsonl"):rs.extend(read_jsonl(p))
        if not rs:continue
        audit=directory/"offline_reward_audit.json"
        if audit.exists():
            lookup={(r["step"],r["prompt"],r["completion"]):r["jev_reward"] for r in json.loads(audit.read_text())["rows"]}
            for row in rs:
                if row["jev_reward"] is None:
                    row["jev_reward"]=lookup.get((row["step"],row["prompt"],row["completion"]))
        df=pd.DataFrame(rs)
        agg=df.groupby("step").agg(objective_accuracy=("objective_reward","mean"),mean_jev_reward=("jev_reward","mean"),
              mean_reward=("reward","mean"),completion_length=("completion_tokens","mean"),reward_std=("reward","std"))
        agg["correlation"]=[correlation(g.dropna(subset=["jev_reward"]).jev_reward,g.dropna(subset=["jev_reward"]).objective_reward)["pearson"]
                             for _,g in df.groupby("step")]
        agg["jev_audited_samples"]=df.groupby("step")["jev_reward"].count()
        curves[directory.name]=agg
        agg.to_csv(root/(directory.name+"-curves.csv"))
    for filename,column,ylabel in [
        ("accuracy_vs_training_step.png","objective_accuracy","Online rollout objective accuracy"),
        ("reward_vs_training_step.png","mean_reward","Optimized reward"),
        ("reward_correlation.png","correlation","Pearson(Jev, objective)"),
        ("completion_length_vs_step.png","completion_length","Completion tokens")]:
        fig,ax=plt.subplots(figsize=(7,4))
        curve_groups={}
        for name,df in curves.items():curve_groups.setdefault(name.split("-seed")[0],[]).append(df[column])
        for name,series in curve_groups.items():
            frame=pd.concat(series,axis=1)
            mean=frame.mean(axis=1)
            if mean.notna().any():
                ax.plot(mean.index,mean,label=name)
                if len(series)>1:
                    std=frame.std(axis=1)
                    ax.fill_between(mean.index,mean-std,mean+std,alpha=.15)
        ax.set(xlabel="Optimizer step",ylabel=ylabel)
        if ax.lines:ax.legend(fontsize=8)
        else:ax.text(.5,.5,"No measured data available",ha="center",transform=ax.transAxes)
        fig.tight_layout();fig.savefig(plots/filename,dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for arm,rows in evaluated.items():
        rows=[r for r in rows if r["jev_reward"] is not None]
        if rows:ax.scatter([r["jev_reward"] for r in rows],[r["objective_reward"] for r in rows],s=8,alpha=.2,label=arm)
    ax.set(xlabel="Jev reward",ylabel="Independent objective reward")
    if ax.collections:ax.legend()
    fig.tight_layout();fig.savefig(plots/"jev_vs_objective_scatter.png",dpi=160);plt.close(fig)
    db=sqlite3.connect("cache/jev_rewards.sqlite")
    requests=pd.read_sql_query("SELECT * FROM requests",db);db.close()
    efficiency={}
    for run,g in requests.groupby("run"):
        lat=g.loc[g.status=="ok","latency"]
        efficiency[run]=dict(request_attempts=len(g),failed_attempts=int((g.status=="failed").sum()),
            input_tokens=int(g.input_tokens.sum()) if g.input_tokens.notna().any() else None,
            output_tokens=int(g.output_tokens.sum()) if g.output_tokens.notna().any() else None,
            latency_p50=None if not len(lat) else float(lat.quantile(.5)),
            latency_p95=None if not len(lat) else float(lat.quantile(.95)),
            returned_models=sorted(g.model.dropna().unique().tolist()))
    fig,ax=plt.subplots(figsize=(6,4));ax.hist(requests.loc[requests.status=="ok","latency"].dropna(),bins=30)
    ax.set(xlabel="Jev latency (seconds)",ylabel="Successful requests")
    fig.tight_layout();fig.savefig(plots/"jev_latency_histogram.png",dpi=160);plt.close(fig)
    throughput={}
    for p in Path("outputs").glob("*/completed.json"):
        r=json.loads(p.read_text());meta=json.loads((p.parent/"metadata.json").read_text())
        samples=meta["expected_rollout_samples"]
        raw=[]
        for rp in p.parent.glob("rollouts-rank*.jsonl"):raw.extend(read_jsonl(rp))
        tokens=sum(r["completion_tokens"] for r in raw)
        hits=misses=0
        for rp in p.parent.glob("reward_stats-rank*.json"):
            q=json.loads(rp.read_text());hits+=q["cache_hits"];misses+=q["cache_misses"]
        gpu=[]
        if (p.parent/"gpu.jsonl").exists():
            for sample in read_jsonl(p.parent/"gpu.jsonl"):
                for line in sample["csv"].splitlines():
                    try:
                        index,util,memory=[float(v.strip()) for v in line.split(",")]
                        gpu.append((index,util,memory))
                    except ValueError:pass
        throughput[p.parent.name]=dict(gpu_utilization_mean=float(np.mean([g[1] for g in gpu])) if gpu else None,
                                       gpu_memory_peak_mib=max(g[2] for g in gpu) if gpu else None,wall_seconds=r["wall_seconds"],samples_per_second=samples/r["wall_seconds"],
                                       tokens_per_second=tokens/r["wall_seconds"],cache_hits=hits,cache_misses=misses,
                                       cache_hit_rate=hits/(hits+misses) if hits+misses else None,
                                       actual_metrics=r["metrics"])
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(list(throughput),[r["samples_per_second"] for r in throughput.values()])
    ax.set(ylabel="Rollout samples / second",title="Includes compilation and checkpoint saving");ax.tick_params(axis="x",rotation=25)
    fig.tight_layout();fig.savefig(plots/"training_throughput.png",dpi=160);plt.close(fig)
    validation=json.loads((root/"reward_validation.json").read_text()) if (root/"reward_validation.json").exists() else None
    complete=bool(validation and validation.get("scope")=="full_validation" and all(a in evaluated for a in ("base","reference","jev")) and
                  all((Path("outputs")/f"{s}-{a}"/"completed.json").exists() for s in ("smoke","pilot") for a in ("reference","jev")))
    summary=dict(status="completed" if complete else "smoke_only_or_incomplete",metrics=metrics,
                 paired_jev_minus_reference=paired,seed_statistics=seed_statistics,paired_by_seed=seed_pairs,
                 validation=validation,api_efficiency=efficiency,
                 training_efficiency=throughput,disagreement_count=len(disagreements),
                 full_three_seed_estimate=7473*4*3+1319*3*2)
    dump_json(root/"summary.json",summary)
    env=json.loads((root/"environment.json").read_text()) if (root/"environment.json").exists() else {}
    report=[
    "# Jev × GRPO Experiment",
    "## Setup",json.dumps(env,indent=2),
    "## Research question","Does changing only the GRPO reward source from deterministic final-answer correctness to Jev improve independently measured GSM8K accuracy?",
    "## Reward definitions","Reference = 1 if the extracted final numerical answer equals the reference as an exact rational number, else 0. Jev = answer.noul. Semantic validation uses scores / 4; composite = 0.35H + 0.35C + 0.15Q + 0.15I.",
    "## Jev reward validation",json.dumps(validation,indent=2),
    "## Training configuration","Matched starting checkpoint, prompt ordering, GRPO implementation, sampling, learning rate and rollout budget. See configs/base.yaml and run metadata.json. GPU0: vLLM; GPUs1–7: training. HelpSteer2 labels never enter API state or policy training.",
    "## Main results",pd.DataFrame(table).to_string(index=False) if table else "Not yet measured.",
    "## Reward agreement",json.dumps({m["name"]:m["reward_correlation"] for m in metrics},indent=2),
    "## Reward hacking",f"{len(disagreements)} observed disagreements saved. All observed E2 false positives are retained. No invented cases are added to meet a quota. Online rollout accuracy is not held-out accuracy.",
    "## Training efficiency",json.dumps(dict(api=efficiency,training=throughput),indent=2),
    "## Ablations","Strict Jev mode is the default. Grouped mode requires 100 real samples, Pearson >= .95, Spearman >= .90 and MAE <= .05. Hybrid and best-of-N are optional.",
    "## Limitations",("Smoke scores cannot establish effectiveness. " if not complete else "One-seed pilot results have limited training-seed coverage. ")+
    "Example bootstrap intervals do not measure training-seed uncertainty; seed statistics are reported separately when available. A strict parser can reject semantically valid responses. Jev aliases can change; returned model versions are logged. Timing includes compilation and checkpoint saving, so smoke runtimes do not isolate API overhead.",
    "## Conclusion",("Measured pilot completed; inspect the paired confidence interval before claiming improvement." if complete else "Only smoke tests or partial results are available. No claim of Jev superiority is supported."),
    json.dumps(dict(paired=paired,seed_statistics=seed_statistics,paired_by_seed=seed_pairs),indent=2),
    "## Reproduction","See README.md for exact commands. Formal pilot and multi-seed runs are left to the user.",
    ]
    (root/"REPORT.md").write_text("\n\n".join(report)+"\n")
    print("Report status:",summary["status"])
if __name__=="__main__":main()
