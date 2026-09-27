"""Explicit user-invoked orchestration. Never starts a formal run on import."""
import argparse,os,signal,subprocess,sys,time,urllib.request,json
from pathlib import Path
from src.common import dump_json
PY=sys.executable

def run(args,env=None):
    print("+ "+" ".join(args),flush=True)
    subprocess.run(args,check=True,env=env)

def server():
    log=open("outputs/vllm-managed.log","a")
    p=subprocess.Popen(["bash","scripts/start_vllm.sh"],stdout=log,stderr=subprocess.STDOUT,
                       start_new_session=True)
    try:
        for _ in range(300):
            if p.poll() is not None:raise RuntimeError("vLLM exited; see outputs/vllm-managed.log")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health",timeout=2) as r:
                    if r.status==200:return p,log
            except Exception:pass
            time.sleep(2)
        raise TimeoutError("vLLM startup timed out")
    except BaseException:
        stop_server(p,log);raise

def stop_server(p,log):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    log.close()

def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["smoke","pilot","main"],default="smoke")
    p.add_argument("--eval-limit",type=int);a=p.parse_args()
    smoke=a.stage=="smoke"
    if a.stage=="main":
        for arm in ("reference","jev"):
            pilot=Path(f"outputs/pilot-{arm}/completed.json")
            metric=Path(f"results/{arm}.metrics.json")
            if not pilot.exists() or not metric.exists():
                raise RuntimeError("Complete and evaluate the pilot before main experiment")
            m=json.loads(metric.read_text())
            if m["accuracy"]["mean"]<=0 or m["invalid_answer_rate"]>=.5:
                raise RuntimeError("Pilot unhealthy; diagnose before scaling")
    seeds=[42,43,44] if a.stage=="main" else [42]
    prompts=28 if smoke else 1008 if a.stage=="pilot" else 7473
    steps=2 if smoke else 144 if a.stage=="pilot" else 1068
    eval_limit=a.eval_limit or (16 if smoke else 1319)
    # Strict upper estimate includes validation, E2 rollouts, independent audit and final reward audits.
    estimate=(16 if smoke else 1038)+len(seeds)*(steps*28+eval_limit*3+200)
    budget=int(os.environ.get("JEV_MAX_REQUESTS","10000"))
    print(f"Stage={a.stage}; prompts={prompts}; steps/arm={steps}; seeds={seeds}; estimated API requests <= {estimate} (before retries); budget={budget}",flush=True)
    if a.stage=="main" and "JEV_MAX_REQUESTS" not in os.environ:
        raise RuntimeError("Main experiment requires an explicit high JEV_MAX_REQUESTS")
    if estimate>budget:raise RuntimeError("Estimated requests exceed configured budget")
    os.environ["JEV_BUDGET_ID"]=a.stage
    dump_json(f"results/{a.stage}-plan.json",dict(stage=a.stage,seeds=seeds,prompts=prompts,steps=steps,eval_limit=eval_limit,estimated_requests=estimate,budget=budget))
    # Do not attach to or stop an unrelated existing server.
    import socket
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1",8000))==0:
            raise RuntimeError("Port 8000 is already in use. Stop your existing generation server first.")
    run([PY,"-m","src.prepare_model"])
    run(["bash","scripts/check_env.sh"])
    validation=[PY,"-m","src.reward_validation"]
    if smoke:validation+=["--limit-prompts","8"]
    run(validation)
    for seed in seeds:
        for arm in ("reference","jev"):
            name=f"{a.stage}-{arm}"+(f"-seed{seed}" if a.stage=="main" else "")
            target=f"outputs/{name}"
            process,log=server()
            try:
                env=dict(os.environ,CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7")
                run([PY,"-m","torch.distributed.run","--standalone","--nproc_per_node=7","-m","src.train_grpo",
                     "--config",f"configs/grpo_{arm}.yaml","--output",target,"--prompts",str(prompts),
                     "--steps",str(steps),"--seed",str(seed)],env)
            finally:stop_server(process,log)
            run([PY,"-m","src.smoke_check",target])
            run([PY,"-m","src.audit_rollouts","--run",target,"--limit","16" if smoke else "200"])
        for arm in ("base","reference","jev"):
            suffix=f"-seed{seed}" if a.stage=="main" else ""
            name=(a.stage+"-" if smoke else "")+arm+suffix
            model=os.environ.get("MODEL_PATH","cache/qwen3.5-4b-text") if arm=="base" else f"outputs/{a.stage}-{arm}{suffix}/final"
            output=f"outputs/eval-{name}.jsonl"
            run([PY,"-m","src.generate","--model",model,"--output",output,"--limit",str(eval_limit),"--seed",str(seed)],
                dict(os.environ,CUDA_VISIBLE_DEVICES="0"))
            run([PY,"-m","src.evaluate","--input",output,"--name",name])
    run([PY,"-m","src.analyze"])

if __name__=="__main__":main()
