import argparse, asyncio, json, os, time, math, threading, subprocess
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,TrainerCallback
from trl import GRPOConfig,GRPOTrainer
from src.common import config,dump_json,append_jsonl,metadata,fingerprint,correlation
from src.credentials import load_credentials
from src.data.gsm8k import load_gsm8k
from src.rewards.reference_reward import reference_reward,completion_text
from src.rewards.jev_reward import JevReward

class Metrics(TrainerCallback):
    def __init__(self,path):
        self.path=path;self.start=time.perf_counter();self.stop=threading.Event()
    def on_train_begin(self,args,state,control,**kwargs):
        if not state.is_world_process_zero:return
        def sample():
            while not self.stop.is_set():
                r=subprocess.run(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True)
                append_jsonl(self.path.parent/"gpu.jsonl",dict(timestamp=time.time(),csv=r.stdout.strip()))
                self.stop.wait(2)
        self.worker=threading.Thread(target=sample,daemon=True);self.worker.start()
    def on_train_end(self,args,state,control,**kwargs):
        self.stop.set()
        if hasattr(self,"worker"):self.worker.join(timeout=5)
    def on_log(self,args,state,control,logs=None,**kwargs):
        logs=logs or {}
        if any(isinstance(v,(float,int)) and not math.isfinite(v) for v in logs.values()):
            raise RuntimeError("Non-finite training metrics")
        if state.is_world_process_zero:
            append_jsonl(self.path,dict(step=state.global_step,wall_seconds=time.perf_counter()-self.start,
                                       gpu_peak_bytes=torch.cuda.max_memory_allocated(),**logs))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.add_argument("--steps",type=int);p.add_argument("--prompts",type=int);p.add_argument("--seed",type=int)
    p.add_argument("--resume", help="Full checkpoint directory; use a fresh --output directory")
    p.add_argument("--save-steps", type=int)
    a=p.parse_args();cfg=config(a.config)
    if a.save_steps is not None:
        if a.save_steps < 1: raise ValueError("--save-steps must be positive")
        cfg["save_steps"]=a.save_steps
    resume=None
    if a.resume:
        from src.checkpoints import validate_checkpoint
        resume=validate_checkpoint(a.resume, world_size=int(os.environ.get("WORLD_SIZE","1")))
        cfg["resume_from_checkpoint"]=str(resume)
    for key,val in (("max_steps",a.steps),("train_prompts",a.prompts),("seed",a.seed)):
        if val is not None: cfg[key]=val
    if resume:
        previous=json.loads((resume.parent/"metadata.json").read_text())["config"]
        keys=("arm","model_path","seed","train_prompts","num_generations",
              "per_device_train_batch_size","gradient_accumulation_steps",
              "learning_rate","max_completion_length","beta","loss_type","reward_mode",
              "task","dataset_path","max_prompt_length")
        changed=[k for k in keys if previous.get(k)!=cfg.get(k)]
        if changed: raise ValueError(f"Resume configuration mismatch: {changed}")
        resume_step=json.loads((resume/"trainer_state.json").read_text())["global_step"]
        if resume_step >= cfg["max_steps"]: raise ValueError("Checkpoint already reached target max_steps")
    else:
        resume_step=0
    rank=int(os.environ.get("RANK","0"));world=int(os.environ.get("WORLD_SIZE","1"))
    if world!=7: raise RuntimeError("Use seven training GPUs and one dedicated vLLM GPU")
    out=Path(a.output)
    if (out/"completed.json").exists(): raise FileExistsError("Completed run exists; choose a fresh --output")
    if a.resume and rank==0 and out.exists() and any(out.iterdir()):
        raise FileExistsError("Resume requires a fresh output directory to preserve interrupted logs")
    out.mkdir(parents=True,exist_ok=True)
    task=cfg.get("task","gsm8k")
    if task not in ("gsm8k","tool_calling"): raise ValueError(f"Unknown task: {task}")
    if task=="tool_calling" and cfg["arm"] not in ("reference","jev","jev_no_answer"):
        raise ValueError("Tool-calling supports reference and jev arms only")
    uses_jev=cfg["arm"] in ("jev","hybrid","jev_no_answer")
    if uses_jev:
        load_credentials()
        validation=json.loads(Path("results/reward_validation.json").read_text())
        if validation.get("scope")!="full_validation" and cfg["max_steps"]>10:
            raise RuntimeError("Run full reward validation before Jev training")
        if validation["status"]!="completed": raise RuntimeError("Reward validation must complete")
        if cfg["reward_mode"]=="grouped":
            agreement=json.loads(Path("results/grouped_agreement.json").read_text())
            if not agreement.get("approved") or agreement["n"]<100: raise RuntimeError("Grouped calibration gate failed")
    tokenizer=AutoTokenizer.from_pretrained(cfg["model_path"],local_files_only=True,padding_side="left")
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    if task=="tool_calling":
        from src.data.tool_calling import load_tool_calling
        from src.rewards.tool_reward import tool_scores, VERSION
        cfg["tool_reward_version"]=VERSION
        ds=load_tool_calling(tokenizer,limit=cfg["train_prompts"],seed=cfg["seed"],
                            path=cfg["dataset_path"],max_prompt_length=cfg["max_prompt_length"])
    else:
        ds=load_gsm8k(limit=cfg["train_prompts"],seed=cfg["seed"])
    estimate=cfg["max_steps"]*world*cfg["per_device_train_batch_size"]*cfg["gradient_accumulation_steps"]
    print(f"Selected model: {cfg['model_path']}; task={task}; rollout samples={estimate}; Jev requests <= {estimate if uses_jev else 0}",flush=True)
    engine_class=JevReward
    if task=="tool_calling" and uses_jev:
        from src.rewards.tool_jev_reward import ToolJevReward, combine_tool_rewards, VERSION as JEV_TOOL_VERSION
        if cfg["arm"]=="jev_no_answer":
            from src.rewards.tool_jev_no_answer_reward import (
                ToolJevNoAnswerReward, format_score_no_answer, VERSION as JEV_NO_ANSWER_VERSION
            )
            cfg["jev_tool_reward_version"]=JEV_NO_ANSWER_VERSION
            engine_class=ToolJevNoAnswerReward
        else:
            cfg["jev_tool_reward_version"]=JEV_TOOL_VERSION
            engine_class=ToolJevReward
    engine=engine_class(run_id=out.name,mode=cfg["reward_mode"]) if uses_jev else None
    async def reward(prompts,completions,reference,problem,trainer_state=None,**kwargs):
        components=[tool_scores(c,r) for c,r in zip(completions,reference)] if task=="tool_calling" else None
        if components is not None and cfg["arm"]=="jev_no_answer":
            components=[dict(c,reference_format_reward=c["format_reward"],
                             format_reward=format_score_no_answer(completion))
                        for c,completion in zip(components,completions)]
        objective=[r["exact_match"] for r in components] if components is not None else reference_reward(prompts,completions,reference)
        reference_values=[r["reward"] for r in components] if components is not None else objective
        j=None
        if cfg["arm"]=="jev_no_answer":
            j=await engine.score(problem,completions)
        elif cfg["arm"] in ("jev","hybrid"):
            j=await engine.score(problem,completions,reference)
        values=reference_values if j is None else j if cfg["arm"] in ("jev","jev_no_answer") else [
            cfg["alpha"]*x+(1-cfg["alpha"])*y for x,y in zip(objective,j)]
        if task=="tool_calling" and j is not None:
            values=combine_tool_rewards(components,j)
            components=[dict(c,jev_probability=p,jev_correctness_reward=0.0 if p is None else 6*p-3,optimized_reward=v)
                        for c,p,v in zip(components,j,values)]
        step=trainer_state.global_step if trainer_state is not None else -1
        for i,c in enumerate(completions):
            text=completion_text(c)
            append_jsonl(out/f"rollouts-rank{rank}.jsonl",dict(step=step,prompt=problem[i],completion=text,
                          reference=reference[i],objective_reward=objective[i],jev_reward=None if j is None else j[i],
                          task=task,reward_components=None if components is None else components[i],
                          reward=values[i],completion_chars=len(text),completion_tokens=len(tokenizer.encode(text,add_special_tokens=False))))
        return values
    reward.__name__=cfg["arm"]+"_reward"
    fields=("learning_rate","num_generations","temperature","top_p","max_completion_length",
            "bf16","tf32","gradient_checkpointing","per_device_train_batch_size",
            "gradient_accumulation_steps","logging_steps","save_steps","use_vllm","vllm_mode",
            "vllm_server_host","vllm_server_port","vllm_group_port","beta","loss_type","max_steps","seed")
    args=GRPOConfig(output_dir=str(out),**{k:cfg[k] for k in fields},
        report_to="none",save_total_limit=2,save_only_model=False,logging_nan_inf_filter=False,
        chat_template_kwargs={"enable_thinking":False},ddp_find_unused_parameters=False,
        gradient_checkpointing_kwargs={"use_reentrant":False},optim="adamw_torch",
        remove_unused_columns=False,shuffle_dataset=False, data_seed=cfg["seed"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK","0")))
    model=AutoModelForCausalLM.from_pretrained(str(resume) if resume else cfg["model_path"],dtype=torch.bfloat16,
                                             attn_implementation="sdpa",local_files_only=True)
    if rank==0:
        info=metadata(cfg);info.update(dataset_fingerprint=ds._fingerprint,
                    prompt_order_sha256=fingerprint(list(ds["example_id"])),topology="GPU0 vLLM; GPU1-7 DDP",
                    expected_rollout_samples=(cfg["max_steps"]-resume_step)*world*cfg["per_device_train_batch_size"]*cfg["gradient_accumulation_steps"],resume_step=resume_step,total_target_rollout_samples=estimate,model_config_sha256=fingerprint(model.config.to_dict()))
        dump_json(out/"metadata.json",info)
    trainer=GRPOTrainer(model=model,args=args,processing_class=tokenizer,train_dataset=ds,
                        reward_funcs=reward,callbacks=[Metrics(out/"training.jsonl")])
    probe_name,probe=next((n,p) for n,p in model.named_parameters() if ".layers.0." in n and p.ndim==2)
    before=probe.detach().float().cpu().clone()
    start=time.perf_counter()
    result=trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    update=float((probe.detach().float().cpu()-before).abs().max())
    if not math.isfinite(update):raise RuntimeError("Invalid optimizer update")
    dump_json(out/f"reward_stats-rank{rank}.json",dict(cache_hits=engine.hits if engine else 0,cache_misses=engine.misses if engine else 0,
              parameter_probe=probe_name,max_parameter_change=update))
    trainer.save_model(str(out/"final"));tokenizer.save_pretrained(out/"final") if rank==0 else None
    if rank==0:
        dump_json(out/"completed.json",dict(status="completed",global_step=trainer.state.global_step,
                    wall_seconds=time.perf_counter()-start,metrics=result.metrics,parameter_probe=probe_name,max_parameter_change=update))
    if engine is not None: engine.cache.close()
    if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()

if __name__=="__main__": main()
