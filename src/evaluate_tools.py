"""Offline ToolRL held-out evaluation (not the BFCL benchmark)."""
import argparse
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from src.common import append_jsonl, dump_json, bootstrap
from src.data.tool_calling import load_tool_calling
from src.rewards.tool_reward import tool_scores, parse_calls

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--name",required=True)
    p.add_argument("--training-step",required=True,type=int)
    args=p.parse_args()
    path=Path("results")/(args.name+".tool-eval.jsonl")
    if path.exists():raise FileExistsError(path)
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    ds=load_tool_calling(tok,split="test",max_prompt_length=2048)
    from vllm import LLM, SamplingParams
    llm=LLM(model=args.model,dtype="bfloat16",tensor_parallel_size=1,
             gpu_memory_utilization=.85,max_model_len=3072,enforce_eager=True,seed=42)
    generated=llm.generate(list(ds["prompt"]),SamplingParams(temperature=0,top_p=1,max_tokens=1024,seed=42))
    rows=[]
    for record,result in zip(ds,generated):
        out=result.outputs[0]
        scores=tool_scores(out.text,record["reference"])
        requires_call=bool(parse_calls(record["reference"]))
        try:
            predicted=parse_calls(out.text)
            parse_valid=True
        except (ValueError,TypeError):
            predicted=None
            parse_valid=False
        rows.append(dict(example_id=record["example_id"],prompt=record["problem"],reference=record["reference"],
                         completion=out.text,completion_tokens=len(out.token_ids),finish_reason=out.finish_reason,
                         requires_call=requires_call,parse_valid=parse_valid,
                         correctly_abstained=(not requires_call and parse_valid and not predicted),
                         **scores))
    for row in rows:append_jsonl(path,row)
    call=[r for r in rows if r["requires_call"]]
    no_call=[r for r in rows if not r["requires_call"]]
    result=dict(name=args.name,model=args.model,training_step=args.training_step,n=len(rows),
                exact_match=bootstrap([r["exact_match"] for r in rows]),
                call_exact_match=bootstrap([r["exact_match"] for r in call]),
                format_accuracy=bootstrap([r["format_reward"] for r in rows]),
                no_call_abstention=bootstrap([float(r["correctly_abstained"]) for r in no_call]),
                call_count=len(call),no_call_count=len(no_call),
                mean_length=float(np.mean([r["completion_tokens"] for r in rows])),
                truncated_count=sum(r["finish_reason"]=="length" for r in rows),
                mean_reference_reward=float(np.mean([r["reward"] for r in rows])),
                dataset_fingerprint=ds._fingerprint,
                evaluation=dict(split="ToolRL test",max_prompt_length=2048,max_tokens=1024,
                                temperature=0,seed=42,jev_api_calls=0,bfcl=False))
    dump_json(Path("results")/(args.name+".tool-metrics.json"),result)
    print(result,flush=True)

if __name__=="__main__":main()
