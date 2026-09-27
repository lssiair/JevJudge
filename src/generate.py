"""Deterministic held-out generation with the same settings for every arm."""
import argparse,time
from pathlib import Path
from src.common import append_jsonl,dump_json,metadata,config
from src.data.gsm8k import load_gsm8k

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True);p.add_argument("--output",required=True)
    p.add_argument("--limit",type=int);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--max-tokens",type=int,default=512);p.add_argument("--n",type=int,default=1)
    p.add_argument("--split",choices=["train","test"],default="test")
    a=p.parse_args()
    if Path(a.output).exists(): raise FileExistsError("Choose a new output; never mix evaluation runs")
    from vllm import LLM,SamplingParams
    ds=load_gsm8k(split=a.split,limit=a.limit,seed=a.seed)
    llm=LLM(model=a.model,dtype="bfloat16",tensor_parallel_size=1,gpu_memory_utilization=.85,
            max_model_len=2048,enable_prefix_caching=True,seed=a.seed,enforce_eager=True)
    tok=llm.get_tokenizer()
    prompts=[tok.apply_chat_template(r["prompt"],tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in ds]
    params=SamplingParams(n=a.n,temperature=0.0 if a.n==1 else .9,top_p=1.0 if a.n==1 else .95,
                          max_tokens=a.max_tokens,seed=a.seed)
    start=time.perf_counter()
    generated=llm.generate(prompts,params)
    for row,result in zip(ds,generated):
        for i,o in enumerate(result.outputs):
            append_jsonl(a.output,dict(example_id=row["example_id"],prompt=row["problem"],reference=row["reference"],
                completion=o.text,completion_tokens=len(o.token_ids),finish_reason=o.finish_reason,sample_index=i))
    dump_json(a.output+".meta.json",dict(**metadata(vars(a)),wall_seconds=time.perf_counter()-start,
                dataset_fingerprint=ds._fingerprint,n_examples=len(ds),split=a.split))

if __name__=="__main__":main()
