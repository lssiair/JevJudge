"""Export the unchanged text backbone so both backends use identical parameter names."""
import json
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from src.common import dump_json,fingerprint
SOURCE=os.environ.get("QWEN_SOURCE_PATH","data/Qwen3.5-4B")
TARGET=os.environ.get("MODEL_PATH","cache/qwen3.5-4b-text")
def main():
    target=Path(TARGET)
    if (target/"export_verified.json").exists():
        print("Verified text checkpoint exists:",TARGET);return
    model=AutoModelForCausalLM.from_pretrained(SOURCE,dtype=torch.bfloat16,local_files_only=True)
    tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
    model.config.eos_token_id=tok.eos_token_id
    model.generation_config.eos_token_id=tok.eos_token_id
    model.save_pretrained(target,max_shard_size="5GB",save_original_format=False)
    tok.save_pretrained(target)
    # Compare every saved tensor exactly against the source-loaded text backbone.
    from safetensors import safe_open
    original=model.state_dict();checked=0
    for path in target.glob("*.safetensors"):
        with safe_open(path,framework="pt",device="cpu") as f:
            for name in f.keys():
                if not torch.equal(f.get_tensor(name),original[name]):
                    raise RuntimeError("Text export changed weights: "+name)
                checked+=1
    dump_json(target/"export_verified.json",dict(source=SOURCE,target=TARGET,
              exact_tensors_checked=checked,model_class=type(model).__name__,
              source_config_sha256=fingerprint(json.loads((Path(SOURCE)/"config.json").read_text()))))
    print("Exact text-backbone export verified:",checked,"tensors",TARGET)
if __name__=="__main__":main()
