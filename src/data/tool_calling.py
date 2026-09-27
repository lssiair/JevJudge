"""Load ToolRL training data without exposing target calls in prompts."""
import json
import os
from pathlib import Path
from datasets import Dataset
import pyarrow.parquet as pq
from src.rewards.tool_reward import parse_calls

DEFAULT = "data/ToolRL/dataset/rlla_4k"

def render_prompt(tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    # Qwen nonthinking mode otherwise inserts an already-closed think block.
    # Let the generated completion supply the full ToolRL format itself.
    suffix = "<think>\n\n</think>\n\n"
    if text.endswith(suffix):
        text = text[:-len(suffix)]
    return text

def load_tool_calling(tokenizer, split="train", limit=None, seed=42, path=None,
                      max_prompt_length=2048):
    dataset_path = Path(path or os.environ.get("TOOLRL_PATH", DEFAULT))
    rows = pq.read_table(dataset_path/f"{split}.parquet").to_pylist()
    output = []
    excluded = 0
    for index, row in enumerate(rows):
        reference = row["reward_model"]["ground_truth"]
        parse_calls(reference)
        prompt = render_prompt(tokenizer, row["prompt"])
        length = len(tokenizer.encode(prompt, add_special_tokens=False))
        if length > max_prompt_length:
            excluded += 1
            continue
        output.append(dict(prompt=prompt, problem=json.dumps(row["prompt"],ensure_ascii=False),
                           reference=reference, example_id=f"{split}:{index}", prompt_tokens=length))
    if not output:
        raise ValueError("No tool-calling examples fit the prompt limit")
    ds = Dataset.from_list(output)
    if split == "train":
        ds = ds.shuffle(seed=seed)
    if limit is not None:
        ds = ds.select(range(min(limit,len(ds))))
    print(f"ToolRL {split}: source={len(rows)}, overlength_excluded={excluded}, selected={len(ds)}",flush=True)
    return ds
