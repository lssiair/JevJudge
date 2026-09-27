import os
from pathlib import Path
from datasets import load_dataset
from src.rewards.reference_reward import extract_answer

DEFAULT = "data/gsm8k"
INSTRUCTION = "Solve the problem. Give your final numerical answer on the last line as #### <answer>."

def format_record(row, index):
    ref = extract_answer(row["answer"])
    if ref is None:
        raise ValueError(f"Invalid GSM8K reference at {index}")
    return {
        "prompt": [{"role": "user", "content": row["question"] + "\n\n" + INSTRUCTION}],
        "problem": row["question"], "reference": str(ref), "example_id": str(index),
    }

def load_gsm8k(split="train", limit=None, seed=42, path=None):
    root = Path(path or os.environ.get("GSM8K_PATH", DEFAULT))
    files = sorted((root / "main").glob(f"{split}*.parquet"))
    if not files:
        raise FileNotFoundError(f"No local GSM8K {split} parquet files in {root}/main")
    ds = load_dataset("parquet", data_files={split: [str(p) for p in files]}, split=split)
    ds = ds.map(format_record, with_indices=True, remove_columns=ds.column_names)
    if split == "train":
        ds = ds.shuffle(seed=seed)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds
