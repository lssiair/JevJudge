import gzip, json, os
from pathlib import Path
DIMENSIONS = ("helpfulness", "correctness", "coherence")

def load_helpsteer2(limit=None, path=None):
    root = Path(path or os.environ.get("HELPSTEER2_PATH", "data/HelpSteer2"))
    with gzip.open(root / "validation.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    # Retain complete prompt groups for pairwise metrics.
    if limit is not None:
        prompts = list(dict.fromkeys(r["prompt"] for r in rows))[:limit]
        rows = [r for r in rows if r["prompt"] in set(prompts)]
    for row in rows:
        if not all(k in row for k in ("prompt", "response", *DIMENSIONS)):
            raise ValueError("Unexpected HelpSteer2 schema")
    return rows
