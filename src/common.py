import hashlib, importlib.metadata, json, os, socket, subprocess, time
from pathlib import Path
import numpy as np
import yaml

def dump_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)

def append_jsonl(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")

def read_jsonl(path):
    # Unicode separators inside JSON strings are not JSONL record boundaries.
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]

def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def config(path):
    base = yaml.safe_load(Path("configs/base.yaml").read_text())
    if path:
        base.update(yaml.safe_load(Path(path).read_text()) or {})
    base["model_path"] = os.environ.get("MODEL_PATH", base["model_path"])
    if "dataset_path" in base:
        base["dataset_path"] = os.environ.get("TOOLRL_PATH", base["dataset_path"])
    return base

def metadata(cfg):
    versions = {}
    for p in ("torch", "transformers", "trl", "vllm", "typesafe-sdk", "accelerate", "datasets", "peft"):
        try: versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError: versions[p] = None
    try: commit = subprocess.check_output(["git","rev-parse","HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception: commit = None
    import torch
    source_files=sorted(Path("src").rglob("*.py"))+sorted(Path("configs").glob("*.yaml"))
    code_sha256=fingerprint({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    provenance=Path(cfg.get("model_path",""))/"export_verified.json"
    return dict(timestamp=time.time(), hostname=socket.gethostname(), git_commit=commit,code_sha256=code_sha256,
                model_provenance=json.loads(provenance.read_text()) if provenance.is_file() else None,
                versions=versions, cuda=torch.version.cuda,
                gpu=[dict(name=torch.cuda.get_device_name(i),
                          memory_bytes=torch.cuda.get_device_properties(i).total_memory)
                     for i in range(torch.cuda.device_count())], config=cfg)

def correlation(x,y):
    from scipy.stats import pearsonr, spearmanr
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    if len(x)<2 or np.ptp(x)==0 or np.ptp(y)==0:
        return dict(pearson=None,spearman=None)
    return dict(pearson=float(pearsonr(x,y).statistic),spearman=float(spearmanr(x,y).statistic))

def bootstrap(x, seed=42, repeats=2000):
    x=np.asarray(x,dtype=float)
    if not len(x): return None
    rng=np.random.default_rng(seed)
    vals=[float(rng.choice(x,len(x),replace=True).mean()) for _ in range(repeats)]
    return dict(mean=float(x.mean()),std=float(x.std(ddof=1)) if len(x)>1 else 0.,
                ci95=[float(v) for v in np.quantile(vals,[.025,.975])],n=len(x))
