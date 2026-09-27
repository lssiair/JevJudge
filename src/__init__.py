"""Jev × GRPO controlled experiment."""
import os
from pathlib import Path
_tmp=Path(os.environ.get("JEV_TMPDIR",str(Path.cwd()/".tmp")))
_tmp.mkdir(parents=True,exist_ok=True)
for _key,_suffix in {
 "TMPDIR":"","HF_DATASETS_CACHE":"datasets-cache","MPLCONFIGDIR":"matplotlib",
 "TRITON_CACHE_DIR":"triton","TORCHINDUCTOR_CACHE_DIR":"torchinductor","VLLM_CACHE_ROOT":"vllm"
}.items():
 os.environ.setdefault(_key,str(_tmp/_suffix))
