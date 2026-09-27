#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
nvidia-smi
python - <<'PY'
import importlib
from src.common import metadata,dump_json,config
from src.credentials import load_credentials
for name in ("torch","transformers","trl","vllm","typesafe_sdk"):
 module=importlib.import_module(name)
 print(name,getattr(module,"__version__","import OK"))
load_credentials()
print("API credential loaded (redacted)")
dump_json("results/environment.json",metadata(config(None)))
PY
python -m pytest -q
